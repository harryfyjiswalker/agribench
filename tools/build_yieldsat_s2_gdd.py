"""Thermal-time-aligned Sentinel-2 input `s2-gdd` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

Same idea as `s2-aligned` (tools/build_yieldsat_s2_aligned.py), but the time axis is accumulated growing degree
days (GDD) since the field's sowing date instead of calendar days. Crop development follows heat accumulation
more closely than the calendar, so a fixed GDD step should mean the same growth stage across years and regions
(Nyborg et al. 2022, thermal positional encoding; Franch et al. 2026, GDD-normalised S2 yield models).

Daily GDD from the field's own daily ERA5 weather file (Raw/<country>/<country>/<field>/weather/<field>.csv):
    GDD = max(0, (clip(Tmax, Tbase, Tcap) + clip(Tmin, Tbase, Tcap)) / 2 - Tbase)
with crop base temperatures from the agronomy literature: wheat 0 °C, rapeseed 5 °C, corn and soybean 10 °C, all
capped at 30 °C. Thermal time of a date = GDD summed from sowing to that date (negative before sowing). Missing
days are linearly interpolated. Only sowing date and weather are used, never harvest date or yield.

Grid: from -50 °C·d in STEPS_PER_SEASON equal steps up to the subset's 95th percentile of the last observation's
thermal time, the step rounded to 25 °C·d (label-free, like s2-aligned). Per pixel and band, linear interpolation
between valid observations; NaN outside the observed range.

    python -m tools.build_yieldsat_s2_gdd --subset GER-W      (CPU job; one subset per job)

Writes features/yieldsat/s2-gdd/<subset>.npy float32 (pixels, steps x 12) + <subset>.json; names <band>_g<gdd>.
"""
import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd

from tools import yieldsat as ys
from tools.build_yieldsat_s2_aligned import interpolate, sowing_dates
from tools.embed_yieldsat_galileo import slot_months

BASE_CAP = {"wheat": (0.0, 30.0), "rapeseed": (5.0, 30.0), "corn": (10.0, 30.0), "soybean": (10.0, 30.0)}
GRID_START = -50.0
STEPS_PER_SEASON = 16


def daily_gdd(field, country, base, cap):
    """pd.Series of daily GDD indexed by date, from the field's daily ERA5 file (K -> °C)."""
    path = ys.DATA / "Raw" / country / country / field / "weather" / f"{field}.csv"
    w = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    t = w[["Temp_max", "Temp_min"]].interpolate(limit_area="inside") - 273.15
    gdd = ((t.Temp_max.clip(base, cap) + t.Temp_min.clip(base, cap)) / 2 - base).clip(lower=0)
    return gdd.dropna()


def thermal_time(dates, sow, gdd):
    """Thermal time (°C·d since sowing) for each date; NaN if outside the weather record."""
    cum = gdd.cumsum()
    days = gdd.index.date
    if not days[0] <= sow <= days[-1]:
        return np.full(len(dates), np.nan)
    at_sow = cum.iloc[np.searchsorted(days, sow)] - gdd.iloc[np.searchsorted(days, sow)]
    out = []
    for d in dates:
        if d is None or not days[0] <= d <= days[-1]:
            out.append(np.nan)
            continue
        k = np.searchsorted(days, d)
        out.append(float(cum.iloc[k] - gdd.iloc[k] - at_sow))  # GDD of the days before d
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "s2-gdd"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    t0 = time.time()
    crop = {v: k for k, v in ys.CROPS.items()}[args.subset.split("-")[1]]
    base, cap = BASE_CAP[crop]
    names = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in ys.S2_BANDS] for s in range(ys.N_STEPS)])
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    _, _, dates, country = slot_months(args.subset, fields)
    sow = sowing_dates()

    tt = np.full((len(fields), ys.N_STEPS), np.nan)
    no_weather = []
    for i, f in enumerate(fields):
        tt[i] = thermal_time(dates[f], sow[f], daily_gdd(f, country, base, cap))
        if not np.isfinite(tt[i]).any():
            no_weather.append(f)
        # Thermal time stalls on cold days; a tiny offset per slot keeps the axis strictly increasing.
        tt[i] += 1e-3 * np.arange(ys.N_STEPS)
    last = np.nanmax(tt, axis=1)
    p95 = float(np.nanpercentile(last, 95))
    step = max(25.0, 25.0 * round((p95 - GRID_START) / STEPS_PER_SEASON / 25.0))
    grid = np.arange(GRID_START, p95 + step, step)
    feature_names = [f"{b}_g{g:+05.0f}" for g in grid for b in ys.S2_BANDS]

    tmp = out_dir / f".{args.subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(fidx), len(feature_names)))
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        i = fidx[a]
        slots = np.flatnonzero(np.isfinite(tt[i]))
        if not len(slots):
            out[a:b] = np.nan
            continue
        values = np.asarray(cache[a:b][:, cols[slots]], np.float32)
        out[a:b] = interpolate(values, tt[i][slots], grid).reshape(b - a, -1)
    out.flush()
    del out
    tmp.rename(target)

    arr = np.load(target, mmap_mode="r")
    filled = np.isfinite(arr[:, ys.S2_BANDS.index("B04")::len(ys.S2_BANDS)])
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)), "feature_names": feature_names,
            "crop": crop, "base_temp_c": base, "cap_temp_c": cap, "grid_gdd_since_sowing": grid.tolist(),
            "step_gdd": step, "share_filled_per_step": [round(float(x), 3) for x in filled.mean(0)],
            "fields_without_weather_cover": len(no_weather), "last_obs_gdd_p95": p95,
            "method": "per pixel and band, linear interpolation between valid observations onto growing degree days "
                      "since sowing (daily ERA5 from Raw weather files); NaN outside the observed range",
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "crop", "step_gdd", "grid_gdd_since_sowing",
                                           "share_filled_per_step", "fields_without_weather_cover", "seconds")}),
          flush=True)


if __name__ == "__main__":
    main()
