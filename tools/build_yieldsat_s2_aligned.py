"""Season-aligned Sentinel-2 input `s2-aligned` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

Why: the raw `s2` input stores each field's observations in 24 slots that are not tied to the calendar or the
season (1, note 5). The same slot can be a month before sowing in one year and two months after it in another, so a
model that learns "slot 12" from some years can misread it in a new year. `s2-aligned` puts every pixel's series on
one fixed time axis instead: days since the field's sowing date.

How, per pixel and band: linear interpolation between the pixel's own valid observations onto the grid
GRID_START, GRID_START + STEP, ... days since sowing. Grid points before the first or after the last valid
observation stay NaN (no extrapolation). Only the sowing date is used (known at the start of the season); the
harvest date and yield are not. The grid's end is the subset's 95th percentile of the last observation day,
rounded up to STEP, so it covers the season without long all-NaN tails. It is a feature definition that uses no
labels, computed on all fields of the subset.

    python -m tools.build_yieldsat_s2_aligned --subset GER-W      (CPU job; one subset per array task)

Writes features/yieldsat/s2-aligned/<subset>.npy float32 (pixels, steps x 12), rows aligned with v1/<subset>.npy,
plus <subset>.json. Feature names are <band>_d<day>, e.g. B04_d+045.
"""
import argparse
import datetime as dt
import glob
import json
import time

import numpy as np

from tools import yieldsat as ys
from tools.embed_yieldsat_galileo import slot_months

STEP = 15          # days between grid points
GRID_START = -15   # first grid point, days since sowing (bare soil just before sowing)
MAX_END = 330      # cap on the grid's end, days since sowing


def sowing_dates():
    out = {}
    for path in glob.glob(str(ys.DATA / "Raw" / "*" / "*" / "*" / "metadata-*.json")):
        with open(path) as f:
            m = json.load(f)
        out[m["field_shared_name"]] = dt.datetime.strptime(m["seeding_date"], "%d.%m.%Y").date()
    return out


def fill(values, days, reverse=False):
    """Carry each pixel's last valid value (and its day) along the slot axis; NaN where none yet."""
    v, d = values.copy(), np.where(np.isfinite(values), days[None, :, None], np.nan)
    order = range(v.shape[1] - 2, -1, -1) if reverse else range(1, v.shape[1])
    for s in order:
        src = s + 1 if reverse else s - 1
        gap = ~np.isfinite(v[:, s])
        v[:, s][gap], d[:, s][gap] = v[:, src][gap], d[:, src][gap]
    return v, d


def interpolate(values, days, grid):
    """values (pixels, slots, bands) with NaN gaps, days (slots,) increasing -> (pixels, len(grid), bands)."""
    fv, fd = fill(values, days)
    bv, bd = fill(values, days, reverse=True)
    out = np.full((values.shape[0], len(grid), values.shape[2]), np.nan, np.float32)
    for k, g in enumerate(grid):
        before = np.searchsorted(days, g, side="right") - 1   # last slot on or before g
        after = np.searchsorted(days, g, side="left")         # first slot on or after g
        if before < 0 or after >= len(days):
            continue
        pv, pd, nv, nd = fv[:, before], fd[:, before], bv[:, after], bd[:, after]
        span = nd - pd
        w = np.divide(g - pd, span, out=np.zeros_like(span), where=span > 0)
        out[:, k] = pv + w * (nv - pv)  # NaN wherever there is no valid observation on one side
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "s2-aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    t0 = time.time()
    names = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in ys.S2_BANDS] for s in range(ys.N_STEPS)])  # (24, 12)
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    _, exists, dates, _ = slot_months(args.subset, fields)
    sow = sowing_dates()

    # Days since sowing per field and slot (NaN for empty slots).
    rel = np.full((len(fields), ys.N_STEPS), np.nan)
    for i, f in enumerate(fields):
        rel[i] = [(d - sow[f]).days if d is not None else np.nan for d in dates[f]]
        ok = rel[i][np.isfinite(rel[i])]
        if np.any(np.diff(ok) <= 0):
            raise ValueError(f"{f}: slot dates are not strictly increasing")
    last = np.nanmax(rel, axis=1)
    end = int(min(MAX_END, STEP * np.ceil(np.nanpercentile(last, 95) / STEP)))
    grid = np.arange(GRID_START, end + 1, STEP)
    feature_names = [f"{b}_d{g:+04d}" for g in grid for b in ys.S2_BANDS]

    tmp = out_dir / f".{args.subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(fidx), len(feature_names)))
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        i = fidx[a]
        slots = np.flatnonzero(np.isfinite(rel[i]))
        values = np.asarray(cache[a:b][:, cols[slots]], np.float32)  # (pixels, observed slots, 12)
        out[a:b] = interpolate(values, rel[i][slots], grid).reshape(b - a, -1)
    out.flush()
    del out
    tmp.rename(target)

    arr = np.load(target, mmap_mode="r")
    filled = np.isfinite(arr[:, ys.S2_BANDS.index("B04")::len(ys.S2_BANDS)])
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)), "feature_names": feature_names,
            "grid_days_since_sowing": grid.tolist(), "step_days": STEP,
            "share_filled_per_step": [round(float(x), 3) for x in filled.mean(0)],
            "fields_last_obs_day_p95": float(np.nanpercentile(last, 95)),
            "method": "per pixel and band, linear interpolation between valid observations onto days since sowing; "
                      "NaN outside the observed range",
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "grid_days_since_sowing", "share_filled_per_step",
                                           "seconds")}), flush=True)


if __name__ == "__main__":
    main()
