"""Phenology-aligned weather input `weather-pheno` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

Why: transfer to unseen years and regions is poor for every imagery representation, possibly because nothing tells
the model the year's yield level. `weather-pheno` summarises each field's season weather in agronomic terms, keyed
to its sowing date and to crop stages estimated from thermal time, so year-level heat, frost and drought enter as
explicit features. `s2-adm` already holds weather, but as calendar-month slots that are not tied to the season.

Daily weather (`daily_weather`, shared with the WOFOST builders): the field's own ERA5 file
Raw/<country>/<country>/<field>/weather/<field>.csv (Temp_max/Temp_min in K, Total_prec in m). Reference
evapotranspiration ET0 by Hargreaves-Samani (FAO-56 eq. 52) from Tmax, Tmin and extraterrestrial radiation at the
field's centroid latitude, for every country alike (only Germany and Uruguay have a radiation column, of unclear
units). Gaps inside the record are interpolated linearly; precipitation gaps are 0.

**Leakage guard.** Every weather file runs from 60 days before sowing to exactly 60 days after harvest (checked on
all 2,173 fields), so the record's end would reveal the harvest date. Only days up to sowing + `season_cap_days`
are read, where the cap is one constant per subset: the shortest record in that subset (141 days in ARG-S up to
368 in GER-R). The cap is the same for every field of a subset, so it carries no per-field information. Harvest
date and yield are never used.

Crop stages, from thermal time since sowing (same GDD definition and base temperatures as `s2-gdd`): flowering
at TT_FLOWER and maturity at TT_MATURITY (crop constants from the agronomy literature, below). Windows:
    pre       sowing - 60 days .. sowing              (antecedent rain, soil water at sowing)
    early     0 .. 0.5 TT_FLOWER                      (establishment, early vegetative)
    late      0.5 .. 0.9 TT_FLOWER                    (late vegetative)
    flower    0.9 .. 1.1 TT_FLOWER                    (flowering)
    fill      1.1 TT_FLOWER .. TT_MATURITY            (grain fill)
    season    0 .. TT_MATURITY
Every window is truncated at the cap (the same for every field of a subset).

Per window: length (days), mean Tmean and Tmax, heat degree-days sum(max(0, Tmax - T_HEAT)), frost days
(Tmin < 0 °C), hard-frost days (Tmin < -5 °C), rainfall, ET0, rainfall - ET0, the longest dry spell (days with
< 1 mm), and the water-stress index 1 - AET / ETc of a one-layer soil water balance: available water capacity
150 mm (generic loam, about 1 m rooting), starting half full 60 days before sowing, ETc = Kc * ET0 with FAO-56-style
Kc rising from 0.4 to 1.15 between 0.25 and 0.9 TT_FLOWER and falling to 0.4 at maturity (0.3 before sowing), AET =
Ks * ETc with Ks = min(1, SW / (0.5 * AWC)). Plus: sowing day of year (sin, cos), days from sowing to estimated
flowering and maturity (the cap if not reached), soil water at sowing and at flowering, GDD in the first 30/60/90
days, and the minimum Tmin of the season.

    python -m tools.build_yieldsat_weather_pheno --subset GER-W      (CPU job, a minute per subset)

Writes features/yieldsat/weather-pheno/<subset>.npy float32 (pixels, features), field values repeated on every
pixel of the field, plus <subset>.json.
"""
import argparse
import datetime as dt
import glob
import json
import time

import numpy as np
import pandas as pd

from tools import yieldsat as ys
from tools.build_yieldsat_s2_gdd import BASE_CAP

# Thermal time (°C·d since sowing, base temperatures of BASE_CAP) at flowering and physiological maturity.
# Approximate literature values for mid-season cultivars: wheat anthesis ~1500 and maturity ~2300 (base 0 °C, no
# vernalisation or photoperiod term); maize silking ~800, black layer ~1500 (base 10 °C); soybean R2 ~700, R7 ~1500
# (base 10 °C); winter oilseed rape flowering ~850, maturity ~1700 (base 5 °C). Sowing-to-harvest thermal time in
# YieldSAT is about 300-500 °C·d above these maturity values in every subset (harvest follows dry-down), which is the
# expected order; it is a check only, harvest dates are not used.
TT_FLOWER = {"wheat": 1500.0, "corn": 800.0, "soybean": 700.0, "rapeseed": 850.0}
TT_MATURITY = {"wheat": 2300.0, "corn": 1500.0, "soybean": 1500.0, "rapeseed": 1700.0}
# Heat-stress thresholds on daily Tmax (°C) around flowering (Porter & Gawith 1999; Schlenker & Roberts 2009;
# Siebers et al. 2015; Morrison & Stewart 2002 for rapeseed).
T_HEAT = {"wheat": 32.0, "corn": 35.0, "soybean": 35.0, "rapeseed": 29.0}
AWC_MM = 150.0
PRE_DAYS = 60
KC = dict(pre=0.3, ini=0.4, mid=1.15, end=0.4)
WINDOWS = ["pre", "early", "late", "flower", "fill", "season"]
WINDOW_STATS = ["days", "tmean", "tmax", "heat_dd", "frost_days", "hardfrost_days", "rain", "et0",
                "rain_minus_et0", "dry_spell", "water_stress"]
PRE_STATS = ["tmean", "rain", "et0", "rain_minus_et0"]


def field_info():
    """field -> (country, crop, sowing date, centroid latitude) from the Raw metadata (never harvest or yield)."""
    out = {}
    for path in glob.glob(str(ys.DATA / "Raw" / "*" / "*" / "*" / "metadata-*.json")):
        with open(path) as f:
            m = json.load(f)
        out[m["field_shared_name"]] = (m["adm_units"]["country"], m["crop"],
                                       dt.datetime.strptime(m["seeding_date"], "%d.%m.%Y").date(),
                                       float(m["centroid_latitude_wgs84"]))
    return out


def extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 eq. 21: daily extraterrestrial radiation Ra (MJ m-2 d-1)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def daily_weather(field, country, lat, first, last):
    """Daily DataFrame indexed by date, first..last inclusive: tmax, tmin, tmean (°C), prec (mm), ra, rs (MJ m-2),
    et0 (mm). Raises if the record does not cover first..last."""
    path = ys.DATA / "Raw" / country / country / field / "weather" / f"{field}.csv"
    w = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    w = w.loc[str(first):str(last)]
    days = pd.date_range(str(first), str(last), freq="D")
    if len(w) == 0 or w.index[0].date() > first or w.index[-1].date() < last:
        raise ValueError(f"{field}: weather record does not cover {first}..{last}")
    w = w.reindex(days)
    out = pd.DataFrame(index=days)
    for name, col in (("tmax", "Temp_max"), ("tmin", "Temp_min"), ("tmean", "Temp_mean")):
        out[name] = w[col].interpolate(limit_direction="both") - 273.15
    out["prec"] = (w["Total_prec"].fillna(0).clip(lower=0) * 1000.0)
    out["ra"] = extraterrestrial_radiation(lat, days.dayofyear.to_numpy())
    dtr = (out.tmax - out.tmin).clip(lower=0)
    out["rs"] = 0.16 * np.sqrt(dtr) * out.ra                     # Hargreaves radiation estimate (FAO-56 eq. 50)
    out["et0"] = (0.0023 * (out.tmean + 17.8) * np.sqrt(dtr) * out.ra * 0.408).clip(lower=0)  # FAO-56 eq. 52
    return out


def gdd(w, crop):
    base, cap = BASE_CAP[crop]
    return ((w.tmax.clip(base, cap) + w.tmin.clip(base, cap)) / 2 - base).clip(lower=0)


def kc_curve(tt, crop):
    """Crop coefficient from thermal time since sowing (NaN tt = before sowing)."""
    fl, mat = TT_FLOWER[crop], TT_MATURITY[crop]
    a, b, c = 0.25 * fl, 0.9 * fl, 1.1 * fl + 0.5 * (mat - 1.1 * fl)
    kc = np.interp(tt, [0, a, b, c, mat], [KC["ini"], KC["ini"], KC["mid"], KC["mid"], KC["end"]])
    kc[tt > mat] = KC["end"]
    kc[~np.isfinite(tt)] = KC["pre"]
    return kc


def water_balance(prec, et0, kc):
    """One-layer bucket: returns soil water at the end of each day (mm), ETc and AET (mm/day)."""
    sw = 0.5 * AWC_MM
    sws, etc, aet = np.empty(len(prec)), kc * et0, np.empty(len(prec))
    for k in range(len(prec)):
        ks = min(1.0, sw / (0.5 * AWC_MM))
        aet[k] = ks * etc[k]
        sw = min(AWC_MM, max(0.0, sw + prec[k] - aet[k]))
        sws[k] = sw
    return sws, etc, aet


def longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def field_features(w, sow, crop, cap_days):
    """Feature dict for one field; w covers sow - PRE_DAYS .. sow + cap_days."""
    sow_ts = pd.Timestamp(sow)
    after = w.index >= sow_ts
    g = gdd(w, crop).to_numpy()
    tt = np.full(len(w), np.nan)
    tt[after] = np.cumsum(g[after]) - g[after]              # thermal time at the start of each day
    kc = kc_curve(tt, crop)
    sw, etc, aet = water_balance(w.prec.to_numpy(), w.et0.to_numpy(), kc)
    fl, mat = TT_FLOWER[crop], TT_MATURITY[crop]
    ttz = np.where(np.isfinite(tt), tt, -1.0)
    masks = {"pre": ~after, "early": after & (ttz < 0.5 * fl), "late": (ttz >= 0.5 * fl) & (ttz < 0.9 * fl),
             "flower": (ttz >= 0.9 * fl) & (ttz < 1.1 * fl), "fill": (ttz >= 1.1 * fl) & (ttz < mat),
             "season": after & (ttz < mat)}
    feats = {}
    for name in WINDOWS:
        m = masks[name]
        d = w[m]
        stats = {"days": float(m.sum())}
        if m.any():
            stats.update(tmean=d.tmean.mean(), tmax=d.tmax.mean(), rain=d.prec.sum(), et0=d.et0.sum(),
                         heat_dd=(d.tmax - T_HEAT[crop]).clip(lower=0).sum(), frost_days=float((d.tmin < 0).sum()),
                         hardfrost_days=float((d.tmin < -5).sum()), dry_spell=float(longest_run(d.prec < 1.0)),
                         water_stress=1 - aet[m].sum() / etc[m].sum() if etc[m].sum() > 0 else 0.0)
            stats["rain_minus_et0"] = stats["rain"] - stats["et0"]
        for s in (PRE_STATS if name == "pre" else WINDOW_STATS):
            feats[f"{name}_{s}"] = float(stats.get(s, np.nan))
    day = np.arange(len(w)) - PRE_DAYS                       # days since sowing
    reached = lambda thr: float(day[np.flatnonzero(ttz >= thr)[0]]) if (ttz >= thr).any() else float(cap_days)
    feats["days_to_flower"] = reached(fl)
    feats["days_to_maturity"] = reached(mat)
    feats["sw_at_sowing"] = float(sw[PRE_DAYS - 1])
    k_fl = np.flatnonzero(ttz >= fl)
    feats["sw_at_flower"] = float(sw[k_fl[0]]) if len(k_fl) else float(sw[-1])
    for n in (30, 60, 90):
        feats[f"gdd_first{n}d"] = float(g[PRE_DAYS:PRE_DAYS + min(n, cap_days + 1)].sum())
    feats["season_tmin_min"] = float(w.tmin[after & (ttz < mat)].min())
    doy = sow.timetuple().tm_yday
    feats["sow_doy_sin"], feats["sow_doy_cos"] = float(np.sin(2 * np.pi * doy / 365.25)), float(np.cos(2 * np.pi * doy / 365.25))
    return feats


def record_days(field, country, sow):
    """Days from sowing to the last complete row of the weather record (used only to set the subset-wide cap)."""
    path = ys.DATA / "Raw" / country / country / field / "weather" / f"{field}.csv"
    w = pd.read_csv(path, parse_dates=["Date"], usecols=["Date", "Temp_max"]).dropna()
    return (w.Date.max().date() - sow).days


def subset_fields(subset):
    meta = np.load(ys.FEATURES / "v1" / f"{subset}-meta.npz")
    return [str(f) for f in meta["fields"]], meta["field_idx"]


def season_cap(subset, fields, info):
    """Subset-wide cap (days after sowing) = the shortest weather record in the subset."""
    return int(min(record_days(f, info[f][0], info[f][2]) for f in fields))


def write_dense(name, subset, values, fidx, feature_names, extra):
    """Repeat field rows on pixels and write features/yieldsat/<name>/<subset>.npy + .json (never overwrite)."""
    out_dir = ys.FEATURES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    tmp = out_dir / f".{subset}.npy.tmp.npy"
    np.save(tmp, np.asarray(values, np.float32)[fidx])
    tmp.rename(target)
    info = {"subset": subset, "kind": "dense", "rows": int(len(fidx)), "fields": int(len(values)),
            "feature_names": feature_names, **extra,
            "share_finite_per_feature": [round(float(x), 3) for x in np.isfinite(values).mean(0)],
            "aligned_with": f"v1/{subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz")}
    (out_dir / f"{subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", nargs="+", choices=ys.SUBSETS, required=True)
    args = parser.parse_args()
    info = field_info()
    for subset in args.subset:
        t0 = time.time()
        crop = {v: k for k, v in ys.CROPS.items()}[subset.split("-")[1]]
        fields, fidx = subset_fields(subset)
        cap = season_cap(subset, fields, info)
        rows = []
        for f in fields:
            country, _, sow, lat = info[f]
            w = daily_weather(f, country, lat, sow - dt.timedelta(days=PRE_DAYS), sow + dt.timedelta(days=cap))
            rows.append(field_features(w, sow, crop, cap))
        names = list(rows[0])
        values = np.array([[r[n] for n in names] for r in rows], np.float64)
        extra = {"crop": crop, "season_cap_days": cap, "tt_flower": TT_FLOWER[crop], "tt_maturity": TT_MATURITY[crop],
                 "t_heat_c": T_HEAT[crop], "base_cap_c": BASE_CAP[crop], "awc_mm": AWC_MM, "kc": KC,
                 "pre_days": PRE_DAYS, "et0": "Hargreaves-Samani (FAO-56 eq. 52), Ra from centroid latitude",
                 "share_fields_flowering_by_cap": round(float((values[:, names.index("days_to_flower")] < cap).mean()), 3),
                 "share_fields_maturity_by_cap": round(float((values[:, names.index("days_to_maturity")] < cap).mean()), 3),
                 "method": "per field, agronomic weather summaries in thermal-time crop-stage windows from sowing "
                           "(daily ERA5 from Raw weather files, read only up to the subset-wide cap); repeated per pixel",
                 "seconds": round(time.time() - t0, 1)}
        out = write_dense("weather-pheno", subset, values, fidx, names, extra)
        print(json.dumps({k: out[k] for k in ("subset", "fields", "season_cap_days", "share_fields_flowering_by_cap",
                                              "share_fields_maturity_by_cap", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
