"""Crop model with assimilated satellite LAI, input `wofost-da` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

The classic hybrid: WOFOST 7.2 water-limited (`Wofost72_WLP_CWB`, set-up as `wofost`, tools/yieldsat_wofost.py)
corrected during the season by the field's satellite LAI with an ensemble Kalman filter (Evensen 2003; de Wit &
van Diepen 2007, the PCSE EnKF tutorial). Per field:

1. Observations: the field median over pixels of an existing per-pixel LAI cache on the `s2-aligned` grid (days
   since sowing -15, 0, 15, ..., columns lai_d±NNN): `arc` (ARC retrieval) -> input `wofost-da`, or `biophys` (SNAP
   biophysical processor) -> input `wofost-da-snap`. Grid days in 1..cap with at least MIN_PIXELS valid pixels.
   Observation error SD = max(0.3, 0.15 LAI) (typical LAI retrieval uncertainty).
2. Ensemble: N_MEMBERS WOFOST runs with perturbed, LAI-relevant parameters, drawn with a seed per field:
   TDWI (initial dry weight) x lognormal(0, 0.3), SPAN (leaf life span) x N(1, 0.1), SLATB (specific leaf area)
   and AMAXTB (max leaf photosynthesis) x N(1, 0.15), TSUM1 x N(1, 0.05) (phenology), and the initial available
   water WAV x U(0.5, 1.5).
3. Filter: at each observation day every member is run up to that day; members whose crop has emerged and not yet
   matured get the stochastic EnKF update of their LAI state (perturbed observations):
       LAI_i <- LAI_i + K (y + e_i - LAI_i),  K = var(LAI) / (var(LAI) + R),  e_i ~ N(0, R)
   applied with PCSE's set_variable("LAI"), which rescales the leaf biomass consistently. Then the season is
   run to the end (maturity or the subset cap).
4. Features (field values repeated per pixel): ensemble mean and SD of grain yield (TWSO), mean total biomass,
   max LAI, LAI at flowering, cumulative transpiration, simulated days to flowering and maturity, the number of
   observations assimilated, and the mean and mean absolute innovation (observed minus forecast LAI, before the
   update), which say how far the satellite canopy departs from the weather-driven one.

    python -m tools.build_yieldsat_wofost_da --lai arc --submit ARG-C ...        # chunk array -> finalize
    python -m tools.build_yieldsat_wofost_da --lai biophys --stage run --subset GER-W --chunk 0

Writes features/yieldsat/<wofost-da | wofost-da-snap>/<subset>.npy float32 (pixels, 12) + <subset>.json.
"""
import datetime as dt
import json
import re

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_wofost import (context, crop_of, finalize, parser, run_chunk, season_cap_of, setup,
                                         submit, task_arg)

LAI_SOURCES = {"arc": "wofost-da", "biophys": "wofost-da-snap"}
N_MEMBERS = 32
MIN_PIXELS = 5
FIELDS_PER_CHUNK = 25
OBS_SD_FLOOR, OBS_SD_REL = 0.3, 0.15


def lai_series(source, subset, fidx):
    """field index -> (days since sowing, field-median LAI) from features/yieldsat/<source>/<subset>.npy."""
    names = json.loads((ys.FEATURES / source / f"{subset}.json").read_text())["feature_names"]
    cols = [k for k, n in enumerate(names) if re.fullmatch(r"lai_d[+-]\d{3}", n)]
    days = np.array([int(names[k][5:]) for k in cols])
    cache = np.load(ys.FEATURES / source / f"{subset}.npy", mmap_mode="r")
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    out = {}
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        v = np.asarray(cache[a:b][:, cols], np.float64)
        ok = np.isfinite(v).sum(0) >= MIN_PIXELS
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(v), v, np.nan), axis=0)
        out[int(fidx[a])] = (days[ok], med[ok])
    return out


def da_context(source):
    def ctx_fn(subset):
        ctx = context(subset)
        ctx["lai"] = lai_series(source, subset, ctx["fidx"])
        return ctx
    return ctx_fn


def perturbed_overrides(fs, rng):
    p = fs.parameters()
    scale_table = lambda tb, f: [x if k % 2 == 0 else x * f for k, x in enumerate(tb)]
    return {"TDWI": float(p["TDWI"]) * float(np.exp(rng.normal(0, 0.3))),
            "SPAN": float(p["SPAN"]) * float(rng.normal(1, 0.1)),
            "SLATB": scale_table(list(p["SLATB"]), float(np.clip(rng.normal(1, 0.15), 0.5, 1.5))),
            "AMAXTB": scale_table(list(p["AMAXTB"]), float(np.clip(rng.normal(1, 0.15), 0.5, 1.5))),
            "TSUM1": float(p["TSUM1"]) * float(rng.normal(1, 0.05)),
            "WAV": float(np.clip(fs.wav * rng.uniform(0.5, 1.5), 0, 100))}


def da_field(ctx, i):
    from tools import yieldsat_wofost as yw
    fs, sow = setup(ctx, i)
    rng = np.random.default_rng([20261109, i])
    members = [fs.model("WLP", perturbed_overrides(fs, rng)) for _ in range(N_MEMBERS)]
    days, lai_obs = ctx["lai"][i]
    keep = (days >= 1) & (days <= ctx["cap"])
    innov, n_obs = [], 0
    for d, y in zip(days[keep], lai_obs[keep]):
        target = sow + dt.timedelta(days=int(d))
        for m in members:
            if m.day < target and not m.flag_terminate:
                m.run(days=(target - m.day).days)
        lai = np.array([m.get_variable("LAI") if m.get_variable("LAI") is not None else np.nan for m in members])
        dvs = np.array([m.get_variable("DVS") if m.get_variable("DVS") is not None else np.nan for m in members])
        active = np.isfinite(lai) & (dvs >= 0) & (dvs < 2)
        if active.sum() < N_MEMBERS // 2:
            continue
        f = lai[active]
        r = max(OBS_SD_FLOOR, OBS_SD_REL * y) ** 2
        k = f.var(ddof=1) / (f.var(ddof=1) + r)
        a = np.maximum(0.0, f + k * (y + rng.normal(0, np.sqrt(r), len(f)) - f))
        for m, v in zip([m for m, ok in zip(members, active) if ok], a):
            m.set_variable("LAI", float(v))
        innov.append(y - f.mean())
        n_obs += 1
    res = []
    for m in members:
        m.run_till_terminate()
        res.append(yw.summarise(m, sow, ctx["cap"], "DA"))
    mean = lambda key: float(np.nanmean([r[key] for r in res]))
    return {"DA_twso": mean("DA_twso"), "DA_twso_sd": float(np.nanstd([r["DA_twso"] for r in res])),
            "DA_tagp": mean("DA_tagp"), "DA_laimax": mean("DA_laimax"), "DA_lai_flower": mean("DA_lai_flower"),
            "DA_ctrat": mean("DA_ctrat"), "DA_days_to_flower": mean("_doa"), "DA_days_to_maturity": mean("_dom"),
            "n_obs": float(n_obs), "innovation_mean": float(np.mean(innov)) if innov else np.nan,
            "innovation_abs": float(np.mean(np.abs(innov))) if innov else np.nan,
            "obs_lai_max": float(np.max(lai_obs[keep])) if keep.any() else np.nan}


def main():
    p = parser(__doc__)
    p.add_argument("--lai", choices=sorted(LAI_SOURCES), default="arc")
    args = p.parse_args()
    name = LAI_SOURCES[args.lai]
    if args.submit:
        return submit("tools.build_yieldsat_wofost_da", name, args.submit, per_chunk=FIELDS_PER_CHUNK,
                      time_limit="02:00:00", extra_args=("--lai", args.lai))
    if args.stage == "run":
        subset, chunk = task_arg(args)
        return run_chunk(name, subset, chunk, da_field, ctx_fn=da_context(args.lai), per_chunk=FIELDS_PER_CHUNK)
    from tools import yieldsat_wofost as yw
    finalize(name, args.subset, {
        "crop": crop_of(args.subset), "season_cap_days": season_cap_of(args.subset), "lai_source": args.lai,
        "model": "Wofost72_WLP_CWB", "pcse": yw.PCSE_VERSION, "members": N_MEMBERS,
        "crop_parameters": f"ajwdewit/WOFOST_crop_parameters@{yw.CROP_PARAMS_COMMIT} (wofost72)",
        "filter": "stochastic EnKF on the LAI state at each LAI observation (field median, s2-aligned grid days "
                  f"1..cap, >= {MIN_PIXELS} pixels); obs SD max({OBS_SD_FLOOR}, {OBS_SD_REL} LAI)",
        "perturbations": "TDWI x LN(0,0.3), SPAN x N(1,0.1), SLATB and AMAXTB x N(1,0.15), TSUM1 x N(1,0.05), "
                         "WAV x U(0.5,1.5)",
        "method": "per field WOFOST 7.2 water-limited ensemble with assimilated satellite LAI; repeated per pixel"},
        per_chunk=FIELDS_PER_CHUNK)


if __name__ == "__main__":
    main()
