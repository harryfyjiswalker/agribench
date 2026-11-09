"""Harmonic (Fourier) Sentinel-2 input `s2-harmonic` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

Each pixel's series is summarised, per band, by the coefficients of a harmonic regression, the standard compact
representation of irregular, cloud-gapped time series (e.g. CCDC; Jakubauskas et al. 2001):

    y(t) = c0 + sum_{k=1..K} a_k cos(2 pi k t / P) + b_k sin(2 pi k t / P)

with t = days since the field's sowing date (so phase means growth stage, as in s2-aligned), P = 365 days, K = 2.
That is 5 coefficients x 12 bands = 60 features, regardless of how many observations a pixel has. Fitted by
least squares on the pixel's valid observations, with a small ridge penalty (RIDGE, in units of observations) on
the harmonic terms only, so pixels with few observations stay stable. A band needs at least MIN_OBS valid
observations; otherwise its coefficients are NaN.

    python -m tools.build_yieldsat_s2_harmonic --subset GER-W      (CPU job; one subset per job)

Writes features/yieldsat/s2-harmonic/<subset>.npy float32 (pixels, 60) + <subset>.json.
"""
import argparse
import json
import time

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_s2_aligned import sowing_dates
from tools.embed_yieldsat_galileo import slot_months

PERIOD = 365.0
K = 2
RIDGE = 0.5
MIN_OBS = 5
TERMS = ["c0"] + [f"{f}{k}" for k in range(1, K + 1) for f in ("cos", "sin")]


def design(t):
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        w = 2 * np.pi * k * t / PERIOD
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)  # (slots, 2K+1)


def fit(values, t):
    """values (pixels, slots, bands) with NaN gaps, t (slots,) -> coefficients (pixels, bands, 2K+1)."""
    a = design(t)
    m = np.isfinite(values)
    y = np.where(m, values, 0.0).astype(np.float64)
    g = np.einsum("sk,psb,sl->pbkl", a, m.astype(np.float64), a)
    g += RIDGE * np.diag([0.0] + [1.0] * (2 * K))
    g[..., 0, 0] += 1e-9  # keeps the intercept solvable for all-missing bands (set to NaN below)
    rhs = np.einsum("sk,psb->pbk", a, y)
    coef = np.linalg.solve(g, rhs[..., None])[..., 0]
    coef[m.sum(1) < MIN_OBS] = np.nan
    return coef.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "s2-harmonic"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    t0 = time.time()
    names = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in ys.S2_BANDS] for s in range(ys.N_STEPS)])
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    _, _, dates, _ = slot_months(args.subset, fields)
    sow = sowing_dates()
    feature_names = [f"{b}_{term}" for b in ys.S2_BANDS for term in TERMS]

    tmp = out_dir / f".{args.subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(fidx), len(feature_names)))
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        f = fields[fidx[a]]
        t = np.array([(d - sow[f]).days if d is not None else np.nan for d in dates[f]], float)
        slots = np.flatnonzero(np.isfinite(t))
        values = np.asarray(cache[a:b][:, cols[slots]], np.float32)
        out[a:b] = fit(values, t[slots]).reshape(b - a, -1)
    out.flush()
    del out
    tmp.rename(target)

    arr = np.load(target, mmap_mode="r")
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)), "feature_names": feature_names,
            "period_days": PERIOD, "harmonics": K, "ridge": RIDGE, "min_obs": MIN_OBS,
            "time_axis": "days since sowing",
            "share_pixels_fitted": round(float(np.isfinite(arr[:, feature_names.index("B04_c0")]).mean()), 4),
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "share_pixels_fitted", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
