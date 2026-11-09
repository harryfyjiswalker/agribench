"""ARC-VAE input `arcvae` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench, internal testing].

ARC-VAE (harryfyjiswalker/arc-vae, GPL-3.0) is a transformer VAE trained on synthetic Sentinel-2 series from the
ARC archetype crop model (Yin et al., UCL). Its encoder maps a pixel's whole season of reflectances to a posterior
over 11 ARC parameters, one forward pass per pixel:
    p (7, canopy scaling): N, Cab, Cm, Cw, LAI, ALA, Cbrown
    h (4, phenology):      growth_speed, start_DOY, senes_speed, end_DOY
Features are the posterior means and standard deviations: 22 columns. Only the encoder is used (it carries its own
input normalisation), so the ARC package is not needed. The published checkpoint is the maize model.

Inputs per observation: 10 bands (B02 B03 B04 B05 B06 B07 B08 B8A B11 B12, reflectance 0-1), angles in degrees
(sun zenith, view zenith, relative azimuth) and day of year.

[AgriBench adaptations]
  - Angles: YieldSAT keeps no geometry. Sun zenith is computed for the field's latitude and date at 10:30 local
    solar time, view zenith is 5 degrees and relative azimuth 90 degrees (as in `biophys`).
  - Day of year: the model was trained on a northern-hemisphere maize calendar (start of season around DOY
    115-210). Our subsets include southern-hemisphere and winter crops, so each observation gets a pseudo-DOY =
    SOWING_DOY + days since the field's sowing date; observations outside DOY 1-365 are dropped. Phenology outputs
    are therefore relative to sowing.
  - Crops other than maize are outside the model's training distribution; the features are tested as they are.

    python -m tools.build_yieldsat_arcvae --subset GER-W      (CPU job; one subset per job)

Writes features/yieldsat/arcvae/<subset>.npy float32 (pixels, 22) + <subset>.json.
"""
import argparse
import importlib.util
import json
import time

import numpy as np
import torch

from tools import yieldsat as ys
from tools.build_yieldsat_biophys import LOCAL_SOLAR_HOUR, sun_zenith_cos
from tools.build_yieldsat_s2_aligned import sowing_dates
from tools.embed_yieldsat_galileo import slot_months

UPSTREAM = ys.PROJECT / "upstream" / "arc-vae"
COMMIT = "794252d"
CHECKPOINT = UPSTREAM / "outputs" / "maize_d128_L4_e20+30_sup10.0_20260518_1637" / "checkpoint_best.pt"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PARAMS = ["N", "cab", "cm", "cw", "lai", "ala", "cbrown", "growth_speed", "start_doy", "senes_speed", "end_doy"]
VIEW_ZENITH_DEG = 5.0
RELATIVE_AZIMUTH_DEG = 90.0
SOWING_DOY = 120
BATCH = 8192


def load_encoder():
    spec = importlib.util.spec_from_file_location("arcvae_encoder", UPSTREAM / "arc_vae" / "encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((CHECKPOINT.parent / "config.json").read_text())
    enc = module.ARCVAEEncoder(d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                               d_ff=cfg["d_ff"], dropout=0.0)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"]
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
    return enc.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "arcvae"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    t0 = time.time()
    torch.set_num_threads(max(1, len(__import__("os").sched_getaffinity(0))))
    enc = load_encoder()
    names = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in BANDS] for s in range(ys.N_STEPS)])  # (24, 10)
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    latlon = np.load(ys.FEATURES / "latlon" / f"{args.subset}.npy", mmap_mode="r")
    _, _, dates, _ = slot_months(args.subset, fields)
    sow = sowing_dates()
    feature_names = [f"mu_{p}" for p in PARAMS] + [f"sigma_{p}" for p in PARAMS]

    tmp = out_dir / f".{args.subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(fidx), len(feature_names)))
    n_obs = []
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        f = fields[fidx[a]]
        doy = np.array([SOWING_DOY + (d - sow[f]).days if d is not None else np.nan for d in dates[f]], float)
        slots = np.flatnonzero((doy >= 1) & (doy <= 365))
        out[a:b] = np.nan
        if not len(slots):
            continue
        lat = float(np.mean(latlon[a:b, 0]))
        sza = [np.degrees(np.arccos(np.clip(sun_zenith_cos(lat, dates[f][s]), 0.05, 1.0))) for s in slots]
        refl = np.asarray(cache[a:b][:, cols[slots]], np.float32) / 10000.0  # (pixels, T, 10)
        mask = np.isfinite(refl).all(-1)
        ok = mask.any(1)
        n_obs.append(mask.sum(1))
        angles = np.broadcast_to(np.stack([sza, np.full(len(slots), VIEW_ZENITH_DEG),
                                           np.full(len(slots), RELATIVE_AZIMUTH_DEG)], -1), refl.shape[:2] + (3,))
        doys = np.broadcast_to(doy[slots], refl.shape[:2])
        rows = np.flatnonzero(ok)
        for k in range(0, len(rows), BATCH):
            r = rows[k:k + BATCH]
            with torch.no_grad():
                mu, sigma = enc(torch.from_numpy(np.nan_to_num(refl[r])), torch.from_numpy(angles[r].astype(np.float32)),
                                torch.from_numpy(doys[r].astype(np.float32)), torch.from_numpy(mask[r]))
            out[a + r] = torch.cat([mu, sigma], 1).numpy()
    out.flush()
    del out
    tmp.rename(target)

    arr = np.load(target, mmap_mode="r")
    obs = np.concatenate(n_obs) if n_obs else np.zeros(1)
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)), "feature_names": feature_names,
            "share_pixels_encoded": round(float(np.isfinite(arr[:, 0]).mean()), 4),
            "median_obs_per_pixel": float(np.median(obs)),
            "median_mu": {p: round(float(np.nanmedian(arr[::50, i])), 3) for i, p in enumerate(PARAMS)},
            "model": f"harryfyjiswalker/arc-vae@{COMMIT} {CHECKPOINT.parent.name} (maize; GPL-3.0)",
            "geometry": {"view_zenith_deg": VIEW_ZENITH_DEG, "relative_azimuth_deg": RELATIVE_AZIMUTH_DEG,
                         "sun": f"computed at {LOCAL_SOLAR_HOUR} h local solar time"},
            "day_of_year": f"pseudo-DOY = {SOWING_DOY} + days since sowing; outside 1-365 dropped",
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "share_pixels_encoded", "median_obs_per_pixel",
                                           "median_mu", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
