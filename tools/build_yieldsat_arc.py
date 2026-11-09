"""ARC input `arc` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench, internal testing].

ARC (Archetypal Crop trait dynamics; Yin et al., UCL; MarcYin/ARC) retrieves per-pixel crop parameters by matching a
pixel's Sentinel-2 season against a library of PROSAIL-simulated seasons drawn from a crop archetype: 7 canopy
scaling parameters (N, Cab, Cm, Cw, LAI, ALA, Cbrown), 4 phenology parameters (growth speed, start, senescence
speed, end) and 4 soil parameters, plus the retrieved LAI trajectory. The method follows the AgriBench author's
county-level ARC pipeline (Whittaker IRLS cleaning, V-curve lambda, EVI2 phenology, generate_arc_refs ->
nearest neighbours -> assimilate), adapted to YieldSAT pixels and made cheap enough for millions of them:

  1. Cleaning, per field (as the original pipeline): Whittaker IRLS (d=2, 3 iterations, Tukey bisquare c=4.685,
     MAD scale) on B03, B8A and B11 of the field's own observations, lambda chosen by the V-curve over
     logspace(10, 1000, 30) as the median over the three bands. An observation down-weighted to zero in any of the
     three bands is removed in all bands.
  2. Shared time axis [AgriBench]: the cleaned observations are interpolated onto the `s2-aligned` days-since-
     sowing grid, and ARC sees pseudo day-of-year = SOWING_DOY + days since sowing. Every pixel of a subset then
     shares one observation calendar, so ONE reference library (2^17 simulated seasons) and ONE nearest-neighbour
     index serve the whole subset. The original pipeline builds both per county-year, which is what made it slow.
  3. Phenology prior, per subset: start of season and season length from the subset's median EVI2 curve with a
     relative threshold of 0.5 (as the original pipeline), passed to generate_arc_refs.
  4. Retrieval: pynndescent query (k=300) on the partitioned medians, then ARC's assimilate (best 50, inverse-
     distance weights), in pixel chunks. Band uncertainty 5% of reflectance (floor 1e-4).

[AgriBench approximations] Angles: sun zenith from the subset's median latitude and the median sowing date + grid
offset at 10:30 local solar time; view zenith 5 degrees; relative azimuth 90 degrees (YieldSAT keeps no geometry).
Archetypes: corn -> maize, soybean -> soybean, wheat -> wheat; rapeseed has no ARC archetype and uses wheat
(winter crop with a similar calendar).

Stages (all Slurm jobs; `--submit` chains them):
    python -m tools.build_yieldsat_arc --submit GER-W BRA-C        # prepare -> chunk array -> finalize, per subset
    python -m tools.build_yieldsat_arc --stage prepare --subset GER-W
    python -m tools.build_yieldsat_arc --stage run --subset GER-W --chunk 3
    python -m tools.build_yieldsat_arc --stage finalize --subset GER-W

Writes features/yieldsat/arc/<subset>.npy float32 (pixels, 16 + grid steps) + <subset>.json. Work files live in
features/yieldsat/arc/_work/<subset>/ and are removed by finalize.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_biophys import LOCAL_SOLAR_HOUR, sun_zenith_cos
from tools.build_yieldsat_s2_aligned import GRID_START, MAX_END, STEP, interpolate, sowing_dates
from tools.embed_yieldsat_galileo import slot_months

ARC_COMMIT = "7a200007b0d48df22dc2b0b5b382ee84675730e1"
ARC_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
IRLS_BANDS = ["B03", "B8A", "B11"]
ARCHETYPE = {"corn": "maize", "soybean": "soybean", "wheat": "wheat", "rapeseed": "wheat"}
PARAMS = ["N", "cab", "cm", "cw", "lai", "ala", "cbrown", "growth_speed", "sos", "sen_speed", "eos",
          "soil_brightness", "soil_shape_p1", "soil_shape_p2", "soil_moisture"]
SOWING_DOY = 30           # pseudo-DOY of sowing; keeps the grid (-15 .. <=330 days) inside 1..365
NUM_SAMPLES = 100_000     # ARC rounds up to 2^17
PHENOLOGY_THRESHOLD = 0.5
BAND_UNC_PCT = 0.05
K_NEIGHBOURS = 300
KNN_STEPS = 10
VIEW_ZENITH_DEG, RELATIVE_AZIMUTH_DEG = 5.0, 90.0
PIXELS_PER_CHUNK = 100_000
# Whittaker settings (as the original pipeline)
WHIT = dict(d=2, ridge=1e-10, normalize="median", eps=1e-12)
IRLS = dict(iterations=3, weighting="tukey", scale="mad", tuning=(0.0, 0.0, 4.685))
IRLS_THRESH = 1e-8
LAM_GRID = np.logspace(1.0, 3.0, 30)
SLURM = "/cm/shared/apps/slurm/current/bin"
ARC_PYTHON = ys.PROJECT / "envs" / "arc" / "bin" / "python"


def work_dir(subset):
    return ys.FEATURES / "arc" / "_work" / subset


# ---------------------------------------------------------------- cleaning (per field)

def strictly_increasing(x):
    x = np.asarray(x, np.float64).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 0.1
    return np.ascontiguousarray(x)


def vcurve_lambda(ws, x, y, w):
    log_r, log_s = np.empty(len(LAM_GRID)), np.empty(len(LAM_GRID))
    valid = w > 0
    for k, lam in enumerate(LAM_GRID):
        z = ws.whittaker_solve_f64(x, y, w, lam=float(lam), parallel=True, **WHIT)
        log_r[k] = np.log(max(float(np.sum(w[valid] * (y[valid] - z[valid]) ** 2)), 1e-12))
        log_s[k] = np.log(max(float(np.sum(np.diff(z, n=2, axis=-1) ** 2)), 1e-12))
    norm = lambda v: (v - v.min()) / (v.max() - v.min() if v.max() - v.min() > 1e-12 else 1.0)
    vc = np.sqrt(np.diff(norm(log_r)) ** 2 + np.diff(norm(log_s)) ** 2)
    return float(np.sqrt(LAM_GRID[:-1] * LAM_GRID[1:])[np.argmin(vc)])


def clean_field(ws, refl, days):
    """refl (pixels, slots, 10) reflectance with NaN gaps, days (slots,) -> (cleaned refl, lambda, share removed)."""
    x = strictly_increasing(days)
    base = ~np.isfinite(refl).all(-1)                       # (pixels, slots)
    if (~base).sum(1).max(initial=0) < 5:
        return refl, np.nan, 0.0
    ys_w = []
    for b in IRLS_BANDS:
        y = refl[..., ARC_BANDS.index(b)].astype(np.float64)
        w = np.where(base | ~np.isfinite(y), 0.0, 1.0)
        ys_w.append((np.ascontiguousarray(np.nan_to_num(y)), np.ascontiguousarray(w)))
    lams = [vcurve_lambda(ws, x, y, w) for y, w in ys_w if w.sum() > 10]
    lam = float(np.median(lams)) if lams else 1000.0
    outlier = np.zeros_like(base)
    for y, w in ys_w:
        if w.sum() < 10:
            continue
        _, w_out = ws.robust_whittaker_irls_f64(x, y, w, lam=lam, parallel=True, return_weights=True, **WHIT, **IRLS)
        outlier |= (w_out < IRLS_THRESH) & ~base
    out = refl.copy()
    out[outlier] = np.nan
    return out, lam, float(outlier.sum() / max(1, (~base).sum()))


# ---------------------------------------------------------------- phenology prior (per subset)

def evi2_phenology(grid_refl, pseudo_doy):
    """Median EVI2 over pixels on the grid -> (start of season, season length) in pseudo-DOY, as the original."""
    nir, red = grid_refl[..., ARC_BANDS.index("B08")], grid_refl[..., ARC_BANDS.index("B04")]
    with np.errstate(all="ignore"):
        evi2 = np.nanmedian(2.5 * (nir - red) / (nir + 2.4 * red + 1.0), axis=0)
    ok = np.isfinite(evi2)
    fallback = (SOWING_DOY + 20.0, 120.0)
    if ok.sum() < 5:
        return fallback
    daily = np.arange(int(pseudo_doy[ok][0]), int(pseudo_doy[ok][-1]) + 1)
    gz = np.interp(daily, pseudo_doy[ok], evi2[ok])
    peak = int(np.argmax(gz))
    asc = gz[:peak + 1]
    cross = np.flatnonzero(asc >= asc.min() + PHENOLOGY_THRESHOLD * (gz[peak] - asc.min()))
    sos = float(daily[cross[0]]) if len(cross) else float(daily[0])
    desc = gz[peak:]
    cross = np.flatnonzero(desc <= desc.min() + PHENOLOGY_THRESHOLD * (gz[peak] - desc.min()))
    eos = float(daily[peak + cross[0]]) if len(cross) else float(daily[-1])
    return sos, max(eos - sos, 30.0)


# ---------------------------------------------------------------- stages

def prepare(subset):
    """Clean every field, interpolate onto the grid, build the subset's reference library."""
    import whitsmooth_rust as ws
    from arc.arc_sample_generator import generate_arc_refs
    t0 = time.time()
    wd = work_dir(subset)
    if (ys.FEATURES / "arc" / f"{subset}.npy").exists():
        raise SystemExit(f"arc/{subset}.npy exists; caches are never overwritten")
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    crop = {v: k for k, v in ys.CROPS.items()}[subset.split("-")[1]]
    names = json.loads((ys.FEATURES / "v1" / f"{subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in ARC_BANDS] for s in range(ys.N_STEPS)])
    meta = np.load(ys.FEATURES / "v1" / f"{subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{subset}.npy", mmap_mode="r")
    latlon = np.load(ys.FEATURES / "latlon" / f"{subset}.npy", mmap_mode="r")
    _, _, dates, _ = slot_months(subset, fields)
    sow = sowing_dates()

    rel = np.full((len(fields), ys.N_STEPS), np.nan)
    for i, f in enumerate(fields):
        rel[i] = [(d - sow[f]).days if d is not None else np.nan for d in dates[f]]
    end = int(min(MAX_END, STEP * np.ceil(np.nanpercentile(np.nanmax(rel, axis=1), 95) / STEP)))
    grid = np.arange(GRID_START, end + 1, STEP)
    pseudo = SOWING_DOY + grid

    refl_grid = np.lib.format.open_memmap(wd / "refl.npy", mode="w+", dtype=np.float32,
                                          shape=(len(fidx), len(grid), len(ARC_BANDS)))
    lams, removed = [], []
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        i = fidx[a]
        slots = np.flatnonzero(np.isfinite(rel[i]))
        if not len(slots):
            refl_grid[a:b] = np.nan
            continue
        refl = np.asarray(cache[a:b][:, cols[slots]], np.float32) / 10000.0
        clean, lam, rm = clean_field(ws, refl, rel[i][slots])
        lams.append(lam)
        removed.append(rm)
        refl_grid[a:b] = interpolate(clean, rel[i][slots], grid)
    refl_grid.flush()
    t_clean = time.time() - t0

    sample = np.asarray(refl_grid[::max(1, len(fidx) // 50_000)])
    sos, gsl = evi2_phenology(sample, pseudo)
    lat = float(np.median(latlon[::100, 0]))
    sow_med = dt.date.fromordinal(int(np.median([sow[f].toordinal() for f in fields])))
    sza = np.array([np.degrees(np.arccos(np.clip(sun_zenith_cos(lat, sow_med + dt.timedelta(days=int(g))), 0.05, 1)))
                    for g in grid])
    angs = np.stack([sza, np.full(len(grid), VIEW_ZENITH_DEG), np.full(len(grid), RELATIVE_AZIMUTH_DEG)])
    np.random.seed(0)
    arc_refs, pheo, bio, orig_bios, soil = generate_arc_refs(
        doys=pseudo, start_of_season=int(round(sos)), growth_season_length=int(round(gsl)),
        num_samples=NUM_SAMPLES, angs=angs, crop_type=ARCHETYPE[crop])
    np.savez(wd / "library.npz", arc_refs=np.asarray(arc_refs, np.float32), pheo=np.asarray(pheo, np.float32),
             bio=np.asarray(bio, np.float32), orig_bios=np.asarray(orig_bios), soil=np.asarray(soil, np.float32))
    n_chunks = int(np.ceil(len(fidx) / PIXELS_PER_CHUNK))
    info = {"subset": subset, "crop": crop, "archetype": ARCHETYPE[crop], "rows": int(len(fidx)),
            "grid_days_since_sowing": grid.tolist(), "pseudo_doy": pseudo.tolist(),
            "evi2_sos_pseudo_doy": sos, "evi2_gsl_days": gsl, "median_lat": lat,
            "sun_zenith_deg": [round(float(v), 1) for v in sza], "n_chunks": n_chunks,
            "whittaker_lambda_median": float(np.nanmedian(lams)) if lams else None,
            "share_obs_removed_as_outliers": round(float(np.mean(removed)), 4) if removed else None,
            "library_shape": list(np.shape(arc_refs)), "orig_bios_shape": list(np.shape(orig_bios)),
            "seconds_clean": round(t_clean, 1), "seconds_prepare": round(time.time() - t0, 1)}
    (wd / "prepare.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


def build_index(lib, pseudo, refl_sample):
    """pynndescent index over the library's partitioned medians, as get_neighbours does, built once."""
    import pynndescent
    from arc.approximate_KNN_search import non_negative_weighted_euclidean, partition_data, partition_doy
    parts = partition_doy(pseudo, steps=KNN_STEPS)
    errs = np.maximum(np.abs(refl_sample) * BAND_UNC_PCT, 1e-4)        # (pixels, T, 10)
    weights = 1 / np.repeat(np.nanmean(errs, axis=(0, 1)), len(parts)) ** 2
    ens = partition_data(parts, lib["arc_refs"])                       # (10 * parts, samples)
    index = pynndescent.NNDescent(ens.T, metric=non_negative_weighted_euclidean, metric_kwds={"weights": weights},
                                  random_state=0, n_jobs=-1)
    index.prepare()
    return index, parts


def n_features(info):
    return len(PARAMS) + 1 + len(info["grid_days_since_sowing"])


def run_chunk(subset, chunk):
    import jax.numpy as jnp
    from arc.approximate_KNN_search import partition_data
    from arc.assimilate_jax import assimilate
    t0 = time.time()
    wd = work_dir(subset)
    info = json.loads((wd / "prepare.json").read_text())
    pseudo = np.array(info["pseudo_doy"])
    refl_all = np.load(wd / "refl.npy", mmap_mode="r")
    lib = dict(np.load(wd / "library.npz"))
    a = chunk * PIXELS_PER_CHUNK
    b = min(len(refl_all), a + PIXELS_PER_CHUNK)
    index, parts = build_index(lib, pseudo, np.asarray(refl_all[::max(1, len(refl_all) // 20_000)]))
    t_index = time.time() - t0
    out = np.full((b - a, n_features(info)), np.nan, np.float32)
    lib_j = {k: jnp.asarray(v) for k, v in lib.items()}
    batch = 20_000
    for s in range(a, b, batch):
        e = min(b, s + batch)
        r = np.transpose(np.asarray(refl_all[s:e]), (2, 1, 0))         # (10, T, pixels), ARC's layout
        valid = np.isfinite(r).any(axis=(0, 1))
        if not valid.any():
            continue
        r, idx = r[..., valid], np.flatnonzero(valid)
        uncs = np.maximum(np.abs(np.nan_to_num(r)) * BAND_UNC_PCT, 1e-4).astype(np.float32)
        obs = partition_data(parts, r)
        obs[~np.isfinite(obs)] = -9999
        neighbours = index.query(obs.T, k=K_NEIGHBOURS)[0]
        post_bio, _, post_sc, post_pheo, post_soil, _, _ = assimilate(
            jnp.asarray(r), lib_j["arc_refs"], jnp.asarray(uncs), lib_j["pheo"], lib_j["bio"], lib_j["soil"],
            lib_j["orig_bios"], jnp.asarray(neighbours))
        post_bio = np.asarray(post_bio)                                # (pixels, 7, time)
        lai = post_bio[:, 4, :] / 100.0
        lai_grid = lai[:, pseudo - 1] if lai.shape[1] == 365 else lai[:, :len(pseudo)]
        feats = np.concatenate([np.asarray(post_sc), np.asarray(post_pheo), np.asarray(post_soil),
                                lai.max(axis=1, keepdims=True), lai_grid], axis=1)
        out[s - a + idx] = feats
    np.save(wd / f"chunk-{chunk:04d}.npy", out)
    print(json.dumps({"subset": subset, "chunk": chunk, "rows": int(b - a), "seconds_index": round(t_index, 1),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def finalize(subset):
    wd = work_dir(subset)
    info = json.loads((wd / "prepare.json").read_text())
    chunks = [wd / f"chunk-{c:04d}.npy" for c in range(info["n_chunks"])]
    missing = [c.name for c in chunks if not c.exists()]
    if missing:
        raise SystemExit(f"{subset}: chunks missing: {missing}")
    names = [f"{p}" for p in PARAMS] + ["lai_peak"] + [f"lai_d{g:+04d}" for g in info["grid_days_since_sowing"]]
    out_dir = ys.FEATURES / "arc"
    tmp = out_dir / f".{subset}.npy.tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(info["rows"], len(names)))
    for c, path in enumerate(chunks):
        x = np.load(path)
        arr[c * PIXELS_PER_CHUNK: c * PIXELS_PER_CHUNK + len(x)] = x
    arr.flush()
    del arr
    tmp.rename(out_dir / f"{subset}.npy")
    arr = np.load(out_dir / f"{subset}.npy", mmap_mode="r")
    meta = {k: v for k, v in info.items() if k != "n_chunks"}
    meta.update({"kind": "dense", "feature_names": names,
                 "share_pixels_retrieved": round(float(np.isfinite(arr[:, 0]).mean()), 4),
                 "median_per_feature": {n: round(float(np.nanmedian(arr[::50, k])), 3) for k, n in enumerate(names[:16])},
                 "source": f"MarcYin/ARC@{ARC_COMMIT} (no licence; internal testing)",
                 "geometry": {"view_zenith_deg": VIEW_ZENITH_DEG, "relative_azimuth_deg": RELATIVE_AZIMUTH_DEG,
                              "sun": f"median latitude, median sowing date + grid offset, {LOCAL_SOLAR_HOUR} h"},
                 "aligned_with": f"v1/{subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz")})
    (out_dir / f"{subset}.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.rmtree(wd)
    print(json.dumps({k: meta[k] for k in ("subset", "rows", "share_pixels_retrieved", "median_per_feature")}))


def submit(subsets, cpus):
    """prepare -> array of chunks -> finalize per subset, with the ARC environment's Python."""
    logs = ys.FEATURES / "arc" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = f"--export=ALL,BENCHMARK_PYTHON={ARC_PYTHON},ARC_ASSIM_BATCH=2048"
    for s in subsets:
        rows = len(np.load(ys.FEATURES / "v1" / f"{s}-meta.npz")["field_idx"])
        n = int(np.ceil(rows / PIXELS_PER_CHUNK))
        sb = lambda *a: subprocess.run([f"{SLURM}/sbatch", "--parsable", env, *a], check=True, capture_output=True,
                                       text=True).stdout.strip().split(";")[0]
        p = sb(f"--job-name=ys-arc-prep-{s}", "--cpus-per-task=8", "--mem=48G", "--time=04:00:00",
               f"--output={logs}/prep-{s}-%j.log", "prepare.sbatch", "tools.build_yieldsat_arc", "--stage", "prepare",
               "--subset", s)
        r = sb(f"--job-name=ys-arc-run-{s}", f"--dependency=afterok:{p}", f"--array=0-{n - 1}%8",
               f"--cpus-per-task={cpus}", "--mem=48G", "--time=04:00:00", f"--output={logs}/run-{s}-%A_%a.log",
               "prepare.sbatch", "tools.build_yieldsat_arc", "--stage", "run", "--subset", s)
        f = sb(f"--job-name=ys-arc-fin-{s}", f"--dependency=afterok:{r}", "--cpus-per-task=1", "--mem=16G",
               "--time=00:30:00", f"--output={logs}/fin-{s}-%j.log", "prepare.sbatch", "tools.build_yieldsat_arc",
               "--stage", "finalize", "--subset", s)
        print(f"{s}: {rows} pixels, {n} chunks -> prepare {p}, run {r}, finalize {f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["prepare", "run", "finalize"])
    parser.add_argument("--subset", choices=ys.SUBSETS)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--submit", nargs="+", choices=ys.SUBSETS)
    parser.add_argument("--cpus", type=int, default=8)
    args = parser.parse_args()
    if args.submit:
        return submit(args.submit, args.cpus)
    {"prepare": lambda: prepare(args.subset), "run": lambda: run_chunk(args.subset, chunk_arg(args)),
     "finalize": lambda: finalize(args.subset)}[args.stage]()


def chunk_arg(args):
    return args.chunk if args.chunk is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])


if __name__ == "__main__":
    main()
