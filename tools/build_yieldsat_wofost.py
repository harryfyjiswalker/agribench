"""Crop-model input `wofost` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

WOFOST 7.2 (PCSE) simulated per field and season, potential (`Wofost72_PP`) and water-limited
(`Wofost72_WLP_CWB`) production, from the field's own daily ERA5 weather, SoilGrids soil and sowing date. Set-up,
compromises and leakage guard: tools/yieldsat_wofost.py. Features (13, field values repeated on every pixel):

    PP_twso, PP_tagp, PP_laimax              potential grain yield, total above-ground biomass (kg/ha), max LAI
    WLP_twso, WLP_tagp, WLP_laimax           the same, water-limited
    WLP_lai_flower, WLP_ctrat                LAI at flowering, cumulative transpiration (cm), water-limited
    yield_ratio                              WLP_twso / PP_twso (water-limited share of potential)
    water_stress                             1 - WLP_ctrat / PP_ctrat (transpiration deficit)
    days_to_flower, days_to_maturity         simulated, days since sowing (the subset cap if not reached)
    soil_awc_mm                              available water capacity of the 100 cm root zone

Stages (Slurm; `--submit` chains them, with the PCSE environment's Python):
    python -m tools.build_yieldsat_wofost --submit ARG-C ARG-S ...      # chunk array -> finalize per subset
    python -m tools.build_yieldsat_wofost --stage run --subset GER-W --chunk 0
    python -m tools.build_yieldsat_wofost --stage finalize --subset GER-W

Writes features/yieldsat/wofost/<subset>.npy float32 (pixels, 13) + <subset>.json; work files in
features/yieldsat/wofost/_work/<subset>/ are removed by finalize.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_weather_pheno import field_info, subset_fields, write_dense

SLURM = "/cm/shared/apps/slurm/current/bin"
FIELDS_PER_CHUNK = 60


def crop_of(subset):
    return {v: k for k, v in ys.CROPS.items()}[subset.split("-")[1]]


def season_cap_of(subset):
    """The subset-wide season cap of `weather-pheno` (built first), so all weather inputs share one guard."""
    return int(json.loads((ys.FEATURES / "weather-pheno" / f"{subset}.json").read_text())["season_cap_days"])


def context(subset):
    from tools import yieldsat_wofost as yw
    fields, fidx = subset_fields(subset)
    lon = ys.field_table().set_index("field_shared_name").lon
    return {"subset": subset, "fields": fields, "fidx": fidx, "info": field_info(), "lon": lon,
            "soils": yw.field_soils(subset), "cap": season_cap_of(subset), "crop": crop_of(subset)}


def setup(ctx, i):
    from tools import yieldsat_wofost as yw
    f = ctx["fields"][i]
    country, _, sow, lat = ctx["info"][f]
    return yw.FieldSetup(f, country, ctx["crop"], sow, lat, float(ctx["lon"][f]), ctx["soils"][i], ctx["cap"]), sow


def wofost_field(ctx, i):
    from tools import yieldsat_wofost as yw
    fs, sow = setup(ctx, i)
    res = {}
    for kind in ("PP", "WLP"):
        m = fs.model(kind)
        m.run_till_terminate()
        res.update(yw.summarise(m, sow, ctx["cap"], kind))
    s = ctx["soils"][i]
    return {k: res[k] for k in ("PP_twso", "PP_tagp", "PP_laimax", "WLP_twso", "WLP_tagp", "WLP_laimax",
                                "WLP_lai_flower", "WLP_ctrat")} | {
        "yield_ratio": res["WLP_twso"] / res["PP_twso"] if res["PP_twso"] > 0 else np.nan,
        "water_stress": 1 - res["WLP_ctrat"] / res["PP_ctrat"] if res["PP_ctrat"] > 0 else np.nan,
        "days_to_flower": res["_doa"], "days_to_maturity": res["_dom"],
        "soil_awc_mm": 10 * (fs.soil["SMFCF"] - fs.soil["SMW"]) * fs.soil["RDMSOL"]}


# ---------------------------------------------------------------- chunked Slurm machinery (shared with wofost-da)

def n_chunks(subset, per_chunk=FIELDS_PER_CHUNK):
    return int(np.ceil(len(subset_fields(subset)[0]) / per_chunk))


def run_chunk(name, subset, chunk, field_fn, ctx_fn=context, per_chunk=FIELDS_PER_CHUNK):
    """Run field_fn(ctx, i) for the chunk's fields; write _work/<subset>/chunk-<k>.json."""
    wd = ys.FEATURES / name / "_work" / subset
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / f"chunk-{chunk:03d}.json"
    if out.exists():
        print(f"{out} exists; skipping", flush=True)
        return
    t0 = time.time()
    ctx = ctx_fn(subset)
    idx = range(chunk * per_chunk, min(len(ctx["fields"]), (chunk + 1) * per_chunk))
    rows, failed = {}, []
    for i in idx:
        try:
            rows[i] = field_fn(ctx, i)
        except Exception as e:   # one field failing must not lose the chunk; it becomes NaN and is counted
            failed.append(i)
            print(f"field {i} failed: {type(e).__name__}: {e}", flush=True)
    tmp = wd / f".chunk-{chunk:03d}.json.tmp"
    tmp.write_text(json.dumps({"rows": {str(k): v for k, v in rows.items()}, "failed": failed}))
    tmp.rename(out)
    print(json.dumps({"subset": subset, "chunk": chunk, "fields": len(idx), "failed": len(failed),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def finalize(name, subset, extra, per_chunk=FIELDS_PER_CHUNK):
    wd = ys.FEATURES / name / "_work" / subset
    fields, fidx = subset_fields(subset)
    rows, failed = {}, []
    for k in range(n_chunks(subset, per_chunk)):
        part = json.loads((wd / f"chunk-{k:03d}.json").read_text())
        rows.update({int(i): v for i, v in part["rows"].items()})
        failed += part["failed"]
    names = list(next(iter(rows.values())))
    values = np.full((len(fields), len(names)), np.nan)
    for i, r in rows.items():
        values[i] = [r[n] for n in names]
    extra = dict(extra, fields_failed=len(failed),
                 median_per_feature={n: round(float(np.nanmedian(values[:, k])), 3) for k, n in enumerate(names)})
    info = write_dense(name, subset, values, fidx, names, extra)
    shutil.rmtree(wd)
    print(json.dumps({k: info[k] for k in ("subset", "fields", "fields_failed", "median_per_feature")}), flush=True)


def submit(module, name, subsets, per_chunk=FIELDS_PER_CHUNK, mem="8G", time_limit="01:00:00", extra_args=()):
    """One array over all (subset, chunk) pairs, then a finalize job per subset; PCSE environment's Python."""
    from tools.yieldsat_wofost import PCSE_PYTHON
    logs = ys.FEATURES / name / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = f"--export=ALL,BENCHMARK_PYTHON={PCSE_PYTHON}"
    pairs = [(s, k) for s in subsets for k in range(n_chunks(s, per_chunk))]
    tasks = logs / f"tasks-{os.getpid()}.txt"
    tasks.write_text("".join(f"{s} {k}\n" for s, k in pairs))
    sb = lambda *a: subprocess.run([f"{SLURM}/sbatch", "--parsable", env, *a], check=True, capture_output=True,
                                   text=True).stdout.strip().split(";")[0]
    r = sb(f"--job-name=ys-{name}-run", f"--array=0-{len(pairs) - 1}%100", "--cpus-per-task=1", f"--mem={mem}",
           f"--time={time_limit}", f"--output={logs}/run-%A_%a.log", "prepare.sbatch", module, "--stage", "run",
           "--tasks", str(tasks), *extra_args)
    for s in subsets:
        f = sb(f"--job-name=ys-{name}-fin-{s}", f"--dependency=afterok:{r}", "--cpus-per-task=1", "--mem=16G",
               "--time=00:30:00", f"--output={logs}/fin-{s}-%j.log", "prepare.sbatch", module, "--stage",
               "finalize", "--subset", s, *extra_args)
        print(f"{s}: finalize {f}", flush=True)
    print(f"{len(pairs)} chunks -> array {r}", flush=True)


def task_arg(args):
    """(subset, chunk) from --subset/--chunk, or from line SLURM_ARRAY_TASK_ID of --tasks."""
    if args.tasks:
        line = open(args.tasks).read().splitlines()[int(os.environ["SLURM_ARRAY_TASK_ID"])]
        s, k = line.split()
        return s, int(k)
    return args.subset, args.chunk


def parser(doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["run", "finalize"])
    p.add_argument("--subset", choices=ys.SUBSETS)
    p.add_argument("--chunk", type=int)
    p.add_argument("--tasks", help="task list written by --submit (one 'subset chunk' per array index)")
    p.add_argument("--submit", nargs="+", choices=ys.SUBSETS)
    return p


def main():
    args = parser(__doc__).parse_args()
    if args.submit:
        return submit("tools.build_yieldsat_wofost", "wofost", args.submit)
    if args.stage == "run":
        subset, chunk = task_arg(args)
        return run_chunk("wofost", subset, chunk, wofost_field)
    from tools import yieldsat_wofost as yw
    finalize("wofost", args.subset, {
        "crop": crop_of(args.subset), "season_cap_days": season_cap_of(args.subset),
        "models": ["Wofost72_PP", "Wofost72_WLP_CWB"], "pcse": yw.PCSE_VERSION,
        "crop_parameters": f"ajwdewit/WOFOST_crop_parameters@{yw.CROP_PARAMS_COMMIT} (wofost72)",
        "variety": yw.VARIETY[crop_of(args.subset)][1],
        "phenology": "temperature only (IDSL=0), TSUM1/TSUM2 from the weather-pheno thermal-time constants",
        "soil": "SoilGrids 0-100 cm field mean -> Saxton & Rawls (2006); rooting depth 100 cm",
        "weather": "ERA5 Raw weather files to the subset cap; Hargreaves radiation, VAP from Tmin, wind 2 m/s",
        "method": "per field WOFOST 7.2 potential and water-limited runs from sowing; repeated per pixel"})


if __name__ == "__main__":
    main()
