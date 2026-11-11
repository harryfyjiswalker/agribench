"""Combine a YieldSAT sweep's fold results and write the public results table.

    python -m tools.collect_yieldsat --sweep phase1                 # combine + write results/yieldsat/phase1.csv
    python -m tools.collect_yieldsat --sweep phase1 --efficiency    # also: requested vs used resources per head

For every run directory of the sweep with a complete set of folds, writes metrics.json (tools/yieldsat_results).
The table in results/yieldsat/ holds aggregate scores only (mean +/- std over folds and pooled scores per subset,
scheme, input and model): no field names, locations, fold lists or predictions, because the repo is public and
the dataset licence covers the data, not our summaries. The dashboard (dashboard/) reads these tables.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tools import yieldsat as ys
from tools import yieldsat_results as res

SLURM = "/cm/shared/apps/slurm/current/bin"


def run_dirs(root):
    return sorted({f.parent for f in root.glob("*/*/*/*/fold-*.json")})


def needs_combine(d):
    """metrics.json is missing or older than one of the run's fold files."""
    m = d / "metrics.json"
    return not m.exists() or max(f.stat().st_mtime for f in res.fold_files(d)) > m.stat().st_mtime


def combine_or_error(d):
    """res.combine for a worker process: the result, or the error message if the run is incomplete."""
    try:
        return res.combine(d)
    except ValueError as error:
        return str(error)


def row_of(result, sweep, input_dir=None):
    """input_dir: the run's input directory name, which carries a variant suffix (s2-aligned@d060) if any."""
    r = {"dataset": "yieldsat", "subset": result["label"], "scheme": result["scheme"],
         "input": input_dir or result["input"],
         "model": result["model"], "folds": result["folds"], "n_features": result["provenance"]["n_features"],
         "pixel_cap": result["pixel_cap"]}
    if result.get("season_cutoff") is not None or result.get("train_fraction", 1.0) < 1:
        r.update(season_cutoff=result.get("season_cutoff"), train_fraction=result.get("train_fraction", 1.0),
                 seed=result["seed"])
    for m in res.METRICS:
        r[f"{m}_mean"] = round(result["summary"][m]["mean"], 4)
        r[f"{m}_std"] = round(result["summary"][m]["std"], 4)
    for m in res.METRICS:
        r[f"pooled_{m}"] = round(result["pooled"][m], 4)
    r["task_hours"] = round(result["cost"]["task_seconds"] / 3600, 3)
    r["git_commit"] = (result["provenance"].get("git_commit") or "")[:10]
    r["sweep"] = sweep
    return r


def loco_rows(result, sweep):
    """Leave one country out: the crop row (pooled over held-out countries) plus one row per held-out country,
    labelled "<crop>>COUNTRY" (a single fold, so its mean and pooled scores are the same and std is 0)."""
    rows = [row_of(result, sweep)]
    for f in result["per_fold"]:
        r = dict(rows[0], subset=f"{result['label']}>{f['held_out_country']}", folds=1,
                 task_hours=round(f["seconds"] / 3600, 3))
        for m in res.METRICS:
            r[f"{m}_mean"] = r[f"pooled_{m}"] = round(f[m], 4)
            r[f"{m}_std"] = 0.0
        rows.append(r)
    return rows


def efficiency(sweep):
    """sacct summary per array (head x width): elapsed vs limit, CPU use, memory use."""
    fmt = "JobID,JobName,State,Elapsed,Timelimit,TotalCPU,AllocCPUS,ReqMem,MaxRSS"
    out = subprocess.run([f"{SLURM}/sacct", "-n", "-P", "-S", "now-14days", "--format", fmt],
                         capture_output=True, text=True).stdout

    def secs(t):
        if not t or t in ("UNLIMITED", "Partition_Limit"):
            return float("nan")
        d, _, hms = t.rpartition("-")
        parts = [float(x) for x in hms.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]

    def gb(m):
        if not m:
            return 0.0
        unit = m[-1]
        v = float(m[:-1]) if unit.isalpha() else float(m)
        return v * {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1, "T": 1024}.get(unit, 1 / 1024 ** 3)

    rows, steps = {}, {}
    for line in out.splitlines():
        jid, name, state, el, lim, cpu, ncpu, req, rss = line.split("|")
        base = jid.split(".")[0]
        if "." in jid:
            steps[base] = max(steps.get(base, 0.0), gb(rss))
            continue
        if not name.startswith(f"ys-{sweep}-") or name.endswith("collect"):
            continue
        rows[base] = {"array": name.removeprefix(f"ys-{sweep}-"), "state": state, "elapsed": secs(el),
                      "limit": secs(lim), "cpu": secs(cpu), "ncpu": int(ncpu or 0), "req_gb": gb(req)}
    df = pd.DataFrame([{**v, "rss_gb": steps.get(k, 0.0)} for k, v in rows.items()])
    if df.empty:
        print("No sacct records for this sweep yet.")
        return
    df["cpu_eff"] = df.cpu / (df.elapsed * df.ncpu).clip(lower=1)
    g = df.groupby("array").agg(tasks=("state", "size"), failed=("state", lambda s: int((~s.str.startswith("COMPLETED")).sum())),
                                elapsed_p95_min=("elapsed", lambda s: s.quantile(0.95) / 60),
                                limit_min=("limit", lambda s: s.max() / 60), cpu_eff=("cpu_eff", "median"),
                                rss_max_gb=("rss_gb", "max"), req_gb=("req_gb", "max"))
    print(g.round(2).to_string())
    print("Suggested: time limit ~1.5x elapsed_p95, memory ~1.3x rss_max, fewer CPUs where cpu_eff is low.")


def write_index(results_root):
    """results/index.json: the tables the dashboard loads."""
    files = sorted(p.relative_to(results_root).as_posix() for p in results_root.glob("*/*.csv"))
    index = {"updated": time.strftime("%Y-%m-%d %H:%M %Z"), "files": [{"path": f} for f in files]}
    (results_root / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="processes for combining runs (default: the job's CPUs)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "yieldsat")
    args = parser.parse_args()
    root = ys.PROJECT / "runs" / "yieldsat" / "sweeps" / args.sweep
    rows, incomplete = [], []
    dirs = run_dirs(root)
    stale = [d for d in dirs if needs_combine(d)]
    combined = {}
    if stale:  # combining reads and rewrites every held-out pixel's prediction: parallel over runs
        workers = max(1, min(args.workers, len(stale)))
        print(f"Combining {len(stale)} of {len(dirs)} runs with {workers} workers", flush=True)
        with ProcessPoolExecutor(workers) as pool:
            combined = dict(zip(stale, pool.map(combine_or_error, stale, chunksize=4)))
    for d in dirs:
        result = combined[d] if d in combined else json.loads((d / "metrics.json").read_text())
        if isinstance(result, str):
            incomplete.append(f"{d.relative_to(root)}: {result}")
            continue
        if result["smoke"]:
            continue
        rows.extend(loco_rows(result, args.sweep) if result["scheme"] == "loco" else
                    [row_of(result, args.sweep, d.parent.name)])
    if incomplete:
        print(f"{len(incomplete)} incomplete runs:\n  " + "\n  ".join(incomplete[:50]))
    if rows:
        df = pd.DataFrame(rows).sort_values([c for c in ("subset", "scheme", "input", "model", "season_cutoff",
                                                              "train_fraction", "seed") if c in {k for r in rows for k in r}])
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.sweep}.csv"
        df.to_csv(path, index=False)
        (root / "summary.csv").write_text(df.to_csv(index=False))
        print(f"{len(df)} complete runs -> {path} ({time.strftime('%Y-%m-%d %H:%M')})")
        write_index(args.out.parent)
    if args.efficiency:
        efficiency(args.sweep)


if __name__ == "__main__":
    main()
