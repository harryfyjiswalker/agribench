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
import subprocess
import time
from pathlib import Path

import pandas as pd

from tools import yieldsat as ys
from tools import yieldsat_results as res

SLURM = "/cm/shared/apps/slurm/current/bin"


def run_dirs(root):
    return sorted({f.parent for f in root.glob("*/*/*/*/fold-*.json")})


def row_of(result, sweep):
    r = {"dataset": "yieldsat", "subset": result["label"], "scheme": result["scheme"], "input": result["input"],
         "model": result["model"], "folds": result["folds"], "n_features": result["provenance"]["n_features"],
         "pixel_cap": result["pixel_cap"]}
    for m in res.METRICS:
        r[f"{m}_mean"] = round(result["summary"][m]["mean"], 4)
        r[f"{m}_std"] = round(result["summary"][m]["std"], 4)
    for m in res.METRICS:
        r[f"pooled_{m}"] = round(result["pooled"][m], 4)
    r["task_hours"] = round(result["cost"]["task_seconds"] / 3600, 3)
    r["git_commit"] = (result["provenance"].get("git_commit") or "")[:10]
    r["sweep"] = sweep
    return r


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "yieldsat")
    args = parser.parse_args()
    root = ys.PROJECT / "runs" / "yieldsat" / "sweeps" / args.sweep
    rows, incomplete = [], []
    for d in run_dirs(root):
        try:
            result = res.combine(d) if not (d / "metrics.json").exists() or \
                max(f.stat().st_mtime for f in res.fold_files(d)) > (d / "metrics.json").stat().st_mtime \
                else json.loads((d / "metrics.json").read_text())
        except ValueError as error:
            incomplete.append(f"{d.relative_to(root)}: {error}")
            continue
        if result["smoke"]:
            continue
        rows.append(row_of(result, args.sweep))
    if incomplete:
        print(f"{len(incomplete)} incomplete runs:\n  " + "\n  ".join(incomplete[:50]))
    if rows:
        df = pd.DataFrame(rows).sort_values(["subset", "scheme", "input", "model"])
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.sweep}.csv"
        df.to_csv(path, index=False)
        (root / "summary.csv").write_text(df.to_csv(index=False))
        print(f"{len(df)} complete runs -> {path} ({time.strftime('%Y-%m-%d %H:%M')})")
    if args.efficiency:
        efficiency(args.sweep)


if __name__ == "__main__":
    main()
