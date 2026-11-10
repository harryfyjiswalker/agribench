"""Run a grid of PLANTEUR experiments as Slurm job arrays, sized per head, then collect the results.

    python -m tools.sweep_planteur --name pa1 --inputs s2-monthly emb-aef emb-tessera-v2 --models probe lightgbm
    python -m tools.sweep_planteur --name pa1 ... --dry-run            # print the plan only
    python -m tools.sweep_planteur --name le1 --inputs emb-aef --models probe --fewshot 10 25 100 1000 --splits default
    python -m tools.sweep_planteur --name es1 --inputs s2-monthly --models probe --months 2 4 6 8 10

Every (split, input, model) combination writes to runs/planteur/sweeps/<name>/<split>/<input>[@variant]/<model>/,
with one array task and one fold-<fold>.json per fold: the default split has one fold, LOIO five (one per island).
Variants: few-shot X (label efficiency) adds @x0100, early season (first k months) adds @m04. Every task is a final
test (experiment.confirm_test=true, logged in the test ledger, protocol 3.7): the heads and their grids are fixed in
advance and every choice inside a run is made on the fold's val pixels. Tasks whose fold file exists are skipped,
so re-running a command resumes a sweep. A final CPU job (afterany) writes results/planteur/<name>.csv.

Tasks run from a frozen copy of the checkout's HEAD (runs/planteur/sweeps/<name>/_code/<commit>/, as in
tools/sweep_yieldsat.py); the checkout must be clean. GPU arrays run with --nice (low priority).
"""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np

from tools import planteur as pl
from tools.sweep_yieldsat import frozen_code, submit

SLURM = "/cm/shared/apps/slurm/current/bin"
FOLDS = {"default": ["default"], "loio": ["reunion", "guadeloupe", "martinique", "guyane", "mayotte"]}
# First estimates (2026-11-10); tighten from sacct once the first sweep has run.
RESOURCES = {
    "background": {"gpu": False, "cpus": 1, "mem": "8G",  "time": "00:20:00"},
    # The probe runs on CPU (2026-11-10: the GPU partition was full for hours). Same head, grid and seed; only
    # the device differs (protocols/planteur.md 6.2).
    "probe":      {"gpu": False, "cpus": 16, "mem": "96G", "time": "12:00:00"},
    "lightgbm":   {"gpu": False, "cpus": 16, "mem": "32G", "time": "06:00:00"},
    "tabpfn":     {"gpu": True,  "cpus": 8, "mem": "48G", "time": "08:00:00"},
    "tabicl":     {"gpu": True,  "cpus": 8, "mem": "48G", "time": "08:00:00"},
}


def resources(model, spec):
    """The head's class; the CPU probe's memory follows the input width (its float16 table holds every training
    pixel: up to ~36 M rows on a LOIO fold). Returns None when the probe table can't fit on one node."""
    res = dict(RESOURCES[model])
    if model == "probe":
        info = pl.FEATURES / spec / "info.json"
        d = len(json.loads(info.read_text())["feature_names"]) if info.exists() else 128
        gb = 32 + int(np.ceil(1.15 * 36e6 * d * 2 / 1e9))
        if gb > 200:
            return None
        res["mem"] = f"{max(96, 16 * int(np.ceil(gb / 16)))}G"
    return res


def variants(args):
    if args.fewshot and args.months:
        raise SystemExit("Use --fewshot or --months, not both")
    if args.fewshot:
        return [(f"@x{x:04d}", [f"experiment.fewshot={x}"]) for x in args.fewshot]
    if args.months:
        return [(f"@m{k:02d}", [f"experiment.months={k}"]) for k in args.months]
    return [("", [])]


def plan(args, runs_root):
    tasks = []
    for split in args.splits:
        for spec in args.inputs:
            for suffix, var in variants(args):
                for model in args.models:
                    out = runs_root / split / f"{spec}{suffix}" / model
                    for fold in FOLDS[split]:
                        if (out / f"fold-{fold}.json").exists():
                            continue
                        tasks.append((model, [f"split={split}", f"fold={fold}", f"input={spec}", f"model={model}",
                                              "experiment.confirm_test=true", f"hydra.run.dir={out}", *var,
                                              *args.extra]))
    return tasks


def active(name):
    out = subprocess.run([f"{SLURM}/squeue", "-h", "--me", "-o", "%j"], capture_output=True, text=True).stdout
    return {j.removeprefix(f"pl-{name}-") for j in out.split() if j.startswith(f"pl-{name}-")} - {"collect"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--splits", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    parser.add_argument("--inputs", nargs="+", required=True, choices=sorted(pl.INPUTS))
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(RESOURCES))
    parser.add_argument("--fewshot", nargs="+", type=int, default=[], help="label efficiency: X pixels per class")
    parser.add_argument("--months", nargs="+", type=int, default=[], help="early season: first k monthly mosaics")
    parser.add_argument("--max-running", type=int, default=40)
    parser.add_argument("--nice", type=int, default=100, help="Slurm --nice for GPU arrays")
    parser.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every task")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-active", action="store_true")
    args = parser.parse_args()

    checkout = Path(__file__).resolve().parents[1]
    runs_root = pl.RUNS / "sweeps" / args.name
    task_dir, log_dir = runs_root / "_tasks", runs_root / "_logs"
    for d in (task_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    busy = active(args.name) & set(args.models)
    if busy and not args.allow_active:
        raise SystemExit(f"Arrays for {sorted(busy)} in sweep {args.name} are still queued or running; wait or pass "
                         "--allow-active.")
    repo = frozen_code(checkout, runs_root, args.dry_run)
    tasks = plan(args, runs_root)
    if not tasks:
        print("Nothing to run: every requested result exists.")
    groups = {}
    for model, overrides in tasks:
        spec = next(o.split("=", 1)[1] for o in overrides if o.startswith("input="))
        res = resources(model, spec)
        if res is None:
            print(f"# skipped: {model} on {spec} (the probe table would not fit on one node)")
            continue
        groups.setdefault((model, res["mem"]), []).append(overrides)
    per_array = max(1, args.max_running // max(1, len(groups)))
    jobs = []
    for (model, mem), items in sorted(groups.items()):
        res = {**RESOURCES[model], "mem": mem}
        tfile = task_dir / f"{model}-{mem}-{os.getpid()}.txt"
        if not args.dry_run:
            tfile.write_text("".join(" ".join(shlex.quote(o) for o in ov) + "\n" for ov in items))
        cmd = [f"{SLURM}/sbatch", "--parsable", f"--job-name=pl-{args.name}-{model}",
               f"--array=0-{len(items) - 1}%{per_array}", f"--cpus-per-task={res['cpus']}", f"--mem={res['mem']}",
               f"--time={res['time']}", f"--output={log_dir}/{model}-%A_%a.log",
               "--partition=gpu" if res["gpu"] else "--partition=main"]
        if res["gpu"]:
            cmd += ["--gres=gpu:1", f"--nice={args.nice}"]
        cmd += [f"--chdir={repo}", str(repo / "planteur-sweep.sbatch"), str(tfile)]
        print(f"# {model}: {len(items)} tasks", flush=True)
        jobs.append(submit(cmd, args.dry_run))
    if jobs:
        dep = ["--dependency=afterany:" + ":".join(jobs)] if not args.dry_run else []
        submit([f"{SLURM}/sbatch", "--parsable", *dep, f"--job-name=pl-{args.name}-collect", f"--chdir={checkout}",
                f"--output={log_dir}/collect-%j.log", "--time=00:20:00", "--mem=4G", "--cpus-per-task=1",
                str(repo / "prepare.sbatch"), "tools.collect_planteur", "--sweep", args.name], args.dry_run)
    print(f"{len(tasks)} tasks in {len(groups)} arrays; outputs under {runs_root}")


if __name__ == "__main__":
    main()
