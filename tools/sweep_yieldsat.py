"""Run a grid of YieldSAT experiments as Slurm job arrays, sized per head, then collect the results.

    python -m tools.sweep_yieldsat --name phase1 --inputs s2 galileo mind galileo+mind --models probe lightgbm
    python -m tools.sweep_yieldsat --name phase1 ... --dry-run      # print the plan and cost estimate only

Every (subset, scheme, input, model) combination writes to runs/yieldsat/sweeps/<name>/<subset>/<scheme>/<input>/
<model>/. CPU heads run one array task per outer fold, so the folds of a run execute in parallel; GPU heads (TabPFN,
TabICL) run all folds of a run in one task, so each GPU job loads the model once. Tasks whose result already exists
are skipped, so re-running the same command resumes a sweep. At most --max-running tasks of the sweep run at once (split evenly over its arrays)
(Anunna guidance: keep Lustre-heavy arrays at about 200). A final CPU job (afterany) combines the folds of every
run and writes the public aggregate table results/yieldsat/<name>.csv.

Every sweep submission runs from a frozen copy of the code at the checkout's HEAD commit
(runs/yieldsat/sweeps/<name>/_code/<commit>/), so editing or pulling the checkout never changes tasks that are
already queued, and every fold records the commit it ran with. The checkout must be clean.

Resources per head are in RESOURCES below. Billing on Anunna is for what is *requested*, and FB-IT's efficiency
score weighs the time limit most, so keep these close to measured use (tools/collect_yieldsat.py --efficiency).
"""
import argparse
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from tools import yieldsat as ys

SLURM = "/cm/shared/apps/slurm/current/bin"
SCHEMES = ("cv10", "loyo", "loro")
# Measured on 2026-11-09 (see protocols/yieldsat.md 2.9); "wide" applies when the input has > 600 columns.
RESOURCES = {
    "mean":     {"gpu": False, "per": "run",  "cpus": 1, "mem": "8G",  "wide_mem": "8G",  "time": "00:20:00"},
    "probe":    {"gpu": False, "per": "fold", "cpus": 4, "mem": "12G", "wide_mem": "24G", "time": "00:30:00"},
    "lightgbm": {"gpu": False, "per": "fold", "cpus": 8, "mem": "12G", "wide_mem": "24G", "time": "01:30:00"},
    "xgboost":  {"gpu": False, "per": "fold", "cpus": 8, "mem": "12G", "wide_mem": "24G", "time": "01:30:00"},
    "rf":       {"gpu": False, "per": "fold", "cpus": 8, "mem": "12G", "wide_mem": "24G", "time": "01:30:00"},
    "extra_trees": {"gpu": False, "per": "fold", "cpus": 8, "mem": "12G", "wide_mem": "24G", "time": "01:30:00"},
    "tabpfn":   {"gpu": True,  "per": "run",  "cpus": 4, "mem": "48G", "wide_mem": "64G", "time": "04:00:00"},
    "tabicl":   {"gpu": True,  "per": "run",  "cpus": 4, "mem": "48G", "wide_mem": "64G", "time": "04:00:00"},
}


def width(spec):
    """Number of input columns for an input spec, from the cache metadata of any subset."""
    import json
    n = 0
    v1 = json.loads((ys.FEATURES / "v1" / f"{ys.SUBSETS[0]}.json").read_text())
    for block in spec.split("+"):
        if block in v1["inputs"]:
            n += len(v1["inputs"][block])
        else:
            info = ys.FEATURES / block / f"{ys.SUBSETS[0]}.json"
            if not info.exists():
                raise SystemExit(f"Input block '{block}' has no cache yet ({info})")
            n += len(json.loads(info.read_text())["feature_names"])
    return n


def frozen_code(checkout, runs_root, dry):
    """Copy the checkout's HEAD into runs_root/_code/<commit>/ (git archive) and return that directory."""
    git = ["git", "-C", str(checkout)]
    if subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"], capture_output=True,
                      text=True).stdout.strip():
        raise SystemExit("The checkout has uncommitted changes; commit them first (sweeps run from a commit).")
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    code = runs_root / "_code" / sha
    if not code.exists() and not dry:
        tmp = runs_root / "_code" / f".{sha}.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(git + ["archive", "--format=tar", sha], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
        (tmp / "COMMIT").write_text(sha + "\n")
        tmp.rename(code)
    print(f"# code: {sha[:10]} -> {code}", flush=True)
    return code


def plan(args, runs_root):
    tasks = []
    for subset in args.subsets:
        for scheme in args.schemes:
            folds = sorted(pd.read_csv(ys.SPLITS / subset / f"{scheme}.csv").fold.unique())
            for spec in args.inputs:
                for model in args.models:
                    out = runs_root / subset / scheme / spec / model
                    res = RESOURCES[model]
                    if (out / "metrics.json").exists():
                        continue
                    base = [f"subset={subset}", f"scheme={scheme}", f"input={spec}", f"model={model}",
                            f"hydra.run.dir={out}"] + args.extra
                    if res["per"] == "fold":
                        for k in folds:
                            if not (out / f"fold-{k}.json").exists():
                                tasks.append((model, spec, base + [f"experiment.fold={k}"]))
                    else:
                        tasks.append((model, spec, base))
    return tasks


def submit(cmd, dry):
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return "DRYRUN"
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--subsets", nargs="+", default=ys.SUBSETS, choices=ys.SUBSETS)
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(RESOURCES))
    parser.add_argument("--max-running", type=int, default=200)
    parser.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkout = Path(__file__).resolve().parents[1]
    runs_root = ys.PROJECT / "runs" / "yieldsat" / "sweeps" / args.name
    task_dir = runs_root / "_tasks"
    log_dir = runs_root / "_logs"
    for d in (task_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    repo = frozen_code(checkout, runs_root, args.dry_run)
    widths = {spec: width(spec) for spec in args.inputs}
    tasks = plan(args, runs_root)
    if not tasks:
        print("Nothing to run: every requested result exists.")
    jobs = []
    groups = {}
    for model, spec, overrides in tasks:
        res = RESOURCES[model]
        wide = widths[spec] > 600
        groups.setdefault((model, wide), []).append(overrides)
    per_array = max(1, args.max_running // max(1, len(groups)))  # the cap is for the whole sweep, not per array
    for (model, wide), items in sorted(groups.items()):
        res = RESOURCES[model]
        tag = f"{model}-{'wide' if wide else 'narrow'}"
        tfile = task_dir / f"{tag}-{os.getpid()}.txt"
        if not args.dry_run:
            tfile.write_text("".join(" ".join(shlex.quote(o) for o in ov) + "\n" for ov in items))
        cmd = [f"{SLURM}/sbatch", "--parsable", f"--job-name=ys-{args.name}-{tag}",
               f"--array=0-{len(items) - 1}%{per_array}", f"--cpus-per-task={res['cpus']}",
               f"--mem={res['wide_mem'] if wide else res['mem']}", f"--time={res['time']}",
               f"--output={log_dir}/{tag}-%A_%a.log", "--partition=gpu" if res["gpu"] else "--partition=main"]
        if res["gpu"]:
            cmd.append("--gres=gpu:1")
        cmd += [f"--chdir={repo}", str(repo / "sweep.sbatch"), str(tfile)]
        print(f"# {tag}: {len(items)} tasks", flush=True)
        jobs.append(submit(cmd, args.dry_run))
    if jobs:
        dep = ["--dependency=afterany:" + ":".join(jobs)] if not args.dry_run else []
        submit([f"{SLURM}/sbatch", "--parsable", *dep, f"--job-name=ys-{args.name}-collect", f"--chdir={checkout}",
                f"--output={log_dir}/collect-%j.log", "--time=00:30:00", "--mem=8G", "--cpus-per-task=1",
                str(repo / "prepare.sbatch"), "tools.collect_yieldsat", "--sweep", args.name], args.dry_run)
    print(f"{len(tasks)} tasks in {len(groups)} arrays; outputs under {runs_root}")


if __name__ == "__main__":
    main()
