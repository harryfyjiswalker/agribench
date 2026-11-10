"""Combine a PLANTEUR sweep's fold results and write the public results table.

    python -m tools.collect_planteur --sweep pa1        # -> results/planteur/pa1.csv

One row per test score (protocols/planteur.md 3.4, 3.8): the default split's test (fold "default"), each LOIO island,
and the LOIO mean (fold "mean": the unweighted mean of the five islands' mIoU-8 and OA, only once all five are in).
Columns: mIoU-8, OA and per-class IoU (%), the val mIoU-8 the setting was chosen by, the chosen setting, the training
regime and pixel count. Aggregates only, because the repository is public: no patch IDs, pixels or predictions.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools import planteur as pl
from tools.collect_yieldsat import write_index
from tools.sweep_planteur import FOLDS

RUN_KEYS = ("split", "input", "model", "regime", "months", "seed")


def params_text(best):
    """Chosen setting as k=v;k=v (no commas: the dashboard's CSV reader splits on them)."""
    return ";".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(best.items()))


def run_rows(d, sweep):
    folds = [json.loads(f.read_text()) for f in sorted(d.glob("fold-*.json"))]
    folds = [f for f in folds if not f["smoke"] and f["stage"] == "test"]
    if not folds:
        return [], None
    first = folds[0]
    for f in folds[1:]:
        for k in RUN_KEYS:
            if f.get(k) != first.get(k):
                return [], f"{d}: folds disagree on {k}"
    rows = []
    for f in folds:
        r = f["fold_result"]
        row = {"dataset": "planteur", "split": first["split"], "fold": r["fold"], "input": d.parent.name,
               "model": first["model"], "regime": first["regime"], "months": first.get("months"),
               "miou8": round(r["test"]["miou8"], 2), "oa": round(r["test"]["oa"], 2),
               **{f"iou_{c}": round(v, 2) for c, v in enumerate(r["test"]["iou"])},
               "val_miou8": round(r["val"]["miou8"], 2), "params": params_text(r["best_params"]),
               "train_pixels": r["train_pixels"], "test_patches": r["test_patches"], "test_pixels": r["test_pixels"],
               "test_pixels_no_input": r["test_pixels_no_input"], "n_features": f["provenance"]["n_features"],
               "folds": 1, "task_hours": round(r["seconds"] / 3600, 3),
               "git_commit": (f["provenance"].get("git_commit") or "")[:10], "sweep": sweep}
        rows.append(row)
    if first["split"] == "loio":
        have = {r["fold"] for r in rows}
        if have == set(FOLDS["loio"]):
            m = dict(rows[0], fold="mean", folds=5, params="", train_pixels="", test_patches="", test_pixels="",
                     test_pixels_no_input="", task_hours=round(sum(r["task_hours"] for r in rows), 3),
                     git_commit=";".join(sorted({r["git_commit"] for r in rows})))
            for c in ["miou8", "oa", "val_miou8", *[f"iou_{c}" for c in range(pl.N_CLASSES)]]:
                m[c] = round(float(np.mean([r[c] for r in rows])), 2)
            rows.append(m)
    return rows, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "planteur")
    args = parser.parse_args()
    root = pl.RUNS / "sweeps" / args.sweep
    rows, problems = [], []
    for d in sorted({f.parent for f in root.glob("*/*/*/fold-*.json")}):
        r, problem = run_rows(d, args.sweep)
        rows += r
        if problem:
            problems.append(problem)
    if problems:
        print(f"{len(problems)} runs skipped:\n  " + "\n  ".join(problems))
    if not rows:
        print("No finished test results yet.")
        return
    order = {f: i for i, f in enumerate(["default", *FOLDS["loio"], "mean"])}
    df = pd.DataFrame(rows)
    df["_o"] = df.fold.map(order)
    df = df.sort_values(["split", "input", "model", "_o"]).drop(columns="_o")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.sweep}.csv"
    df.to_csv(path, index=False)
    (root / "summary.csv").write_text(df.to_csv(index=False))
    print(f"{len(df)} rows -> {path} ({time.strftime('%Y-%m-%d %H:%M')})")
    write_index(args.out.parent)


if __name__ == "__main__":
    main()
