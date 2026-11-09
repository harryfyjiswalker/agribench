"""Summarise EarthShift runs (protocols/earthshift.md, 2.3, 2.4 and 2.8).

    python -m tools.earthshift_results              # best learning rates and the final tables, as Markdown
    python -m tools.earthshift_results --best-lr    # only the chosen rate per model x setting x shift, as JSON
    python -m tools.earthshift_results --csv results/earthshift/final.csv   # public aggregate table (dashboard)

Reads every results.jsonl under $AGRIBENCH_PROJECT/runs/earthshift/ (smoke runs ignored).
Learning rate: the sweep task of each shift type (location: de-geo, temporal: za-season), best mean ID
validation mIoU over seeds (2.3). Final tables: mean +- std over seeds of ID mIoU, OOD mIoU, delta = ID - OOD
and effective robustness rho = OOD - beta(ID) with the paper's fixed lines (2.4). All in mIoU percentage points.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.earthshift_train import RUNS

SWEEP_TASK = {"location": "de-geo", "temporal": "za-season"}
SHIFT_OF = {"de-geo": "location", "de-year": "temporal", "za-season": "temporal"}
# The paper's fitted baselines, OOD = m * ID + b (paper pp.15-16, Tables 5 and 7).
BETA = {("temporal", "frozen"): (1.020, -2.2), ("temporal", "full"): (0.880, 4.3),
        ("location", "frozen"): (0.800, -14.6), ("location", "full"): (0.725, -12.7)}
COLUMNS = {"de-geo": [("Germany → Cambodia", "id", "ood_cambodia"), ("Germany → Denmark", "id", "ood_denmark")],
           "de-year": [("Germany 2018 → 2019", "id", "ood")],
           "za-season": [("South Africa A → B", "id", "ood")]}


def load(root=RUNS):
    rows = []
    for path in sorted(Path(root).rglob("results.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("smoke"):
                rows.append(r)
    return rows


def best_lrs(rows):
    sweep = pd.DataFrame([{"model": r["model"], "setting": r["setting"], "task": r["task"], "lr": r["lr"],
                           "seed": r["seed"], "val": r["val"]["miou"]} for r in rows if r["mode"] == "sweep"])
    out = {}
    if sweep.empty:
        return out, sweep
    for shift, task in SWEEP_TASK.items():
        d = sweep[sweep.task == task]
        for (model, setting), g in d.groupby(["model", "setting"]):
            per_lr = g.groupby("lr").val.agg(["mean", "count"])
            lr = float(per_lr["mean"].idxmax())
            out[(model, setting, shift)] = {"lr": lr, "val_miou": float(per_lr.loc[lr, "mean"]),
                                            "seeds": int(per_lr.loc[lr, "count"]),
                                            "grid": {float(k): round(float(v), 4) for k, v in per_lr["mean"].items()}}
    return out, sweep


def fmt(v):
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}"


def final_tables(rows, chosen):
    lines = []
    finals = [r for r in rows if r["mode"] == "final"]
    for shift in ("location", "temporal"):
        tasks = [t for t, s in SHIFT_OF.items() if s == shift]
        lines.append(f"\n### {shift.capitalize()} shift\n")
        header = "| Model | Setting | LR |"
        sep = "|---|---|---:|"
        for t in tasks:
            for name, _, _ in COLUMNS[t]:
                header += f" {name}: ID | OOD | Δ | ρ |"
                sep += "---:|---:|---:|---:|"
        lines += [header, sep]
        keys = sorted({(r["model"], r["setting"]) for r in finals if r["task"] in tasks})
        for model, setting in keys:
            lr = chosen.get((model, setting, shift), {}).get("lr")
            cells = f"| {model} | {setting} | {lr:g} |" if lr else f"| {model} | {setting} | ? |"
            for t in tasks:
                rs = [r for r in finals if (r["model"], r["setting"], r["task"]) == (model, setting, t)
                      and (lr is None or r["lr"] == lr)]
                for _, id_key, ood_key in COLUMNS[t]:
                    if not rs:
                        cells += " – | – | – | – |"
                        continue
                    idv = np.array([r[id_key]["miou"] for r in rs]) * 100
                    ood = np.array([r[ood_key]["miou"] for r in rs]) * 100
                    m, b = BETA[(shift, setting)]
                    cells += f" {fmt(idv)} | {fmt(ood)} | {fmt(idv - ood)} | {fmt(ood - (m * idv + b))} |"
            lines.append(cells + f" <!-- seeds {len(rs)} -->")
    return "\n".join(lines)


def secondary_table(rows, chosen):
    """Native-resolution mIoU and boundary IoU (2.4, secondary)."""
    lines = ["\n### Secondary: native 256 × 256 mIoU / boundary IoU (ID and OOD test sets)\n",
             "| Model | Setting | Task | Test set | native mIoU | boundary IoU | seeds |", "|---|---|---|---|---:|---:|---:|"]
    finals = [r for r in rows if r["mode"] == "final"]
    for (model, setting, task) in sorted({(r["model"], r["setting"], r["task"]) for r in finals}):
        lr = chosen.get((model, setting, SHIFT_OF[task]), {}).get("lr")
        rs = [r for r in finals if (r["model"], r["setting"], r["task"]) == (model, setting, task) and r["lr"] == lr]
        if not rs:
            continue
        for key, label in [("id", "ID test"), *((ood, name) for name, _, ood in COLUMNS[task])]:
            nat = np.array([r[key]["miou_native"] for r in rs]) * 100
            bnd = np.array([r[key]["boundary_iou_native"] for r in rs]) * 100
            lines.append(f"| {model} | {setting} | {task} | {label} | {fmt(nat)} | {fmt(bnd)} | {len(rs)} |")
    return "\n".join(lines)


CSV_FIELDS = ["dataset", "shift", "task", "test_set", "model", "setting", "lr", "seeds",
              "id_miou_mean", "id_miou_std", "ood_miou_mean", "ood_miou_std", "delta_mean", "delta_std",
              "rho_mean", "rho_std", "id_boundary_iou_mean", "ood_boundary_iou_mean"]


def csv_rows(rows, chosen):
    """One aggregate row per model x setting x task x OOD test set: final runs at the chosen LR only,
    mIoU percentage points (resized headline; boundary IoU at native size), mean and std (ddof=1) over seeds."""
    finals = [r for r in rows if r["mode"] == "final"]
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    out = []
    for (model, setting, task) in sorted({(r["model"], r["setting"], r["task"]) for r in finals}):
        shift = SHIFT_OF[task]
        lr = chosen.get((model, setting, shift), {}).get("lr")
        rs = [r for r in finals if (r["model"], r["setting"], r["task"]) == (model, setting, task) and r["lr"] == lr]
        if not rs:
            continue
        m, b = BETA[(shift, setting)]
        for name, id_key, ood_key in COLUMNS[task]:
            idv = np.array([r[id_key]["miou"] for r in rs]) * 100
            ood = np.array([r[ood_key]["miou"] for r in rs]) * 100
            rho = ood - (m * idv + b)
            vals = {"id_miou": idv, "ood_miou": ood, "delta": idv - ood, "rho": rho}
            row = {"dataset": "earthshift", "shift": shift, "task": task, "test_set": name, "model": model,
                   "setting": setting, "lr": f"{lr:g}", "seeds": len(rs)}
            for k, v in vals.items():
                row[f"{k}_mean"], row[f"{k}_std"] = round(float(np.mean(v)), 2), round(sd(v), 2)
            row["id_boundary_iou_mean"] = round(float(np.mean([r[id_key]["boundary_iou_native"] for r in rs])) * 100, 2)
            row["ood_boundary_iou_mean"] = round(float(np.mean([r[ood_key]["boundary_iou_native"] for r in rs])) * 100, 2)
            out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--best-lr", action="store_true")
    p.add_argument("--root", type=Path, default=RUNS)
    p.add_argument("--csv", type=Path, help="also write the public aggregate table to this path")
    args = p.parse_args()
    rows = load(args.root)
    chosen, sweep = best_lrs(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows(rows, chosen), columns=CSV_FIELDS).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}", file=sys.stderr)
    if args.best_lr:
        json.dump({"/".join(k): v for k, v in chosen.items()}, sys.stdout, indent=2)
        print()
        return
    print("### Chosen learning rates (mean ID validation mIoU over seeds)\n")
    print("| Model | Setting | Shift | LR | val mIoU | seeds |\n|---|---|---|---:|---:|---:|")
    for (model, setting, shift), v in sorted(chosen.items()):
        print(f"| {model} | {setting} | {shift} | {v['lr']:g} | {v['val_miou'] * 100:.2f} | {v['seeds']} |")
    print(final_tables(rows, chosen))
    print(secondary_table(rows, chosen))


if __name__ == "__main__":
    main()
