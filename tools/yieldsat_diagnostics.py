"""Uncertainty and error diagnostics for YieldSAT runs, from the saved predictions (protocols/yieldsat.md, 2.10).

    python -m tools.yieldsat_diagnostics --sweep gpu2 [--sweep phase2 ...] [--reps 1000]

Field level throughout (pixel predictions averaged per field, the reported unit). For every complete run:

1. Bootstrap confidence intervals: pooled field R^2 and RMSE with 95% percentile intervals, resampling test fields
   with replacement within each held-out fold (so every fold keeps its size).
2. Per held-out group (year for loyo, region for loro, country for loco; one row each), the Kobayashi-Salam
   decomposition of the MSD, normalised by the group's yield variance:
       b = (mean(p) - mean(y)) / s_y,   kappa = s_p / s_y,   r = corr(y, p)   (population SDs)
       1 - R^2 = B + V + A,   B = b^2,   V = (kappa - 1)^2,   A = 2 kappa (1 - r)
   plus MBE (t/ha), and R^2 after removing the group's mean bias (= 1 - V - A; a diagnostic only, never a score).
   The same decomposition of the pooled predictions is reported with group "pooled".
3. Rank persistence: within each (subset, scheme), every (input, model) is ranked by within-group R^2 for each
   held-out group; mean rank and rank variance across groups say whether a winner wins consistently.

Outputs aggregates only (the repo is public): results/yieldsat/diagnostics/<sweep>-{ci,decomp,ranks}.csv. Held-out
years and countries are named; held-out regions are reported by fold number only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import yieldsat as ys
from tools.collect_yieldsat import run_dirs

GROUP = {"loyo": "year", "loro": "region", "loco": "country"}


def field_table(run):
    """One row per test field: fold, observed and predicted field-mean yield."""
    z = np.load(run / "predictions.npz")
    df = pd.DataFrame({"f": z["field"], "fold": z["fold"], "y": z["y"], "p": z["p"]})
    t = df.groupby("f").agg(fold=("fold", "first"), y=("y", "mean"), p=("p", "mean")).reset_index()
    t["field_name"] = z["field_names"][t.f.to_numpy()]
    return t


def decompose(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sy, sp = y.std(), p.std()
    mse = np.mean((p - y) ** 2)
    out = {"n_fields": len(y), "mbe": p.mean() - y.mean(), "rmse": np.sqrt(mse)}
    if len(y) < 3 or sy == 0:
        return out
    b, k = (p.mean() - y.mean()) / sy, sp / sy
    r = np.corrcoef(y, p)[0, 1] if sp > 0 else 0.0
    B, V, A = b ** 2, (k - 1) ** 2, 2 * k * (1 - r)
    out.update(r2=1 - mse / sy ** 2, b=b, kappa=k, r=r, B=B, V=V, A=A, r2_bias_removed=1 - V - A,
               B_share=B / (B + V + A), V_share=V / (B + V + A), A_share=A / (B + V + A))
    return out


def bootstrap(t, reps, rng):
    """Pooled field R^2 / RMSE with percentile intervals, resampling fields within folds."""
    y, p = t.y.to_numpy(), t.p.to_numpy()
    idx = []
    for _, g in t.groupby("fold").indices.items():
        idx.append(g[rng.integers(0, len(g), size=(reps, len(g)))])
    idx = np.concatenate(idx, axis=1)
    yb, pb = y[idx], p[idx]
    sse = ((pb - yb) ** 2).sum(1)
    sst = ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1)
    r2, rmse = 1 - sse / sst, np.sqrt(sse / idx.shape[1])
    point = decompose(y, p)
    return {"field_r2": point.get("r2"), "field_r2_lo": np.percentile(r2, 2.5), "field_r2_hi": np.percentile(r2, 97.5),
            "field_rmse": point["rmse"], "field_rmse_lo": np.percentile(rmse, 2.5),
            "field_rmse_hi": np.percentile(rmse, 97.5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="append", required=True)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "yieldsat" / "diagnostics")
    args = parser.parse_args()
    fields = pd.read_csv(ys.SPLITS / "fields.csv").set_index("field_shared_name")
    args.out.mkdir(parents=True, exist_ok=True)
    for sweep in args.sweep:
        rng = np.random.default_rng(args.seed)
        root = ys.PROJECT / "runs" / "yieldsat" / "sweeps" / sweep
        ci, dec = [], []
        for run in run_dirs(root):
            if not (run / "metrics.json").exists() or not (run / "predictions.npz").exists():
                continue
            m = json.loads((run / "metrics.json").read_text())
            if m["smoke"]:
                continue
            # the input directory name carries any variant suffix (s2-aligned@d060, s2-aligned@f0.10-s1)
            key = {"subset": m["label"], "scheme": m["scheme"], "input": run.parent.name, "model": m["model"]}
            t = field_table(run)
            ci.append({**key, **bootstrap(t, args.reps, rng)})
            if m["scheme"] not in GROUP:
                continue
            attr = GROUP[m["scheme"]]
            t["group"] = fields.loc[t.field_name, attr].to_numpy()
            for g, sub in t.groupby("group"):
                label = str(g) if attr != "region" else f"fold {int(sub.fold.iloc[0])}"
                dec.append({**key, "group": label, **decompose(sub.y, sub.p)})
            dec.append({**key, "group": "pooled", **decompose(t.y, t.p)})
        if not ci:
            print(f"{sweep}: no complete runs")
            continue
        ci = pd.DataFrame(ci).round(4)
        ci.to_csv(args.out / f"{sweep}-ci.csv", index=False)
        if dec:
            dec = pd.DataFrame(dec)
            ranks = []
            per = dec[dec.group != "pooled"].dropna(subset=["r2"])
            for (subset, scheme), d in per.groupby(["subset", "scheme"]):
                d = d.assign(rank=d.groupby("group").r2.rank(ascending=False, method="average"))
                n_groups = d.group.nunique()
                for (inp, model), e in d.groupby(["input", "model"]):
                    if len(e) == n_groups:  # ranked in every held-out group
                        ranks.append({"subset": subset, "scheme": scheme, "input": inp, "model": model,
                                      "groups": n_groups, "candidates": d.groupby("group").size().min(),
                                      "mean_rank": e["rank"].mean(), "rank_var": e["rank"].var(ddof=0),
                                      "mean_group_r2": e.r2.mean(), "worst_group_r2": e.r2.min()})
            dec.round(4).to_csv(args.out / f"{sweep}-decomp.csv", index=False)
            if ranks:
                pd.DataFrame(ranks).round(4).sort_values(["subset", "scheme", "mean_rank"]).to_csv(
                    args.out / f"{sweep}-ranks.csv", index=False)
        print(f"{sweep}: {len(ci)} runs -> {args.out}")


if __name__ == "__main__":
    main()
