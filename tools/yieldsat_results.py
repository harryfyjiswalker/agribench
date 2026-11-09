"""YieldSAT scoring and result files (protocols/yieldsat.md, 2.4 and 2.8).

A run directory holds one fold-<k>.json (+ predictions, + tuning table) per outer fold, written by run_yieldsat.py,
and combine() turns a complete set into metrics.json: per-fold metrics, mean +/- std over folds (ddof=0) and
pooled metrics over all test pixels. Folds can therefore run as separate Slurm array tasks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

LEVELS = ("pixel", "field")
METRICS = [f"{lvl}_{m}" for lvl in LEVELS for m in ("r2", "rmse")]
# Keys that must agree between the folds of one run.
RUN_KEYS = ("dataset", "label", "scheme", "input", "model", "split", "smoke", "seed", "pixel_cap", "folds_in_scheme",
            "season_cutoff", "train_fraction")  # the last two are absent (None / 1.0) in older fold files
DEFAULTS = {"season_cutoff": None, "train_fraction": 1.0}


def score(y, p):
    return {"r2": float(r2_score(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def score_levels(y, p, field):
    out = {f"pixel_{k}": v for k, v in score(y, p).items()}
    per_field = pd.DataFrame({"f": field, "y": y, "p": p}).groupby("f").mean()
    out.update({f"field_{k}": v for k, v in score(per_field.y.to_numpy(), per_field.p.to_numpy()).items()})
    out.update(test_pixels=int(len(y)), test_fields=int(len(per_field)))
    return out


def write_fold(out, common, row, field_names, field, y, p, tuning_rows):
    out = Path(out)
    k = row["fold"]
    np.savez(out / f"fold-{k}-predictions.npz", field_names=field_names, field=field, y=y, p=p)
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(out / f"fold-{k}-tuning.csv", index=False)
    tmp = out / f".fold-{k}.json.tmp"
    tmp.write_text(json.dumps({**common, "fold_result": row}, indent=2, default=str) + "\n")
    tmp.rename(out / f"fold-{k}.json")  # written last: its presence means the fold is complete


def fold_files(out):
    return sorted(Path(out).glob("fold-*.json"), key=lambda f: int(f.stem.split("-")[1]))


def combine(out, require_complete=True):
    """Combine the fold files in `out` into metrics.json (+ predictions.npz, tuning.csv). Returns the result."""
    out = Path(out)
    folds = [json.loads(f.read_text()) for f in fold_files(out)]
    if not folds:
        raise FileNotFoundError(f"No fold results in {out}")
    first = folds[0]
    for f in folds[1:]:
        for key in RUN_KEYS:
            if f.get(key, DEFAULTS.get(key)) != first.get(key, DEFAULTS.get(key)):
                raise ValueError(f"{out}: folds disagree on {key}")
        if f["provenance"]["fold_file_sha256"] != first["provenance"]["fold_file_sha256"]:
            raise ValueError(f"{out}: folds come from different fold files")
        if f["provenance"].get("git_commit") != first["provenance"].get("git_commit"):
            raise ValueError(f"{out}: folds were run with different code versions")
    per_fold = sorted((f["fold_result"] for f in folds), key=lambda r: r["fold"])
    expected = first["folds_in_scheme"]
    if require_complete and not first["smoke"] and len(per_fold) != expected:
        raise ValueError(f"{out}: {len(per_fold)} of {expected} folds present")
    parts = [np.load(out / f"fold-{r['fold']}-predictions.npz") for r in per_fold]
    preds = {k: np.concatenate([p[k] for p in parts]) for k in ("field", "y", "p")}
    preds["fold"] = np.concatenate([np.full(len(p["y"]), r["fold"]) for p, r in zip(parts, per_fold)])
    np.savez(out / "predictions.npz", field_names=parts[0]["field_names"], **preds)
    tuning = sorted(out.glob("fold-*-tuning.csv"))
    if tuning:
        pd.concat([pd.read_csv(t) for t in tuning]).to_csv(out / "tuning.csv", index=False)
    df = pd.DataFrame(per_fold)
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in METRICS}
    result = {**{k: first.get(k, DEFAULTS.get(k)) for k in RUN_KEYS}, "folds": len(per_fold), "summary": summary,
              "pooled": score_levels(preds["y"], preds["p"], preds["field"]), "per_fold": per_fold,
              "cost": {"task_seconds": float(df.seconds.sum()), "max_peak_rss_gb": float(df.get("peak_rss_gb", pd.Series([np.nan])).max())},
              "provenance": first["provenance"] | {"slurm_jobs": sorted({str(f["provenance"].get("slurm_job")) for f in folds})}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
