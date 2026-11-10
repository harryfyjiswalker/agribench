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


# Files that determine a YieldSAT result. Folds run with different commits are combined only when none of these
# differ between the commits (e.g. a change to the dashboard or the publish script doesn't matter).
MODEL_CODE = ["run_yieldsat.py", "conf/yieldsat", "tools/yieldsat.py", "tools/yieldsat_results.py"]


# Commits that touched MODEL_CODE without changing any result a run could produce before them, one per line with
# the reason, in a file outside MODEL_CODE (so adding a commit to it doesn't itself change the model code).
NEUTRAL_FILE = Path(__file__).with_name("yieldsat_result_neutral.txt")


def result_neutral():
    if not NEUTRAL_FILE.exists():
        return set()
    return {line.split()[0] for line in NEUTRAL_FILE.read_text().splitlines() if line.strip() and not line.startswith("#")}


def same_model_code(commits, repo=Path(__file__).resolve().parents[1]):
    """True if every commit has the same MODEL_CODE, or differs only by result-neutral commits (NEUTRAL_FILE) (needs the commits
    in the repo at `repo`; dirty or unknown commits never match)."""
    import subprocess
    commits = sorted(commits, key=str)
    if any(c is None or str(c).endswith("-dirty") for c in commits):
        return False
    base = commits[0]
    for c in commits[1:]:
        r = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", base, c, "--", *MODEL_CODE], capture_output=True)
        if r.returncode == 0:
            continue
        if r.returncode != 1:  # unknown commit: can't tell, so refuse
            return False
        # The model code differs: fine only if every commit that changed it between the two is result-neutral.
        log = subprocess.run(["git", "-C", str(repo), "log", "--no-merges", "--format=%H", f"{base}...{c}", "--",
                              *MODEL_CODE], capture_output=True, text=True)
        changed, neutral = log.stdout.split(), result_neutral()
        if log.returncode != 0 or not changed or any(h not in neutral for h in changed):
            return False
    return True


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
    commits = {f["provenance"].get("git_commit") for f in folds}
    if len(commits) > 1 and not same_model_code(commits):
        raise ValueError(f"{out}: folds were run with code versions that differ in model code ({sorted(map(str, commits))})")
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
              "provenance": first["provenance"] | {"slurm_jobs": sorted({str(f["provenance"].get("slurm_job")) for f in folds}),
                                                 "git_commits": sorted(map(str, commits))}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
