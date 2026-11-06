"""Standard AgriBench PASTIS-R run: choose encoder, model (head) and protocol by config.

    sbatch bench.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians model=xgboost validation=final

validation=pilot  dev : fit on the train folds with default settings, score the validation fold.
validation=final  test: tune on the validation fold (predefined split, macro IoU), refit on
                        train+validation, score the reserved test fold once, log it in a ledger.
experiment.smoke=true : subsample, and never read the test fold's labels (plumbing check only).
"""
import getpass
import hashlib
import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, jaccard_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline

K = 18  # PASTIS-R crop classes, labels 0..17 after subtracting 1


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def macro_iou(y, p):
    return float(jaccard_score(y, p, labels=np.arange(K), average="macro", zero_division=0))


def score(y, p):
    return {
        "accuracy": float(np.mean(y == p)),
        "macro_iou": macro_iou(y, p),
        "macro_f1": float(f1_score(y, p, labels=np.arange(K), average="macro", zero_division=0)),
    }


def load_features(path, ids):
    """Rows of an extracted feature table for `ids` (the table may contain more parcels)."""
    with np.load(path, allow_pickle=False) as t:
        fid, x = t["ids"], t["features"]
    if fid.ndim != 1 or x.ndim != 2 or x.dtype.kind != "f" or len(fid) != len(x):
        raise ValueError(f"Malformed feature table: {path}")
    order = np.argsort(fid)
    sid = fid[order]
    if len(np.unique(sid)) != len(sid):
        raise ValueError("Duplicate parcel IDs in the feature table")
    pos = np.clip(np.searchsorted(sid, ids), 0, len(sid) - 1)
    if not np.array_equal(sid[pos], ids):
        raise ValueError("The feature table is missing parcels this protocol needs")
    x = x[order[pos]]
    if np.isinf(x).any():
        raise ValueError("Features contain infinities")
    return x, {"file": str(path), "sha256": digest(path), "rows_in_file": int(len(fid))}


def subsample(idx, y, n, seed, stratify):
    if n is None or n >= len(idx):
        return idx
    keep, _ = train_test_split(idx, train_size=int(n), random_state=seed,
                               stratify=y[idx] if stratify else None)
    return np.sort(keep)


def log_test(cfg, enc, inp, metrics):
    ledger = Path(cfg.paths.runs) / "test-ledger.jsonl"
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(),
             "encoder": enc, "input": inp, "model": cfg.model.name,
             "seed": int(cfg.experiment.seed), **metrics}
    try:
        with ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        try:
            os.chmod(ledger, 0o664)  # keep it appendable by collaborators
        except OSError:
            pass
    except OSError as error:
        print(f"WARNING: could not write the test ledger: {error}", flush=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="benchmark")
def main(cfg):
    choices = HydraConfig.get().runtime.choices
    enc, inp = choices.get("encoder"), choices.get("input")
    if not enc or not inp:
        raise SystemExit("Choose features with e.g. encoder=galileo-base input=pastis-r/s2-parcel-medians")
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.dataset.num_classes != K:
        raise ValueError("This runner expects the 18-class PASTIS-R mapping")

    v = cfg.validation
    train_f, val_f, test_f = (list(v[k]) for k in ("train_folds", "val_folds", "test_folds"))
    all_f = train_f + val_f + test_f
    if len(set(all_f)) != len(all_f) or not set(all_f) <= {1, 2, 3, 4, 5}:
        raise ValueError("Train, validation and test folds must be disjoint folds from 1-5")
    test_mode = v.get("mode", "dev") == "test"
    smoke = bool(cfg.experiment.smoke)
    if test_mode and not smoke and 5 in test_f and not cfg.experiment.confirm_test:
        raise SystemExit("validation=final scores the reserved test fold 5. "
                         "Re-run with experiment.confirm_test=true if you mean to.")
    seed = int(cfg.experiment.seed)

    # Fold-5 labels are only read for a real test run.
    root = Path(cfg.dataset.root)
    use = train_f + val_f + (test_f if (test_mode and not smoke) else [])
    meta = pd.read_csv(root / "metadata_parcel.csv", usecols=["ID_PARCEL", "Fold", "Label"])
    meta = meta[meta.Fold.isin(use)].sort_values("ID_PARCEL")
    ids = meta.ID_PARCEL.to_numpy(np.int64)
    fold = meta.Fold.to_numpy()
    y = meta.Label.to_numpy(np.int64) - 1
    if len(np.unique(ids)) != len(ids) or y.min() < 0 or y.max() >= K:
        raise ValueError("Invalid parcel IDs or labels")

    path = Path(cfg.paths.features) / f"{enc}--{inp.replace('/', '-')}.npz"
    if not path.is_file():
        raise SystemExit(
            f"No feature table at {path}\nExtract it first, e.g.:\n"
            f"  sbatch extract.sbatch encoder={enc} input={inp} --inputs <prepared inputs npz> "
            f"--metadata {root}/metadata_parcel.csv --source-dir <encoder source> "
            f"--weights-dir <weights> --output-dir {cfg.paths.features}")
    x, table = load_features(path, ids)

    tr = np.flatnonzero(np.isin(fold, train_f))
    va = np.flatnonzero(np.isin(fold, val_f))
    te = va if smoke else np.flatnonzero(np.isin(fold, test_f))
    limit = cfg.experiment.train_limit or (3000 if smoke else None)
    tr = subsample(tr, y, limit, seed, True)
    if smoke:
        va = subsample(va, y, 1500, seed, False)
        te = va
    if len(np.unique(y[tr])) != K:
        raise ValueError("The training subset is missing crop classes; raise experiment.train_limit")

    pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("model", instantiate(cfg.model.estimator))])
    best, tuning = {}, None
    if test_mode:
        tune_idx = np.concatenate([tr, va])
        fit_idx, eval_idx = (tr if smoke else tune_idx), te
        split = "smoke-plumbing" if smoke else ("test" if 5 in test_f else "rehearsal")
        search = cfg.model.get("search")
        grid = OmegaConf.to_container(search.param_grid, resolve=True) if search and "param_grid" in search else {}
        if grid:
            if smoke:  # keep the code path, shrink the grid
                grid = {k: (vals[:2] if i == 0 else vals[:1]) for i, (k, vals) in enumerate(grid.items())}
            folds = PredefinedSplit(np.r_[np.full(len(tr), -1), np.zeros(len(va), dtype=int)])
            gs = GridSearchCV(pipe, {f"model__{k}": vals for k, vals in grid.items()},
                              scoring=make_scorer(macro_iou), cv=folds, refit=False,
                              error_score="raise", n_jobs=1)
            gs.fit(x[tune_idx], y[tune_idx])
            res = gs.cv_results_
            i = int(np.argmin(res["rank_test_score"]))
            best = {k.removeprefix("model__"): val for k, val in res["params"][i].items()}
            tuning = float(res["mean_test_score"][i])
            pd.DataFrame({"params": [str(p) for p in res["params"]],
                          "macro_iou": res["mean_test_score"]}).to_csv(out / "tuning.csv", index=False)
        model = clone(pipe).set_params(**{f"model__{k}": val for k, val in best.items()})
    else:
        fit_idx, eval_idx, split = tr, va, "validation"
        model = clone(pipe)

    if np.intersect1d(fit_idx, eval_idx).size:
        raise RuntimeError("Evaluation rows overlap the fitting rows")
    model.fit(x[fit_idx], y[fit_idx])
    pred = model.predict(x[eval_idx])
    truth = y[eval_idx]
    metrics = {**score(truth, pred), "split": split, "train_parcels": int(len(fit_idx)),
               "evaluated_parcels": int(len(eval_idx)), "tuning_macro_iou": tuning,
               "best_params": best, "smoke": smoke}
    if split == "validation":
        metrics["validation_parcels"] = int(len(eval_idx))

    pd.DataFrame({"parcel_id": ids[eval_idx], "label_index": truth, "prediction_index": pred}
                 ).to_csv(out / f"{split}-predictions.csv", index=False)
    pd.DataFrame({"parcel_id": ids[fit_idx]}).to_csv(out / "training-ids.csv", index=False)
    (out / "feature-tables.json").write_text(
        json.dumps([{**table, "representation": f"{enc}--{inp}"}], indent=2))
    (out / "metrics.json").write_text(json.dumps([metrics], indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)
    if split == "test":
        log_test(cfg, enc, inp, metrics)


if __name__ == "__main__":
    main()
