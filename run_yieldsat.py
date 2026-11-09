"""YieldSAT runner: one model x input x subset x scheme, scored exactly as protocols/yieldsat.md sets out.

    sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=rf
    sbatch yieldsat.sbatch subset=GER-W scheme=loro input=s2-adm model=xgboost
    sbatch yieldsat.sbatch crop=soybean scheme=loco input=s2 model=rf experiment.confirm_test=true
    add experiment.smoke=true for a quick plumbing check (scores are meaningless)

For every outer fold of the frozen fold file:
  1. take at most `pixel_cap` random pixels from each training field (2.7);
  2. tune the model's grid with an inner split of the training fields that mirrors the outer scheme (2.6),
     selecting by pixel RMSE, then refit the best setting on all training pixels;
  3. predict every pixel of the test fields and score R2 / RMSE at pixel and field level (2.4).
Results: per-fold metrics, mean +/- std over folds (ddof=0), pooled metrics, and for LOCO one row per held-out
country (2.3b). LOCO runs are the final evaluation: they need experiment.confirm_test=true and are logged.
"""
import getpass
import json
import subprocess
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from tools import yieldsat as ys

SCHEMES = ("cv10", "loyo", "loro", "loco")
LEVELS = ("pixel", "field")


def score(y, p):
    return {"r2": float(r2_score(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def score_levels(y, p, field):
    out = {f"pixel_{k}": v for k, v in score(y, p).items()}
    per_field = pd.DataFrame({"f": field, "y": y, "p": p}).groupby("f").mean()
    out.update({f"field_{k}": v for k, v in score(per_field.y.to_numpy(), per_field.p.to_numpy()).items()})
    out.update(test_pixels=int(len(y)), test_fields=int(len(per_field)))
    return out


class Data:
    """The cached pixels of one or more subsets, addressed by global row and global field id."""

    def __init__(self, cfg, subsets, columns):
        feat = Path(cfg.paths.features)
        self.parts, self.provenance, names = [], [], None
        field_names, starts, stops, part_of_field = [], [], [], []
        targets = []
        for i, s in enumerate(subsets):
            info_path = feat / f"{s}.json"
            info = json.loads(info_path.read_text())
            if names is None:
                names = info["feature_names"]
            elif info["feature_names"] != names:
                raise ValueError(f"Feature columns differ in {s}")
            x = np.load(feat / f"{s}.npy", mmap_mode="r")
            m = np.load(feat / f"{s}-meta.npz")
            fidx = m["field_idx"]
            if np.any(np.diff(fidx) < 0):
                raise ValueError(f"{s}: cache rows are not grouped by field")
            b = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
            for j, f in enumerate(m["fields"]):
                field_names.append(str(f))
                starts.append(b[j])
                stops.append(b[j + 1])
                part_of_field.append(i)
            self.parts.append(x)
            targets.append(m["target"])
            self.provenance.append({"subset": s, "cache_json_sha256": ys.sha256(info_path), "rows": int(len(fidx))})
        self.columns = np.asarray(info["inputs"][columns] if isinstance(columns, str) else columns)
        self.feature_names = [names[c] for c in self.columns]
        self.field_names = np.array(field_names)
        self.start, self.stop = np.array(starts), np.array(stops)
        self.part = np.array(part_of_field)
        self.targets = targets

    def rows(self, fields, cap=None, rng=None):
        """(part, local_row) pairs for the given global field ids, optionally at most `cap` per field."""
        out = []
        for f in fields:
            r = np.arange(self.start[f], self.stop[f])
            if cap is not None and len(r) > cap:
                r = np.sort(rng.choice(r, cap, replace=False))
            out.append(np.c_[np.full(len(r), self.part[f]), r, np.full(len(r), f)])
        return np.concatenate(out) if out else np.empty((0, 3), int)

    def load(self, rows):
        x = np.empty((len(rows), len(self.columns)), np.float32)
        y = np.empty(len(rows), np.float32)
        for p in np.unique(rows[:, 0]):
            sel = np.flatnonzero(rows[:, 0] == p)
            local = rows[sel, 1]
            x[sel] = np.asarray(self.parts[p][local])[:, self.columns]
            y[sel] = self.targets[p][local]
        return x, y


def inner_splits(scheme, attrs, train_fields_per_row, seed, n_train_countries):
    """Inner tuning folds over the training rows, mirroring the outer scheme (2.6)."""
    a = attrs.loc[train_fields_per_row]
    if scheme == "cv10":
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        return list(cv.split(a, y=a.farm.to_numpy(), groups=a.index.to_numpy()))
    if scheme == "loco" and n_train_countries >= 2:
        groups = a.country.to_numpy()
    else:
        groups = {"loyo": a.year, "loro": a.region, "loco": a.farm}[scheme].to_numpy()
    k = min(3, len(np.unique(groups)))
    if k < 2:
        raise ValueError(f"Only {k} inner group(s) for scheme {scheme}; cannot tune")
    return list(GroupKFold(n_splits=k).split(a, groups=groups))


def git_commit(directory):
    try:
        return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


@hydra.main(version_base="1.3", config_path="conf/yieldsat", config_name="config")
def main(cfg):
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    scheme, smoke, seed = cfg.scheme, bool(cfg.experiment.smoke), int(cfg.experiment.seed)
    if scheme not in SCHEMES:
        raise SystemExit(f"scheme must be one of {SCHEMES}")
    if cfg.input not in ("s2", "s2-adm"):
        raise SystemExit("input must be s2 or s2-adm")
    splits = Path(cfg.paths.splits)
    fields = pd.read_csv(splits / "fields.csv").set_index("field_shared_name")

    if scheme == "loco":
        if not cfg.crop or cfg.subset:
            raise SystemExit("scheme=loco needs crop=<corn|soybean|wheat> and no subset")
        if cfg.crop not in ys.LOCO_CROPS:
            raise SystemExit(f"crop must be one of {ys.LOCO_CROPS}")
        if not smoke and not cfg.experiment.confirm_test:
            raise SystemExit("scheme=loco is the final evaluation (protocol 2.3b). Choose models on cv10/loyo/loro "
                             "first, then re-run with experiment.confirm_test=true.")
        fold_file = splits / "loco" / f"{cfg.crop}.csv"
        subsets = sorted(fields[fields.crop == cfg.crop].subset.unique())
        label = cfg.crop
    else:
        if not cfg.subset or cfg.crop:
            raise SystemExit(f"scheme={scheme} needs subset=<one of {ys.SUBSETS}> and no crop")
        if cfg.subset not in ys.SUBSETS:
            raise SystemExit(f"subset must be one of {ys.SUBSETS}")
        fold_file = splits / cfg.subset / f"{scheme}.csv"
        subsets = [cfg.subset]
        label = cfg.subset

    folds = pd.read_csv(fold_file).set_index("field_shared_name").fold
    data = Data(cfg, subsets, cfg.input)
    if set(data.field_names) != set(folds.index):
        raise ValueError("Fields in the feature cache and the fold file differ")
    gfold = folds.loc[data.field_names].to_numpy()
    attrs = fields.loc[data.field_names, ["country", "year", "farm", "region"]].reset_index(drop=True)

    head = cfg.model
    steps = ([("impute", SimpleImputer(strategy="median", keep_empty_features=True))] if head.get("impute", True)
             else []) + [("model", instantiate(head.estimator))]
    pipe = Pipeline(steps)
    search = head.get("search")
    grid = OmegaConf.to_container(search.param_grid, resolve=True) if search and "param_grid" in search else {}
    cap = 16 if smoke else int(cfg.experiment.pixel_cap)
    test_cap = 200 if smoke else None
    if smoke and grid:
        grid = {k: (v[:2] if i == 0 else v[:1]) for i, (k, v) in enumerate(grid.items())}
    fold_ids = sorted(np.unique(gfold))
    if smoke:
        fold_ids = fold_ids[:2]

    per_fold, tuning_rows = [], []
    pred_parts = {"fold": [], "field": [], "y": [], "p": []}
    for k in fold_ids:
        t0 = time.time()
        rng = np.random.default_rng(seed + 1000 * int(k))
        train_f = np.flatnonzero(gfold != k)
        test_f = np.flatnonzero(gfold == k)
        tr = data.rows(train_f, cap, rng)
        x_tr, y_tr = data.load(tr)
        best, inner_score = {}, None
        if grid:
            n_countries = attrs.loc[train_f].country.nunique()
            cv = inner_splits(scheme, attrs, tr[:, 2], seed, n_countries)
            gs = GridSearchCV(pipe, {f"model__{p}": v for p, v in grid.items()}, cv=cv, refit=True, n_jobs=1,
                              scoring="neg_root_mean_squared_error", error_score="raise")
            gs.fit(x_tr, y_tr)
            model = gs.best_estimator_
            best = {p.removeprefix("model__"): v for p, v in gs.best_params_.items()}
            inner_score = float(-gs.best_score_)
            for params, mean in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
                tuning_rows.append({"fold": int(k), "params": json.dumps(params, default=str), "inner_rmse": -mean})
        else:
            model = clone(pipe).fit(x_tr, y_tr)
        del x_tr

        # Predict every pixel of the test fields, in batches (2.7: evaluation is never capped).
        te = data.rows(test_f, test_cap, rng)
        ys_, ps_ = [], []
        for a in range(0, len(te), int(cfg.experiment.predict_batch)):
            x_te, y_te = data.load(te[a:a + int(cfg.experiment.predict_batch)])
            ps_.append(model.predict(x_te).astype(np.float32))
            ys_.append(y_te)
        y, p = np.concatenate(ys_), np.concatenate(ps_)
        row = {"fold": int(k), **score_levels(y, p, te[:, 2]), "train_pixels": int(len(tr)),
               "train_fields": int(len(train_f)), "best_params": best, "inner_rmse": inner_score,
               "seconds": round(time.time() - t0, 1)}
        if scheme == "loco":
            row["held_out_country"] = str(attrs.loc[test_f[0]].country)
        per_fold.append(row)
        for key, v in (("fold", np.full(len(y), k)), ("field", te[:, 2]), ("y", y), ("p", p)):
            pred_parts[key].append(v)
        print(json.dumps(row, default=str), flush=True)

    preds = {key: np.concatenate(v) for key, v in pred_parts.items()}
    np.savez(out / "predictions.npz", field_names=data.field_names, **preds)
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(out / "tuning.csv", index=False)
    df = pd.DataFrame(per_fold)
    metric_cols = [f"{lvl}_{m}" for lvl in LEVELS for m in ("r2", "rmse")]
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in metric_cols}
    pooled = score_levels(preds["y"], preds["p"], preds["field"])
    result = {
        "dataset": "yieldsat", "label": label, "scheme": scheme, "input": cfg.input, "model": head.name,
        "split": "smoke" if smoke else ("final" if scheme == "loco" else "development"),
        "smoke": smoke, "seed": seed, "pixel_cap": cap, "folds": len(per_fold),
        "summary": summary, "pooled": pooled, "per_fold": per_fold,
        "provenance": {"fold_file": str(fold_file), "fold_file_sha256": ys.sha256(fold_file),
                       "folds_manifest_sha256": ys.sha256(splits / "manifest.json"),
                       "caches": data.provenance, "n_features": int(len(data.columns)),
                       "git_commit": git_commit(Path(__file__).parent)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)

    if scheme == "loco" and not smoke:
        ledger = Path(cfg.paths.runs) / "loco-ledger.jsonl"
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(), "crop": cfg.crop,
                 "input": cfg.input, "model": head.name, "seed": seed, "run": str(out),
                 "per_country": {r["held_out_country"]: {c: r[c] for c in metric_cols} for r in per_fold}}
        try:
            with ledger.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            ledger.chmod(0o664)
        except OSError as error:
            print(f"WARNING: could not write the LOCO ledger: {error}", flush=True)


if __name__ == "__main__":
    main()
