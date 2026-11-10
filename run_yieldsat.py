"""YieldSAT runner: one model x input x subset x scheme, scored exactly as protocols/yieldsat.md sets out.

    sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=rf
    sbatch yieldsat.sbatch subset=GER-W scheme=loro input=s2-adm model=xgboost
    sbatch yieldsat.sbatch crop=soybean scheme=loco input=s2 model=rf experiment.confirm_test=true
    sbatch yieldsat.sbatch subset=GER-W scheme=loyo input=galileo+mind model=probe
    add experiment.smoke=true for a quick plumbing check (scores are meaningless)
    add experiment.fold=3 to run one outer fold only (the sweep tool runs folds as parallel array tasks;
    tools/collect_yieldsat.py combines them)

Inputs are joined with "+": s2 and s2-adm are column groups of the v1 cache; any other name is a representation
folder under features/yieldsat/ (latlon, mind, satclip, galileo, ...) whose rows align with the v1 cache.

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
import os
import re
import resource
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
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools import yieldsat as ys
from tools import yieldsat_results as res

SCHEMES = ("cv10", "loyo", "loro", "loco")


class Block:
    """One input block of one subset: columns of a dense (pixels, d) array, or rows of a shared table via an index."""

    def __init__(self, root, name, subset, v1_info, v1_meta_sha):
        if name in v1_info["inputs"]:
            self.array = np.load(root / "v1" / f"{subset}.npy", mmap_mode="r")
            self.columns = np.asarray(v1_info["inputs"][name])
            self.names = [v1_info["feature_names"][c] for c in self.columns]
            self.index = None
            self.provenance = {"block": name, "cache_json_sha256": ys.sha256(root / "v1" / f"{subset}.json")}
            return
        info_path = root / name / f"{subset}.json"
        if not info_path.exists():
            raise SystemExit(f"Unknown input '{name}': no {info_path}")
        info = json.loads(info_path.read_text())
        if info.get("v1_meta_sha256") != v1_meta_sha:
            raise ValueError(f"{info_path} was built for different v1 cache rows")
        self.names = info["feature_names"]
        self.columns = None
        if info["kind"] == "dense":
            self.array = np.load(root / name / f"{subset}.npy", mmap_mode="r")
            self.index = None
        elif info["kind"] == "indexed":
            self.array = np.load(root / name / "table.npy")  # small: one row per grid cell
            self.index = np.load(root / name / f"{subset}-index.npy")
        else:
            raise ValueError(f"{info_path}: unknown kind {info['kind']}")
        if (len(self.index) if self.index is not None else len(self.array)) != info["rows"]:
            raise ValueError(f"{info_path}: row count differs from the stored array")
        self.provenance = {"block": name, "json_sha256": ys.sha256(info_path)}

    def take(self, local):
        if self.index is not None:
            x = self.array[self.index[local]]
            return x[:, self.columns] if self.columns is not None else x
        x = np.asarray(self.array[local])
        return x[:, self.columns] if self.columns is not None else x


class Data:
    """The pixels of one or more subsets, addressed by global row and global field id, for an input spec such as
    "s2", "galileo" or "galileo+mind"."""

    def __init__(self, cfg, subsets, spec, align=False, season_cutoff=None):
        """align: when subsets have different columns (inputs on a per-subset time grid, such as s2-aligned), keep
        only the columns all subsets share, by name (used by scheme=loco, which pools countries).
        season_cutoff: keep only columns named *_d<days> with days <= cutoff (days after sowing; early season)."""
        root = Path(cfg.paths.features)
        blocks = spec.split("+")
        if len(set(blocks)) != len(blocks):
            raise SystemExit(f"Input {spec} repeats a block")
        self.parts, self.provenance = [], []
        field_names, starts, stops, part_of_field, targets = [], [], [], [], []
        for i, s in enumerate(subsets):
            v1_info = json.loads((root / "v1" / f"{s}.json").read_text())
            meta_path = root / "v1" / f"{s}-meta.npz"
            meta_sha = ys.sha256(meta_path)
            part = [Block(root, b, s, v1_info, meta_sha) for b in blocks]
            names = [n for blk in part for n in blk.names]
            if i == 0:
                self.feature_names = names
            elif names != self.feature_names:
                if not align:
                    raise ValueError(f"Feature columns differ in {s}")
                keep = set(names)
                self.feature_names = [n for n in self.feature_names if n in keep]
            m = np.load(meta_path)
            fidx = m["field_idx"]
            if np.any(np.diff(fidx) < 0):
                raise ValueError(f"{s}: cache rows are not grouped by field")
            b = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
            for j, f in enumerate(m["fields"]):
                field_names.append(str(f))
                starts.append(b[j])
                stops.append(b[j + 1])
                part_of_field.append(i)
            self.parts.append(part)
            targets.append(m["target"])
            self.provenance.append({"subset": s, "v1_meta_sha256": meta_sha, "rows": int(len(fidx)),
                                    "blocks": [blk.provenance for blk in part]})
        if season_cutoff is not None:
            days = [re.search(r"_d([+-]\d+)$", n) for n in self.feature_names]
            if not all(days):
                raise SystemExit(f"experiment.season_cutoff needs an input whose columns are all *_d<days after "
                                 f"sowing> (s2-aligned, biophys); {spec} is not")
            self.feature_names = [n for n, d in zip(self.feature_names, days) if int(d.group(1)) <= season_cutoff]
            if not self.feature_names:
                raise SystemExit(f"No columns at or before day {season_cutoff}")
        if align or season_cutoff is not None:
            keep = set(self.feature_names)
            for part in self.parts:
                for blk in part:
                    # in the first subset's order, so every subset's columns line up (some blocks, e.g. the
                    # s2-adm extras, list their columns in a different order per subset)
                    pos = {n: j for j, n in enumerate(blk.names)}
                    names = [n for n in self.feature_names if n in pos]
                    blk.columns = np.array([pos[n] for n in names], dtype=int)
                    blk.names = names
                if [n for blk in part for n in blk.names] != self.feature_names:
                    raise ValueError("Shared columns are in a different order across subsets")
        self.n_features = len(self.feature_names)
        self.field_names = np.array(field_names)
        self.start, self.stop = np.array(starts), np.array(stops)
        self.part = np.array(part_of_field)
        self.targets = targets

    def rows(self, fields, cap=None, rng=None):
        """(part, local_row, field) triples for the given global field ids, optionally at most `cap` per field."""
        out = []
        for f in fields:
            r = np.arange(self.start[f], self.stop[f])
            if cap is not None and len(r) > cap:
                r = np.sort(rng.choice(r, cap, replace=False))
            out.append(np.c_[np.full(len(r), self.part[f]), r, np.full(len(r), f)])
        return np.concatenate(out) if out else np.empty((0, 3), int)

    def load(self, rows):
        x = np.empty((len(rows), self.n_features), np.float32)
        y = np.empty(len(rows), np.float32)
        for p in np.unique(rows[:, 0]):
            sel = np.flatnonzero(rows[:, 0] == p)
            local = rows[sel, 1]
            x[sel] = np.concatenate([blk.take(local) for blk in self.parts[p]], axis=1)
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
    """The code version: a sweep's frozen code copy has a COMMIT file; a checkout asks git (marked if dirty)."""
    stamp = Path(directory) / "COMMIT"
    if stamp.exists():
        return stamp.read_text().strip()
    try:
        sha = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


@hydra.main(version_base="1.3", config_path="conf/yieldsat", config_name="config")
def main(cfg):
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".resolved-{os.getpid()}.yaml"  # parallel fold tasks share the directory
    tmp.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    tmp.rename(out / "resolved.yaml")
    scheme, smoke, seed = cfg.scheme, bool(cfg.experiment.smoke), int(cfg.experiment.seed)
    if scheme not in SCHEMES:
        raise SystemExit(f"scheme must be one of {SCHEMES}")
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
    cutoff = cfg.experiment.get("season_cutoff")
    fraction = float(cfg.experiment.get("train_fraction") or 1.0)
    if not 0 < fraction <= 1:
        raise SystemExit("experiment.train_fraction must be in (0, 1]")
    data = Data(cfg, subsets, cfg.input, align=scheme == "loco",
                season_cutoff=None if cutoff is None else int(cutoff))
    if set(data.field_names) != set(folds.index):
        raise ValueError("Fields in the feature cache and the fold file differ")
    gfold = folds.loc[data.field_names].to_numpy()
    attrs = fields.loc[data.field_names, ["country", "year", "farm", "region"]].reset_index(drop=True)

    head = cfg.model
    pipe = build_pipeline(head, data.n_features, seed)
    search = head.get("search")
    grid = OmegaConf.to_container(search.param_grid, resolve=True) if search and "param_grid" in search else {}
    internal_cv = bool(head.get("internal_cv", False))  # the estimator tunes itself on the inner folds
    cap = 16 if smoke else int(head.get("pixel_cap") or cfg.experiment.pixel_cap)
    test_cap = 200 if smoke else None
    batch = int(head.get("predict_batch") or cfg.experiment.predict_batch)
    if smoke and grid:
        grid = {k: (v[:2] if i == 0 else v[:1]) for i, (k, v) in enumerate(grid.items())}
    fold_ids = sorted(np.unique(gfold))
    if cfg.experiment.fold is not None:
        if int(cfg.experiment.fold) not in fold_ids:
            raise SystemExit(f"fold {cfg.experiment.fold} is not in {fold_file} (folds: {fold_ids})")
        fold_ids = [int(cfg.experiment.fold)]
    elif smoke:
        fold_ids = fold_ids[:2]

    common = {
        "dataset": "yieldsat", "label": label, "scheme": scheme, "input": cfg.input, "model": head.name,
        "split": "smoke" if smoke else ("final" if scheme == "loco" else "development"),
        "smoke": smoke, "seed": seed, "pixel_cap": cap, "folds_in_scheme": len(np.unique(gfold)),
        "season_cutoff": None if cutoff is None else int(cutoff), "train_fraction": fraction,
        "provenance": {"fold_file": str(fold_file), "fold_file_sha256": ys.sha256(fold_file),
                       "folds_manifest_sha256": ys.sha256(splits / "manifest.json"),
                       "caches": data.provenance, "n_features": int(data.n_features),
                       "git_commit": git_commit(Path(__file__).parent), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                       "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID")},
    }

    for k in fold_ids:
        t0 = time.time()
        rng = np.random.default_rng(seed + 1000 * int(k))
        train_f = np.flatnonzero(gfold != k)
        test_f = np.flatnonzero(gfold == k)
        if fraction < 1:  # label efficiency: a random subset of the training fields (own stream, so the pixel
            # sample of the kept fields does not depend on the fraction)
            n_keep = max(10, int(round(fraction * len(train_f))))
            train_f = np.sort(np.random.default_rng([seed, int(k), 7]).choice(train_f, n_keep, replace=False))
        tr = data.rows(train_f, cap, rng)
        x_tr, y_tr = data.load(tr)
        t_load = time.time() - t0
        best, inner_score, tuning_rows = {}, None, []
        if grid or internal_cv:
            n_countries = attrs.loc[train_f].country.nunique()
            cv = inner_splits(scheme, attrs, tr[:, 2], seed, n_countries)
        if internal_cv:
            model = clone(pipe).fit(x_tr, y_tr, model__cv=cv)
            est = model.named_steps["model"]
            best, inner_score = {"lambda": est.lambda_}, est.best_score_
            tuning_rows = [{"fold": int(k), "params": json.dumps({"lambda": r["lambda"]}), "inner_rmse": r["inner_rmse"]}
                           for r in est.cv_results_]
        elif grid:
            gs = GridSearchCV(pipe, {f"model__{p}": v for p, v in grid.items()}, cv=cv, refit=True, n_jobs=1,
                              scoring="neg_root_mean_squared_error", error_score="raise")
            gs.fit(x_tr, y_tr)
            model = gs.best_estimator_
            best = {p.removeprefix("model__"): v for p, v in gs.best_params_.items()}
            inner_score = float(-gs.best_score_)
            tuning_rows = [{"fold": int(k), "params": json.dumps(params, default=str), "inner_rmse": -mean}
                           for params, mean in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])]
        else:
            model = clone(pipe).fit(x_tr, y_tr)
        del x_tr
        t_fit = time.time() - t0 - t_load

        # Predict every pixel of the test fields, in batches (2.7: evaluation is never capped).
        te = data.rows(test_f, test_cap, rng)
        ys_, ps_ = [], []
        for a in range(0, len(te), batch):
            x_te, y_te = data.load(te[a:a + batch])
            ps_.append(np.asarray(model.predict(x_te), dtype=np.float32))
            ys_.append(y_te)
        y, p = np.concatenate(ys_), np.concatenate(ps_)
        row = {"fold": int(k), **res.score_levels(y, p, te[:, 2]), "train_pixels": int(len(tr)),
               "train_fields": int(len(train_f)), "best_params": best, "inner_rmse": inner_score,
               "seconds": round(time.time() - t0, 1), "seconds_load": round(t_load, 1),
               "seconds_fit": round(t_fit, 1), "seconds_predict": round(time.time() - t0 - t_load - t_fit, 1),
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)}
        if scheme == "loco":
            row["held_out_country"] = str(attrs.loc[test_f[0]].country)
        print(json.dumps(row, default=str), flush=True)
        res.write_fold(out, common, row, data.field_names, te[:, 2], y, p, tuning_rows)

    if cfg.experiment.fold is not None:
        print(f"Outputs: {out}", flush=True)
        return
    result = res.combine(out)
    print(json.dumps({"summary": result["summary"]}, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)

    if scheme == "loco" and not smoke:
        ledger = Path(cfg.paths.runs) / "loco-ledger.jsonl"
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(), "crop": cfg.crop,
                 "input": cfg.input, "model": head.name, "seed": seed, "run": str(out),
                 "per_country": {r["held_out_country"]: {c: r[c] for c in res.METRICS} for r in result["per_fold"]}}
        try:
            with ledger.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            ledger.chmod(0o664)
        except OSError as error:
            print(f"WARNING: could not write the LOCO ledger: {error}", flush=True)


class CappedPCA(PCA):
    """PCA whose component count is capped at what the training data allows (small label fractions can have
    fewer training pixels than max_features)."""

    def _fit(self, X):
        self.n_components = min(int(self.n_components), X.shape[0], X.shape[1])
        return super()._fit(X)


def build_pipeline(head, n_features, seed):
    """impute (unless the head handles NaN) -> optional scale + PCA when the head caps its width -> estimator."""
    steps = []
    max_features = head.get("max_features")
    reduce = max_features is not None and n_features > int(max_features)
    if head.get("impute", True) or reduce:
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    if reduce:
        steps += [("scale", StandardScaler()), ("pca", CappedPCA(int(max_features), random_state=seed))]
    steps.append(("model", instantiate(head.estimator)))
    return Pipeline(steps)


if __name__ == "__main__":
    main()
