"""PLANTEUR runner: one model x input x split fold, scored as protocols/planteur.md sets out.

    sbatch planteur.sbatch split=default input=emb-aef model=probe experiment.confirm_test=true
    sbatch planteur.sbatch split=loio fold=reunion input=emb-tessera-v2 model=lightgbm experiment.confirm_test=true
    sbatch planteur.sbatch split=default input=emb-tessera-v2 model=probe experiment.fewshot=100   # label efficiency
    sbatch planteur.sbatch split=default input=s2-monthly model=probe experiment.months=4          # early season
    add experiment.smoke=true for a quick plumbing check (scores are meaningless)

Inputs are the per-pixel tables of features/planteur/ (tools/build_planteur_features.py) or the per-patch token grids
and AgriBench-built tables of tools/planteur_embed.py. For one fold:
  1. choose the training pixels by the head's regime (3.6): every pixel (probe), at most 256 per patch (LightGBM), or
     the few-shot set (TabPFN, TabICL; experiment.fewshot for any head). Pixels the input has no value for are left
     out of training (D7). Every statistic is fitted on these training pixels only (D3);
  2. choose the head's setting (probe learning rate, LightGBM grid) by mIoU-8 on all val pixels;
  3. with experiment.confirm_test=true, predict every test pixel (pixels without input value -> Background, D7), build
     one confusion matrix, score mIoU-8 / OA / per-class IoU (3.4) and append the run to the test ledger (3.7).
Writes fold-<fold>.json (metrics, choices, provenance), fold-<fold>-confusion.npy and fold-<fold>-predictions.npz
(cluster only, never committed). tools/collect_planteur.py combines the five LOIO islands.
"""
import getpass
import itertools
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tools import planteur as pl

FOLD_KEYS = {"default": "default", "reunion": "Réunion", "guadeloupe": "Guadeloupe", "martinique": "Martinique",
             "guyane": "Guyane", "mayotte": "Mayotte"}


class Features:
    """One input's pixel table, optionally cut to the first k months (early season)."""

    def __init__(self, root, name, meta, months=None):
        d = Path(root) / name
        if not (d / "info.json").exists():
            raise SystemExit(f"No cache for input '{name}' ({d}); build it with tools.build_planteur_features")
        self.info = json.loads((d / "info.json").read_text())
        if self.info["meta_sha256"] != meta.sha256:
            raise ValueError(f"{d} was built for different meta.npz rows")
        self.kind = self.info["kind"]
        if self.kind == "grid":  # per-patch token grids (tools/planteur_embed.py), sampled per pixel in take()
            self.grid = np.load(d / "grid.npy")
            self.meta = meta
            self.valid = np.repeat(np.load(d / "patch_valid.npy"), np.diff(meta.offsets))
        else:
            self.x = np.load(d / "X.npy", mmap_mode="r")
            self.valid = np.load(d / "valid.npy")
        names = self.info["feature_names"]
        self.columns = None
        if months is not None:
            if self.info["kind"] != "s2":
                raise SystemExit("experiment.months needs a monthly input (s2-monthly); annual embeddings can't be cut")
            if not 1 <= months <= pl.N_MONTHS:
                raise SystemExit(f"experiment.months must be 1-{pl.N_MONTHS}")
            keep = [i for i, n in enumerate(names) if int(n[1:3]) < months]
            self.columns = np.array(keep)
            names = [names[i] for i in keep]
        self.names = names
        self.nodata = self.info.get("nodata")
        self.provenance = {"input": name, "info_sha256": pl.sha256(d / "info.json"), "rows": self.info["rows"]}

    def take(self, rows):
        """float32 features of the given (sorted) rows; no-data -> NaN. Contiguous runs are copied as slices."""
        rows = np.asarray(rows)
        out = np.empty((len(rows), len(self.names)), np.float32)
        if len(rows) == 0:
            return out
        if self.kind == "grid":
            return self._take_grid(rows, out)
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        starts = np.r_[0, breaks]
        stops = np.r_[breaks, len(rows)]
        if len(starts) > len(rows) // 64:  # scattered rows: one fancy-index read in chunks
            for a in range(0, len(rows), 1 << 20):
                c = self.x[rows[a:a + (1 << 20)]]
                out[a:a + len(c)] = c if self.columns is None else c[:, self.columns]
        else:
            for s, e in zip(starts, stops):
                c = self.x[rows[s]:rows[e - 1] + 1]
                out[s:e] = c if self.columns is None else c[:, self.columns]
        if self.nodata is not None:
            raw = out == self.nodata
            out[raw] = np.nan
        return out

    def _take_grid(self, rows, out, chunk=1 << 18):
        """Bilinear sample of each row's patch grid at the pixel centre, border-clamped: the rule of
        tools.yieldsat_patches.sample_tokens (grid_sample, align_corners=False, padding 'border')."""
        g, o, span = self.info["grid"], self.info["origin"], self.info["span"]
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            k = np.searchsorted(self.meta.offsets, r, side="right") - 1
            pix = self.meta.pix[r].astype(np.int64)
            u = np.clip((pix // pl.SIZE + 0.5 - o) / span * g - 0.5, 0, g - 1)
            v = np.clip((pix % pl.SIZE + 0.5 - o) / span * g - 0.5, 0, g - 1)
            i0, j0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            i1, j1 = np.minimum(i0 + 1, g - 1), np.minimum(j0 + 1, g - 1)
            wu, wv = (u - i0).astype(np.float32)[:, None], (v - j0).astype(np.float32)[:, None]
            G = self.grid
            out[a:a + len(r)] = ((G[k, i0, j0] * (1 - wv) + G[k, i0, j1] * wv) * (1 - wu) +
                                 (G[k, i1, j0] * (1 - wv) + G[k, i1, j1] * wv) * wu)
        return out


class Rows:
    """A lazy (rows, features) view: slicing reads only that slice. Lets the probe standardise and stage its table
    chunk by chunk instead of holding a float32 copy of every training pixel."""

    def __init__(self, feats, rows):
        self.feats, self.rows = feats, np.asarray(rows)
        self.shape = (len(self.rows), len(feats.names))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, s):
        return self.feats.take(self.rows[s])


def training_rows(regime, lists, meta, feats, cfg, rng, smoke):
    """(rows, description) of the training pixels for a regime (3.6). Rows without input value are dropped (D7)."""
    tr = meta.rows(lists["train"])
    fewshot = cfg.experiment.fewshot
    if regime == "fewshot":
        x = int(fewshot)
        if cfg.split == "default":
            rows = meta.rows_of_pixels(pl.fewshot_pixels(x))
            src = f"official splits/fewshot_pixels.json X={x}"
        else:
            # LOIO: the official set includes the held-out islands' patches, so AgriBench draws X pixels per class
            # from this fold's own training pixels (seed 0), the same way.
            rows = []
            ok = tr[feats.valid[tr]]
            for c in range(pl.N_CLASSES):
                pool = ok[meta.y[ok] == c]
                rows.append(rng.choice(pool, min(x, len(pool)), replace=False))
            rows = np.sort(np.concatenate(rows))
            src = f"AgriBench draw from the fold's train pixels, X={x}, seed {cfg.experiment.seed}"
        n_all = len(rows)
        rows = rows[feats.valid[rows]]
        return rows, {"regime": f"fewshot-{x}", "source": src, "dropped_no_input": int(n_all - len(rows))}
    rows = tr[feats.valid[tr]]
    dropped = int(len(tr) - len(rows))
    if regime == "cap" or smoke:
        cap = 16 if smoke else int(cfg.experiment.pixel_cap)
        patch = meta.patch_of_rows(rows)
        b = np.r_[0, np.flatnonzero(np.diff(patch)) + 1, len(rows)]
        keep = [rows[s:e] if e - s <= cap else np.sort(rng.choice(rows[s:e], cap, replace=False))
                for s, e in zip(b[:-1], b[1:])]
        rows = np.concatenate(keep)
        return rows, {"regime": f"cap-{cap}", "dropped_no_input": dropped}
    return rows, {"regime": "all", "dropped_no_input": dropped}


class SklearnHead:
    """impute (if the head can't take NaN) -> optional scale + PCA to max_features -> estimator."""

    def __init__(self, head, n_features, seed, params=None):
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = []
        mf = head.get("max_features")
        reduce = mf is not None and n_features > int(mf)
        if head.get("impute", True) or reduce:
            steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
        if reduce:
            steps += [("scale", StandardScaler()), ("pca", PCA(int(mf), random_state=seed))]
        est = instantiate(head.estimator)
        if params:
            est.set_params(**params)
        steps.append(("model", est))
        self.pipe = Pipeline(steps)

    def fit(self, x, y):
        if "pca" in self.pipe.named_steps:  # PCA can't keep more components than rows (only tiny smoke sets)
            pca = self.pipe.named_steps["pca"]
            pca.set_params(n_components=min(pca.n_components, len(x)))
        self.pipe.fit(x, y)
        return self

    def predict(self, x):
        p = np.asarray(self.pipe.predict(x))
        return p.astype(np.uint8)


def predict_rows(predict_fn, feats, rows, batch):
    """Predicted class for every row; rows without input value are Background (D7)."""
    out = np.zeros(len(rows), np.uint8)
    if predict_fn is None:  # the all-Background reference row
        return out
    ok = feats.valid[rows]
    idx = np.flatnonzero(ok)
    for a in range(0, len(idx), batch):
        sel = idx[a:a + batch]
        out[sel] = predict_fn(feats.take(rows[sel]))
    return out


def evaluate(meta, rows, p):
    cm = pl.confusion(meta.y[rows], p)
    return cm, pl.scores(cm)


def git_commit(directory):
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


@hydra.main(version_base="1.3", config_path="conf/planteur", config_name="config")
def main(cfg):
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".resolved-{os.getpid()}.yaml"
    tmp.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    tmp.rename(out / "resolved.yaml")
    t0 = time.time()
    smoke, seed, head = bool(cfg.experiment.smoke), int(cfg.experiment.seed), cfg.model
    if cfg.fold not in FOLD_KEYS:
        raise SystemExit(f"fold must be one of {list(FOLD_KEYS)}")
    if (cfg.split == "default") != (cfg.fold == "default"):
        raise SystemExit("split=default goes with fold=default; split=loio needs an island fold")
    lists, split_file = pl.fold_lists(cfg.split, FOLD_KEYS[cfg.fold])
    if smoke:
        lists = {k: v[:{"train": 40}.get(k, 10)] for k, v in lists.items()}
    confirm = bool(cfg.experiment.confirm_test)

    meta = pl.Meta(cfg.paths.features)
    months = cfg.experiment.months
    feats = Features(cfg.paths.features, cfg.input, meta, None if months is None else int(months))
    rng = np.random.default_rng(seed)
    regime = head.regime
    if cfg.experiment.fewshot is not None and regime != "none":
        regime = "fewshot"
    elif regime == "fewshot":
        cfg.experiment.fewshot = int(head.fewshot)
    if regime == "fewshot" and smoke:
        cfg.experiment.fewshot = 10
    tr_rows, regime_info = (np.empty(0, np.int64), {"regime": "none"}) if regime == "none" else \
        training_rows(regime, lists, meta, feats, cfg, rng, smoke)
    va_rows, te_rows = meta.rows(lists["val"]), meta.rows(lists["test"])
    batch = int(head.get("predict_batch") or cfg.experiment.predict_batch)
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"{cfg.split}/{cfg.fold} {cfg.input} {head.name}: {len(tr_rows):,} training pixels ({regime_info['regime']}), "
        f"{len(feats.names)} features, {len(va_rows):,} val / {len(te_rows):,} test pixels")

    if head.kind == "probe":  # lazy: the probe reads its training table in chunks
        x_tr = Rows(feats, tr_rows)
    else:
        x_tr = feats.take(tr_rows) if len(tr_rows) else None
    y_tr = meta.y[tr_rows]
    t_load = time.time() - t0
    tuning, best = [], {}
    if head.kind == "constant":
        predict_fn = None
    elif head.kind == "probe":
        from tools import planteur_probe as probe
        lrs = [float(v) for v in np.geomspace(*head.lr_geomspace[:2], int(head.lr_geomspace[2]))]
        heads, scaler, info = probe.train(x_tr, y_tr, pl.N_CLASSES, lrs, epochs=2 if smoke else int(head.epochs),
                                          batch=int(head.batch), seed=seed, weight_decay=float(head.weight_decay),
                                          log=log)
        del x_tr
        import torch
        device = torch.device(info["device"])
        pv = np.zeros((len(lrs), len(va_rows)), np.uint8)
        ok = np.flatnonzero(feats.valid[va_rows])
        for a in range(0, len(ok), batch):
            sel = ok[a:a + batch]
            pv[:, sel] = probe.predict(heads, feats.take(va_rows[sel]), scaler, device)
        for lr, p, loss in zip(lrs, pv, info["final_loss"]):
            s = pl.scores(pl.confusion(meta.y[va_rows], p))
            tuning.append({"params": {"lr": lr}, "val_miou8": s["miou8"], "val_oa": s["oa"], "final_train_loss": loss})
        k = int(np.argmax([t["val_miou8"] for t in tuning]))
        best = {"lr": lrs[k]}
        if k in (0, len(lrs) - 1):
            log(f"WARNING: best learning rate {lrs[k]:g} is on the edge of the grid")
        chosen = [heads[k]]
        predict_fn = lambda x: probe.predict(chosen, x, scaler, device)[0]  # noqa: E731
        best_info = {"probe": {k2: v for k2, v in info.items() if k2 != "final_loss"}}
    elif head.kind == "sklearn":
        search = head.get("search")
        grid = OmegaConf.to_container(search.param_grid, resolve=True) if search else {}
        if smoke and grid:
            grid = {k: v[:1] for k, v in grid.items()}
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())] or [{}]
        fitted = []
        for params in combos:
            m = SklearnHead(head, len(feats.names), seed, params).fit(x_tr, y_tr)
            if len(combos) > 1:
                p = predict_rows(m.predict, feats, va_rows, batch)
                s = pl.scores(pl.confusion(meta.y[va_rows], p))
                tuning.append({"params": params, "val_miou8": s["miou8"], "val_oa": s["oa"]})
                log(f"  {params}: val mIoU-8 {s['miou8']:.2f}")
            fitted.append(m)
        k = int(np.argmax([t["val_miou8"] for t in tuning])) if tuning else 0
        best, model = combos[k], fitted[k]
        del fitted, x_tr
        predict_fn = model.predict
    else:
        raise SystemExit(f"Unknown head kind {head.kind}")
    t_fit = time.time() - t0 - t_load

    p_va = predict_rows(predict_fn, feats, va_rows, batch)
    cm_va, s_va = evaluate(meta, va_rows, p_va)
    row = {"fold": cfg.fold, "fold_name": FOLD_KEYS[cfg.fold], "val": s_va, "best_params": best, "tuning": tuning,
           "train_pixels": int(len(tr_rows)), **regime_info,
           "train_class_pixels": np.bincount(y_tr, minlength=pl.N_CLASSES).tolist(),
           "val_patches": len(lists["val"]), "val_pixels": int(len(va_rows)),
           "test_patches": len(lists["test"]), "test_pixels": int(len(te_rows)),
           "test_pixels_no_input": int((~feats.valid[te_rows]).sum())}
    if head.kind == "probe":
        row.update(best_info)
    if confirm:
        p_te = predict_rows(predict_fn, feats, te_rows, batch)
        cm_te, s_te = evaluate(meta, te_rows, p_te)
        row["test"] = s_te
        np.save(out / f"fold-{cfg.fold}-confusion.npy", cm_te)
        np.savez_compressed(out / f"fold-{cfg.fold}-predictions.npz", rows=te_rows.astype(np.int64), p=p_te)
    row.update(seconds=round(time.time() - t0, 1), seconds_load=round(t_load, 1), seconds_fit=round(t_fit, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2))
    common = {"dataset": "planteur", "split": cfg.split, "input": cfg.input, "model": head.name,
              "stage": "smoke" if smoke else ("test" if confirm else "development"), "smoke": smoke, "seed": seed,
              "regime": regime_info["regime"], "months": None if months is None else int(months),
              "provenance": {"split_file": split_file, "split_file_sha256": pl.manifest_sha(split_file),
                             "fewshot_file_sha256": pl.manifest_sha("splits/fewshot_pixels.json")
                             if regime == "fewshot" else None,
                             "meta_sha256": meta.sha256, "input": feats.provenance, "n_features": len(feats.names),
                             "config_sha256": pl.sha256(out / "resolved.yaml"),
                             "git_commit": git_commit(Path(__file__).parent),
                             "slurm_job": os.environ.get("SLURM_JOB_ID"),
                             "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID")}}
    tmpf = out / f".fold-{cfg.fold}.json.tmp"
    tmpf.write_text(json.dumps({**common, "fold_result": row}, indent=1, default=str) + "\n")
    tmpf.rename(out / f"fold-{cfg.fold}.json")  # written last: its presence means the fold is complete
    summary = {"val_miou8": round(s_va["miou8"], 2), "val_oa": round(s_va["oa"], 2), "best": best}
    if confirm:
        summary.update(test_miou8=round(row["test"]["miou8"], 2), test_oa=round(row["test"]["oa"], 2))
    log(json.dumps(summary))
    log(f"Outputs: {out}")

    if confirm and not smoke:
        ledger = Path(cfg.paths.runs) / "test-ledger.jsonl"
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(), "split": cfg.split,
                 "fold": cfg.fold, "input": cfg.input, "model": head.name, "regime": regime_info["regime"],
                 "months": common["months"], "seed": seed, "run": str(out), "best_params": best,
                 "test_miou8": row["test"]["miou8"], "test_oa": row["test"]["oa"],
                 "git_commit": common["provenance"]["git_commit"]}
        try:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            ledger.chmod(0o664)
        except OSError as error:
            log(f"WARNING: could not write the test ledger: {error}")


if __name__ == "__main__":
    main()
