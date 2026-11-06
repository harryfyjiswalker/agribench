"""Deterministic per-sensor features and a common tabular classifier path."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

CHANNELS = {"S2": 10, "S1A": 3, "S1D": 3}
START, STOP, WIDTH = -200, 600, 30


def digest(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def summarize(values, dates, channels):
    dates = np.asarray(dates)
    if values.ndim != 3 or values.shape[:2] != (len(dates), channels):
        raise ValueError(f"Expected [acquisition, channel, pixel]; got {values.shape}")
    if np.any((dates < START) | (dates >= STOP)):
        raise ValueError("Dates outside the configured feature window")
    count = (STOP - START + WIDTH - 1) // WIDTH
    out = np.full((count, channels), np.nan, dtype=np.float32)
    if values.shape[-1]:
        per_date = np.median(values, axis=-1)
        bins = (dates - START) // WIDTH
        for index in np.unique(bins):
            out[index] = np.median(per_date[bins == index], axis=0)
    out[~np.isfinite(out)] = np.nan
    return out.ravel()


def feature_table(root, sensor, folds, ids, cache, loader):
    import inspect

    provenance = {
        "recipe": "pixel_median_then_30day_median_v1",
        "reference_date": "2018-09-01", "bins": [START, STOP, WIDTH],
        "sensor": sensor, "folds": sorted(folds), "root": str(root),
        "metadata": {name: digest(root / name) for name in
                     ("metadata_parcel.csv", "metadata.geojson")},
        "loader_sha256": digest(inspect.getfile(loader)),
        "builder_sha256": digest(__file__),
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{sensor}-{key[:20]}.npz"
    if not path.exists():
        ds = loader(folder=str(root), sats=[sensor], folds=sorted(folds),
                    reference_date=provenance["reference_date"], norm=False,
                    cache=False, geom_features=False)
        if not np.array_equal(ds.id_parcels, ids):
            raise ValueError("Loader parcel IDs differ from metadata")
        rows = []
        dates_by_patch = {}
        for i, parcel in enumerate(ids):
            patch = ds.id_patches[parcel]
            if patch not in dates_by_patch:
                dates_by_patch[patch] = ds.get_dates(patch, sensor)
            values = np.load(root / f"DATA_{sensor}" / f"{sensor}_{parcel}.npy",
                             allow_pickle=False)
            rows.append(summarize(values, dates_by_patch[patch], CHANNELS[sensor]))
            if (i + 1) % 10000 == 0:
                print(f"{sensor}: {i + 1}/{len(ids)} parcels", flush=True)
        # Write beside the destination so readers see only a complete cache.
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as tmp:
            staged = Path(tmp.name)
            try:
                np.savez(tmp, ids=ids, features=np.stack(rows))
                tmp.flush()
                os.chmod(staged, 0o644)
                os.replace(staged, path)
            finally:
                staged.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as table:
        if not np.array_equal(table["ids"], ids):
            raise ValueError(f"Parcel IDs mismatch in {path}")
        features = table["features"]
    return features, {"file": str(path), "sha256": digest(path), **provenance}


def embedding_table(path, ids, representation):
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        with np.load(file, allow_pickle=False) as table:
            parcel_ids = table["ids"]
            features = table["features"]
    if parcel_ids.ndim != 1 or parcel_ids.dtype.kind not in "iu":
        raise ValueError("Embedding IDs must be a one-dimensional integer array")
    if (features.ndim != 2 or features.shape[0] != len(parcel_ids)
            or not features.shape[1] or features.dtype.kind != "f"
            or not np.isfinite(features).all()):
        raise ValueError("Expected a finite floating-point embedding vector per parcel")
    order = np.argsort(parcel_ids)
    sorted_ids = parcel_ids[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError("Embedding IDs must be unique")
    pos = np.clip(np.searchsorted(sorted_ids, ids), 0, len(sorted_ids) - 1)
    if not np.array_equal(sorted_ids[pos], ids):
        raise ValueError("Embeddings are missing train or validation parcels")
    return features[order][pos], {"file": str(path), "sha256": sha256,
                                  "representation": representation,
                                  "rows_in_file": int(len(parcel_ids))}


def run(cfg, folds, out, loader, score_predictions):
    root = Path(cfg.dataset.root).resolve()
    sensors = list(cfg.inputs.sensors)
    if not sensors or len(set(sensors)) != len(sensors) or set(sensors) - CHANNELS.keys():
        raise ValueError("Choose distinct sensors from S2, S1A, S1D")
    if cfg.experiment.max_batches is not None:
        raise ValueError("Tabular runs use experiment.train_limit, not max_batches")
    checkpoint = cfg.model.estimator.get("model_path")
    if checkpoint:
        (out / "checkpoint-sha256.txt").write_text(digest(Path(checkpoint)) + "\n")
    estimator = instantiate(cfg.model.estimator)
    metadata = pd.read_csv(root / "metadata_parcel.csv")
    meta = metadata[metadata.Fold.isin(folds[0] + folds[1])].sort_values("ID_PARCEL")
    ids = meta.ID_PARCEL.to_numpy(dtype=np.int64)
    y = meta.Label.to_numpy(dtype=np.int64) - 1
    if len(np.unique(ids)) != len(ids) or np.any((y < 0) | (y >= cfg.dataset.num_classes)):
        raise ValueError("Invalid parcel IDs or labels")
    train = np.flatnonzero(meta.Fold.isin(folds[0]).to_numpy())
    val = np.flatnonzero(meta.Fold.isin(folds[1]).to_numpy())
    limit = cfg.experiment.train_limit
    if limit is None:
        limit = cfg.validation.get("train_limit")
    if limit is not None and limit < len(train):
        train, _ = train_test_split(train, train_size=limit,
                                    random_state=cfg.experiment.seed, stratify=y[train])
        train.sort()
    if not len(val) or len(np.unique(y[train])) != cfg.dataset.num_classes:
        raise ValueError("Empty validation split or training subset missing crop classes")
    if "embedding_path" in cfg.inputs:
        tables = [embedding_table(Path(cfg.inputs.embedding_path), ids, cfg.inputs.representation)]
    else:
        tables = [feature_table(root, sensor, folds[0] + folds[1], ids,
                                Path(cfg.cache_dir), loader) for sensor in sensors]
    x = np.concatenate([features for features, _ in tables], axis=1)
    (out / "feature-tables.json").write_text(json.dumps([p for _, p in tables], indent=2))
    pd.DataFrame({"parcel_id": ids[train]}).to_csv(out / "training-ids.csv", index=False)
    # Both classifiers receive the same fitted, training-only imputation.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x[train])
    x_val = imputer.transform(x[val])
    estimator.fit(x_train, y[train])
    pred = estimator.predict(x_val)
    record = score_predictions(y[val], pred)
    record.update(train_parcels=len(train), validation_parcels=len(val))
    pd.DataFrame({"parcel_id": ids[val], "label_index": y[val],
                  "prediction_index": pred}).to_csv(out / "validation-predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps([record], indent=2))
    print(record, flush=True)
