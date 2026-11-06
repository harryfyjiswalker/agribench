import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.linear import LogisticRegression
from geo_eval_harness.main import select_logistic_c
from geo_eval_harness.results import bootstrap_accuracy
from geo_eval_harness.utils import FeatureSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import jaccard_score, precision_recall_fscore_support

parser = argparse.ArgumentParser(description="Evaluate frozen PASTIS-R parcel features")
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("--recipe", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
args = parser.parse_args()
if args.out.exists():
    parser.error(f"Output directory already exists: {args.out}")

counts = {1: 17229, 2: 17559, 3: 17045, 4: 17478, 5: 16640}
classes = np.arange(1, 19)
meta = pd.read_csv(
    args.metadata, usecols=["ID_PARCEL", "ID_PATCH", "Label", "Fold"],
    dtype="Int64",
).sort_values("ID_PARCEL")
if (
    meta.isna().any().any()
    or meta.ID_PARCEL.duplicated().any()
    or meta.Fold.value_counts().to_dict() != counts
    or not meta.Label.isin(classes).all()
    or (meta.groupby("ID_PATCH").Fold.nunique() > 1).any()
):
    raise ValueError("Unexpected PASTIS-R parcel metadata")

ids = meta.ID_PARCEL.to_numpy(dtype=np.int64)
y = meta.Label.to_numpy(dtype=np.int64)
fold = meta.Fold.to_numpy(dtype=np.int64)
recipe = json.loads(args.recipe.read_text())

with np.load(args.features, allow_pickle=False) as table:
    feature_ids = table["ids"]
    x = table["features"]
if feature_ids.ndim != 1 or feature_ids.dtype.kind not in "iu":
    raise ValueError("Expected a one-dimensional array of integer parcel IDs")
order = np.argsort(feature_ids)
if not np.array_equal(feature_ids[order], ids):
    raise ValueError("Expected exactly one feature row for each of the 85,951 parcels")
if x.ndim != 2 or x.shape[0] != len(ids) or x.shape[1] == 0 or x.dtype.kind != "f":
    raise ValueError("Expected a floating-point feature matrix with one row per parcel")
x = x[order].astype(np.float32, copy=False)
if np.isinf(x).any() or np.isnan(x).all(axis=1).any():
    raise ValueError("Features contain infinities or parcels with no observed features")


def prepare(fit, evaluate):
    if not (np.isnan(fit).any() or np.isnan(evaluate).any()):
        return fit, evaluate
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(fit), imputer.transform(evaluate)


train, val, test = fold <= 3, fold == 4, fold == 5
cfg = RunConfig(
    model={"name": "parcel_features"},
    datasets=["pastis-r-pixelset"],
    runtime={"device": args.device, "seed": 0},
)
linear = cfg.classification.linear
grid = 10 ** np.linspace(linear.c_log10_start, linear.c_log10_stop, linear.c_count)
x_train, x_val = prepare(x[train], x[val])
selected_c = select_logistic_c(
    FeatureSplit(torch.from_numpy(x_train), torch.from_numpy(y[train])),
    FeatureSplit(x_val, y[val]),
    grid.tolist(),
    cfg,
)
del x_train, x_val

fit = train | val
x_fit, x_test = prepare(x[fit], x[test])
probe = LogisticRegression(
    C=selected_c, max_iter=4000, tol=1e-6, random_state=0, device=args.device,
)
probe.fit(torch.from_numpy(x_fit), torch.from_numpy(y[fit]))
pred = probe.predict(torch.from_numpy(x_test))

accuracy, low, high = bootstrap_accuracy(y[test], pred, n_boot=200, seed=0)
precision, recall, f1, support = precision_recall_fscore_support(
    y[test], pred, labels=classes, zero_division=0,
)
iou = jaccard_score(y[test], pred, labels=classes, average=None, zero_division=0)

args.out.mkdir(parents=True)
pd.DataFrame({"parcel_id": ids[test], "label": y[test], "prediction": pred}).to_csv(
    args.out / "test-predictions.csv", index=False,
)
pd.DataFrame({
    "label": classes, "support": support, "precision": precision,
    "recall": recall, "f1": f1, "iou": iou,
}).to_csv(args.out / "per-class.csv", index=False)
metrics = {
    "protocol": "pastis-r-parcel-linear-probe-v1",
    "representation": recipe,
    "selected_c": float(selected_c),
    "test_parcels": int(test.sum()),
    "accuracy": accuracy,
    "accuracy_ci95": [low, high],
    "macro_f1": float(f1.mean()),
    "macro_iou": float(iou.mean()),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
