"""Export a validation summary without paths, parcel IDs or checkpoints."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import yaml
from sklearn.metrics import f1_score


def ids_hash(ids):
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate parcel IDs")
    return hashlib.sha256("\n".join(map(str, sorted(ids))).encode()).hexdigest()


def csv_ids(path):
    with path.open(newline="") as file:
        return [int(row["parcel_id"]) for row in csv.DictReader(file)]


def export(run, label):
    cfg = yaml.safe_load((run / "resolved.yaml").read_text())
    if cfg["experiment"].get("max_batches") is not None:
        raise ValueError("Smoke checks do not belong on the leaderboard")
    metrics = json.loads((run / "metrics.json").read_text())
    # The training loop saves predictions from the earliest best validation epoch.
    best = max(metrics, key=lambda row: row["macro_iou"])
    with (run / "validation-predictions.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    predictions = [int(row["parcel_id"]) for row in rows]
    truth = [int(row["label_index"]) for row in rows]
    predicted = [int(row["prediction_index"]) for row in rows]
    classes = range(cfg["dataset"]["num_classes"])
    if (set(truth) | set(predicted)) - set(classes):
        raise ValueError("Prediction file contains an invalid class index")
    macro_f1 = f1_score(truth, predicted, labels=list(classes), average="macro", zero_division=0)
    if len(predictions) != best["validation_parcels"]:
        raise ValueError("Prediction count differs from saved metrics")
    train_path = run / "training-ids.csv"
    train = csv_ids(train_path) if train_path.exists() else None
    if train is not None and (len(train) != best["train_parcels"] or set(train) & set(predictions)):
        raise ValueError("Training IDs disagree with the recorded split")
    features = run / "feature-tables.json"
    tables = json.loads(features.read_text()) if features.exists() else []
    checkpoint = run / "checkpoint-sha256.txt"
    return {
        "run": label,
        "dataset": "PASTIS-R PixelSet (Zenodo 5745151)",
        "model": cfg["model"]["name"],
        "seed": cfg["experiment"]["seed"],
        "sensors": cfg.get("inputs", {}).get("sensors", [cfg["dataset"]["sensor"]]),
        "train_folds": cfg["validation"]["train_folds"],
        "validation_folds": cfg["validation"]["val_folds"],
        "train_parcels": best["train_parcels"],
        "validation_parcels": len(predictions),
        "validation_ids_sha256": ids_hash(predictions),
        "training_ids_sha256": ids_hash(train) if train is not None else None,
        "feature_sha256": [table["sha256"] for table in tables],
        **({"representation": tables[0]["representation"]}
           if tables and "representation" in tables[0] else {}),
        "checkpoint_sha256": checkpoint.read_text().strip() if checkpoint.exists() else None,
        "selected_epoch": best.get("epoch"),
        "accuracy": best["accuracy"],
        "macro_iou": best["macro_iou"],
        "macro_f1": float(macro_f1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--backfill-macro-f1", action="store_true",
                        help="Add macro F1 to an existing summary of this same run")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("Name must contain only lowercase letters, digits, underscores or hyphens")
    result = export(args.run_dir, args.name)
    dest = Path(__file__).resolve().parents[1] / "results" / f"{args.name}.json"
    if args.backfill_macro_f1:
        existing = json.loads(dest.read_text())
        original = {key: value for key, value in result.items() if key != "macro_f1"}
        if existing == result:
            print(dest)
            raise SystemExit(0)
        if existing != original:
            parser.error("Saved run does not match the existing result summary")
    with dest.open("w" if args.backfill_macro_f1 else "x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(dest)
