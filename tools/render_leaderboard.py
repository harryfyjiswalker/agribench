"""Render all exported validation runs, grouped by comparison protocol."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(results):
    groups = defaultdict(list)
    for row in results:
        key = (
            row["dataset"], tuple(row["sensors"]), tuple(row["train_folds"]),
            tuple(row["validation_folds"]), row["train_parcels"],
            row["validation_ids_sha256"], row["training_ids_sha256"],
            tuple(row["feature_sha256"]),
        )
        groups[key].append(row)
    lines = [
        "# PASTIS-R parcel leaderboard", "",
        "Validation results. Fold 5 is reserved for test. Smoke checks are excluded.",
        "Rows are grouped by sensors, split, training budget and available data fingerprints.",
        "Sequence runs lack an exact training ID fingerprint in the current runner.", "",
    ]
    if not groups:
        lines.append("No validation results published yet.")
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        dataset, sensors, train, val, n_train, val_ids, train_ids, tables = key
        comparison = " / ".join(filter(None, [val_ids[:8], (train_ids or "")[:8],
                                                *(x[:8] for x in tables)]))
        lines.extend([
            f"## {dataset} · {' + '.join(sensors)} · {n_train:,} training parcels", "",
            f"Train folds {list(train)}, validation folds {list(val)}. Fingerprints `{comparison}`.", "",
            "| Run | Model | Seed | Epoch | Validation parcels | mIoU | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(rows, key=lambda r: (-r["macro_iou"], r["run"])):
            lines.append(
                f'| {row["run"]} | {row["model"]} | {row["seed"]} | '
                f'{row["selected_epoch"] or "—"} | {row["validation_parcels"]:,} | '
                f'{row["macro_iou"]:.2%} | {row["accuracy"]:.2%} |'
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    data = [json.loads(path.read_text()) for path in sorted((ROOT / "results").glob("*.json"))]
    (ROOT / "LEADERBOARD.md").write_text(render(data))
