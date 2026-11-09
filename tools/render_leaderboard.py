"""Render the PASTIS-R pilot leaderboard into LEADERBOARD.md (the README no longer holds per-dataset tables)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(results):
    if not results:
        return "No validation results published yet."

    lines = [
        "",
        "| Run | Model | Representation | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda row: (-row["macro_iou"], row["run"])):
        representation = row.get("representation")
        if representation == "galileo_s2":
            representation = "Galileo Base embeddings"
        elif not representation:
            if row["model"] == "adapted_pse_ltae2d":
                representation = "Parcel pixel sets"
            else:
                representation = "Parcel summaries" if row.get("feature_sha256") else "—"
        sensors = " + ".join(row["sensors"])
        train = ", ".join(map(str, row["train_folds"]))
        val = ", ".join(map(str, row["validation_folds"]))
        epoch = row["selected_epoch"] or "—"
        macro_f1 = row.get("macro_f1")
        f1_display = f"{macro_f1:.3f}" if macro_f1 is not None else "—"
        lines.append(
            f'| [`{row["run"]}`](results/{row["run"]}.json) | {row["model"]} | '
            f'{representation} | {sensors} | {train} → {val} | {row["train_parcels"]:,} | '
            f'{row["validation_parcels"]:,} | {row["seed"]} | {epoch} | '
            f'{row["macro_iou"]:.3f} | {row["accuracy"]:.3f} | {f1_display} |'
        )
    lines.extend([
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    data = [json.loads(path.read_text()) for path in sorted((ROOT / "results").glob("*.json"))]
    table = render(data)
    (ROOT / "LEADERBOARD.md").write_text(f"# PASTIS-R parcel leaderboard\n\n{table}\n")
