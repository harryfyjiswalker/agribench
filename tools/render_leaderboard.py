"""Render the validation leaderboard in both Markdown pages."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- leaderboard:start -->"
END = "<!-- leaderboard:end -->"


def render(results):
    if not results:
        return "No validation results published yet."

    lines = [
        "Parcel-level crop type classification on PASTIS-R PixelSet (Zenodo 5745151). "
        "Scores are from validation. Fold 5 is reserved for testing.",
        "",
        "| Run | Model | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda row: (-row["macro_iou"], row["run"])):
        sensors = " + ".join(row["sensors"])
        train = ", ".join(map(str, row["train_folds"]))
        val = ", ".join(map(str, row["validation_folds"]))
        epoch = row["selected_epoch"] or "—"
        lines.append(
            f'| [`{row["run"]}`](results/{row["run"]}.json) | {row["model"]} | '
            f'{sensors} | {train} → {val} | {row["train_parcels"]:,} | '
            f'{row["validation_parcels"]:,} | {row["seed"]} | {epoch} | '
            f'{row["macro_iou"]:.2%} | {row["accuracy"]:.2%} |'
        )
    lines.extend([
        "",
        "Run links contain the recorded parcel and feature fingerprints. "
        "Sequence runs lack a training parcel fingerprint. "
        "Check inputs and training budgets before comparing scores.",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    data = [json.loads(path.read_text()) for path in sorted((ROOT / "results").glob("*.json"))]
    table = render(data)
    readme = ROOT / "README.md"
    source = readme.read_text()
    before, start, section = source.partition(START)
    _, end, after = section.partition(END)
    if not start or not end or source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("README.md must contain one ordered pair of leaderboard markers")
    readme.write_text(f"{before}{start}\n\n{table}\n\n{end}{after}")
    (ROOT / "LEADERBOARD.md").write_text(f"# PASTIS-R parcel leaderboard\n\n{table}\n")
