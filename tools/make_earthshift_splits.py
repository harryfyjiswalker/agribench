"""Build and freeze the EarthShift split files (protocols/earthshift.md, 2.1, 2.1b and 2.6).

    python -m tools.make_earthshift_splits            # write, once (refuses to overwrite)
    python -m tools.make_earthshift_splits --check    # rebuild in memory and compare with the frozen files

Reads the chip tables inside the FTW zips and writes under $AGRIBENCH_PROJECT/splits/earthshift/:
    <task>.csv      columns country, aoi_id, window, role (train / val / id_test / ood_test / gap)
    chips.csv       every chip of the four countries: country, aoi_id, FTW split, grid, Germany band, files present
    manifest.json   SHA-256 of every file above, of the zips and of the code that made them
The splits are fixed by chip position (no seed). The 2.1b rule is checked before anything is written.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from tools import earthshift as es


def build():
    tables = {c: es.chip_table(c) for c in es.COUNTRIES}
    splits = es.build_splits(tables)
    es.check_splits(splits, tables)
    band = es.germany_bands(tables["germany"])
    chips = pd.concat([t.drop(columns="geometry").assign(band=band if c == "germany" else None)
                       for c, t in tables.items()])
    chips["band"] = chips.band.astype("Int64")
    files = {f"{t}.csv": d for t, d in splits.items()}
    files["chips.csv"] = chips[["country", "aoi_id", "split", "grid", "band", "label", "has_a", "has_b"]]
    return files


def summary(files):
    lines = []
    for name, d in files.items():
        if name == "chips.csv":
            continue
        n = d.groupby("role").size().reindex(es.ROLES, fill_value=0)
        chips = d[d.role != "gap"].groupby("role").aoi_id.nunique().reindex(es.ROLES[:4], fill_value=0)
        lines.append(f"{name:<15} samples " + "  ".join(f"{r} {n[r]}" for r in es.ROLES[:4])
                     + f"  | chips " + " ".join(str(chips[r]) for r in es.ROLES[:4]) + f"  | gap chips {n['gap']}")
    return "\n".join(lines)


def to_csv_bytes(d):
    return d.to_csv(index=False, lineterminator="\n").encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare a fresh build with the frozen files")
    p.add_argument("--out", type=Path, default=es.SPLITS)
    args = p.parse_args()

    files = build()
    print(summary(files), flush=True)

    if args.check:
        bad = [n for n, d in files.items() if not (args.out / n).is_file()
               or (args.out / n).read_bytes() != to_csv_bytes(d)]
        if bad:
            sys.exit(f"MISMATCH with frozen splits in {args.out}: {bad}")
        print(f"OK: frozen splits in {args.out} match a fresh build")
        return

    if (args.out / "manifest.json").exists():
        sys.exit(f"{args.out} already holds frozen splits; refusing to overwrite (use --check to verify them)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".earthshift-splits-", dir=args.out.parent))
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": os.environ.get("USER"),
                "rule": "protocols/earthshift.md 2.1b", "tasks": es.TASKS,
                "zips": {c: es.sha256(es.zip_path(c)) for c in es.COUNTRIES},
                "code": {f: es.sha256(Path(__file__).parent / f) for f in ("earthshift.py", "make_earthshift_splits.py")},
                "files": {}}
    for name, d in files.items():
        (stage / name).write_bytes(to_csv_bytes(d))
        manifest["files"][name] = es.sha256(stage / name)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in [stage, *stage.rglob("*")]:
        os.chmod(path, 0o775 if path.is_dir() else 0o664)
    if args.out.exists():
        if any(args.out.iterdir()):
            sys.exit(f"{args.out} exists and is not empty; refusing to overwrite. Staged copy: {stage}")
        args.out.rmdir()
    os.replace(stage, args.out)
    print(f"Wrote {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
