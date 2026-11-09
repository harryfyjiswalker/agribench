"""Build and freeze the YieldSAT fold files (protocols/yieldsat.md, 2.2, 2.3, 2.3b and 2.8).

    python -m tools.make_yieldsat_folds            # write, once (refuses to overwrite)
    python -m tools.make_yieldsat_folds --check    # rebuild in memory and compare with the frozen files

Writes under $AGRIBENCH_PROJECT/splits/yieldsat/:
    fields.csv                        one row per field (name, country, crop, year, farm, region, centroid, quality)
    <subset>/cv10.csv, loyo.csv, loro.csv   columns field_shared_name, fold
    loco/<crop>.csv                   columns field_shared_name, fold (fold = held-out country)
    manifest.json                     SHA-256 of every file above plus the code that made them
These files hold field names and locations: keep them on Lustre, never commit them to the public repo.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tools import yieldsat as ys

SEED = 42


def build():
    fields = ys.field_table()
    fields["region"] = -1
    tables = {}
    for subset in ys.SUBSETS:
        d = fields[fields.subset == subset]
        if d.empty:
            raise ValueError(f"No fields for subset {subset}")
        idx = d.index

        # 2.2: stratified by farm (region in the paper's wording), grouped by field.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # farms with fewer than 10 fields
            cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
            fold = np.empty(len(d), dtype=int)
            for k, (_, test) in enumerate(cv.split(d, y=d.farm, groups=d.field_shared_name)):
                fold[test] = k
        tables[f"{subset}/cv10.csv"] = pd.DataFrame({"field_shared_name": d.field_shared_name, "fold": fold})

        # 2.3: one fold per year, in calendar order.
        years = sorted(d.year.unique())
        tables[f"{subset}/loyo.csv"] = pd.DataFrame(
            {"field_shared_name": d.field_shared_name, "fold": d.year.map({y: i for i, y in enumerate(years)})})

        # 2.3: one fold per region of neighbouring farms.
        region = d.farm.map(ys.regions(d))
        fields.loc[idx, "region"] = region.to_numpy()
        tables[f"{subset}/loro.csv"] = pd.DataFrame({"field_shared_name": d.field_shared_name, "fold": region})

    # 2.3b: per crop, one fold per country.
    for crop in ys.LOCO_CROPS:
        d = fields[fields.crop == crop]
        countries = sorted(d.country.unique())
        if len(countries) < 2:
            raise ValueError(f"{crop} is grown in fewer than two countries")
        tables[f"loco/{crop}.csv"] = pd.DataFrame(
            {"field_shared_name": d.field_shared_name, "fold": d.country.map({c: i for i, c in enumerate(countries)})})

    for name, t in tables.items():
        if t.field_shared_name.duplicated().any() or t.fold.isna().any():
            raise RuntimeError(f"Bad fold table {name}")
    cols = ["field_shared_name", "subset", "country", "crop", "year", "farm", "region", "quality", "lat", "lon"]
    tables["fields.csv"] = fields[cols]
    return tables


def summary(tables):
    lines = []
    for name, t in tables.items():
        if name == "fields.csv":
            continue
        counts = t.fold.value_counts().sort_index().tolist()
        lines.append(f"{name:<18} {t.fold.nunique():>2} folds  fields per fold {counts}")
    return "\n".join(lines)


def to_csv_bytes(t):
    return t.to_csv(index=False, lineterminator="\n").encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare a fresh build with the frozen files")
    p.add_argument("--out", type=Path, default=ys.SPLITS)
    args = p.parse_args()

    tables = build()
    print(summary(tables), flush=True)

    if args.check:
        bad = [n for n, t in tables.items() if not (args.out / n).is_file()
               or (args.out / n).read_bytes() != to_csv_bytes(t)]
        if bad:
            sys.exit(f"MISMATCH with frozen folds in {args.out}: {bad}")
        print(f"OK: frozen folds in {args.out} match a fresh build")
        return

    if (args.out / "manifest.json").exists():
        sys.exit(f"{args.out} already holds frozen folds; refusing to overwrite (use --check to verify them)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Build in a staging folder next to the destination, then move it into place in one step.
    stage = Path(tempfile.mkdtemp(prefix=".yieldsat-folds-", dir=args.out.parent))
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": os.environ.get("USER"), "seed": SEED,
                "code": {f: ys.sha256(Path(__file__).parent / f) for f in ("yieldsat.py", "make_yieldsat_folds.py")},
                "files": {}}
    for name, t in tables.items():
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(to_csv_bytes(t))
        manifest["files"][name] = ys.sha256(dest)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in [stage, *stage.rglob("*")]:
        os.chmod(path, 0o775 if path.is_dir() else 0o664)
    if args.out.exists():
        if any(args.out.iterdir()):
            sys.exit(f"{args.out} exists and is not empty; refusing to overwrite. Staged copy: {stage}")
        args.out.rmdir()
    os.replace(stage, args.out)
    print(f"Wrote {len(tables)} files to {args.out}")


if __name__ == "__main__":
    main()
