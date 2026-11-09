"""Build the packed EarthShift input cache that every model reads (protocols/earthshift.md, section 0).

    python -m tools.build_earthshift_inputs                     # all four countries
    python -m tools.build_earthshift_inputs --country denmark   # one country

Needs the frozen split files (tools.make_earthshift_splits). For each country, keeps every (chip, window)
sample that any task uses outside the gaps, and writes under $AGRIBENCH_PROJECT/inputs/earthshift/v1/:
    <country>-images.npy   uint16 (N, 4, 256, 256), FTW reflectance x 10,000, bands R, G, B, NIR (memory-mappable)
    <country>-masks.npy    uint8  (N, 256, 256), 0 background, 1 field interior, 2 boundary
    <country>-index.csv    row, aoi_id, window (row = position in the two arrays)
    <country>.json         shapes, label counts, source zip and split hashes, code provenance
Chips are read straight from the zip (nothing is unpacked). Existing caches are never overwritten.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

from tools import earthshift as es

VERSION = "v1"


def read_tif(z, name, count, dtype):
    with MemoryFile(z.read(name)) as mem, mem.open() as src:
        if (src.count, src.height, src.width) != (count, es.CHIP_PX, es.CHIP_PX):
            raise ValueError(f"{name}: shape {(src.count, src.height, src.width)}")
        return src.read().astype(dtype, copy=False)


def needed_samples(country):
    rows = []
    for task in es.TASKS:
        d = pd.read_csv(es.SPLITS / f"{task}.csv")
        rows.append(d[(d.country == country) & (d.role != "gap")][["aoi_id", "window"]])
    return pd.concat(rows).drop_duplicates().sort_values(["aoi_id", "window"]).reset_index(drop=True)


def build_country(country, out):
    index = needed_samples(country)
    n = len(index)
    if (out / f"{country}.json").exists():
        sys.exit(f"{out / country}.json exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{country}-", dir=out))
    images = np.lib.format.open_memmap(stage / f"{country}-images.npy", mode="w+", dtype=np.uint16,
                                       shape=(n, es.N_BANDS, es.CHIP_PX, es.CHIP_PX))
    masks = np.lib.format.open_memmap(stage / f"{country}-masks.npy", mode="w+", dtype=np.uint8,
                                      shape=(n, es.CHIP_PX, es.CHIP_PX))
    t0 = time.time()
    with zipfile.ZipFile(es.zip_path(country)) as z:
        for i, (aoi, w) in enumerate(zip(index.aoi_id, index.window)):
            images[i] = read_tif(z, es.image_name(aoi, w), es.N_BANDS, np.uint16)
            masks[i] = read_tif(z, es.label_name(aoi), 1, np.uint8)[0]
            if (i + 1) % 500 == 0:
                print(f"  {country}: {i + 1}/{n} samples, {time.time() - t0:.0f} s", flush=True)
    images.flush()
    masks.flush()
    labels, counts = np.unique(masks, return_counts=True)
    if set(labels.tolist()) - set(es.CLASSES):
        raise RuntimeError(f"{country}: unexpected label values {labels.tolist()}")
    index.insert(0, "row", np.arange(n))
    index.to_csv(stage / f"{country}-index.csv", index=False, lineterminator="\n")
    meta = {"country": country, "version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user": os.environ.get("USER"), "samples": n, "chips": int(index.aoi_id.nunique()),
            "images": {"dtype": "uint16", "shape": list(images.shape), "bands": ["R", "G", "B", "NIR"],
                       "scale": "reflectance x 10000"},
            "masks": {"dtype": "uint8", "shape": list(masks.shape), "classes": ["background", "field", "boundary"],
                      "pixel_counts": {int(k): int(v) for k, v in zip(labels, counts)}},
            "source": {"zip": str(es.zip_path(country)), "zip_sha256": es.sha256(es.zip_path(country)),
                       "splits_manifest_sha256": es.sha256(es.SPLITS / "manifest.json")},
            "code": {f: es.sha256(Path(__file__).parent / f) for f in ("earthshift.py", "build_earthshift_inputs.py")}}
    del images, masks
    meta["files"] = {f.name: es.sha256(f) for f in sorted(stage.iterdir())}
    (stage / f"{country}.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in stage.iterdir():
        os.chmod(f, 0o664)
        os.replace(f, out / f.name)
    stage.rmdir()
    print(f"{country}: {n} samples from {meta['chips']} chips in {time.time() - t0:.0f} s; "
          f"label pixels {meta['masks']['pixel_counts']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", choices=es.COUNTRIES, action="append")
    p.add_argument("--out", type=Path, default=es.INPUTS / VERSION)
    args = p.parse_args()
    if not (es.SPLITS / "manifest.json").exists():
        sys.exit("No frozen splits; run python -m tools.make_earthshift_splits first")
    for country in args.country or es.COUNTRIES:
        build_country(country, args.out)


if __name__ == "__main__":
    main()
