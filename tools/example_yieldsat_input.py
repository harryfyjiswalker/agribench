"""Worked example: add your own model to YieldSAT as a new input (README, "Add your own model").

Copy this file, put your model in embed(), and run it once per subset as a Slurm job:

    sbatch prepare.sbatch tools.example_yieldsat_input --name $USER-ndvi --subset URG-S

As shipped, embed() turns each pixel's raw Sentinel-2 series into an NDVI series (24 values), so the whole route
can be tested end to end without a GPU or model weights. It writes, next to the existing inputs:

    features/yieldsat/<name>/<subset>.npy    (pixels, d) float16, one row per pixel, in the row order of
                                             features/yieldsat/v1/<subset>-meta.npz
    features/yieldsat/<name>/<subset>.json   kind, feature_names, rows and the SHA-256 of that meta file

The runner checks the row count and the meta hash, so a file built for the wrong rows is refused. After all nine
subsets exist, `input=<name>` works like any other input, alone or joined with + (e.g. <name>+mind).

What embed() gets: the pixels' raw S2 values from the v1 cache, float32, NaN where there was no observation, with
columns named <band>_t<step> (12 bands B01..B12 incl. B8A, steps t00..t23 over the season). If your model needs
other data (image patches, dates, the field geometry), load it in embed() from $project/datasets/yieldsat; the
meta file's field_idx, row and col tell you which field and which pixel of the field's raster each row is.
"""
import argparse
import datetime as dt
import json
import subprocess
import time

import numpy as np

from tools import yieldsat as ys

CHUNK = 200_000  # pixels per batch (bounded memory)
METHOD = "NDVI per Sentinel-2 time step (worked example)"  # one line describing your input; saved in the .json


def embed(s2, names):
    """Your model goes here. s2: (pixels, 288) float32 raw S2 (NaN = missing); names: its column names.
    Return (pixels, d) features and their d names. The example returns NDVI per time step."""
    col = {n: i for i, n in enumerate(names)}
    steps = sorted({n.split("_")[1] for n in names})
    red = s2[:, [col[f"B04_{t}"] for t in steps]]
    nir = s2[:, [col[f"B08_{t}"] for t in steps]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    return ndvi, [f"ndvi_{t}" for t in steps]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="input name, e.g. $USER-mymodel (lower case, no '+')")
    p.add_argument("--subset", required=True, choices=ys.SUBSETS)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if "+" in a.name or a.name in ("v1", "s2", "s2-adm"):
        raise SystemExit(f"Choose another name than {a.name!r}")

    v1 = ys.FEATURES / "v1"
    info = json.loads((v1 / f"{a.subset}.json").read_text())
    cols = info["inputs"]["s2"]
    names = [info["feature_names"][c] for c in cols]
    meta_path = v1 / f"{a.subset}-meta.npz"
    rows = len(np.load(meta_path)["field_idx"])
    cache = np.load(v1 / f"{a.subset}.npy", mmap_mode="r")
    if cache.shape[0] != rows:
        raise SystemExit("v1 cache and meta file disagree; ask the maintainers")

    out = ys.FEATURES / a.name
    out.mkdir(exist_ok=True)
    npy, js = out / f"{a.subset}.npy", out / f"{a.subset}.json"
    if js.exists() and not a.overwrite:
        raise SystemExit(f"{js} exists; pass --overwrite to rebuild it")

    t0 = time.time()
    result, feature_names = None, None
    for start in range(0, rows, CHUNK):
        x = np.asarray(cache[start:start + CHUNK][:, cols], dtype=np.float32)
        x[x == info.get("s2_nodata_to_nan", 65535)] = np.nan
        f, feature_names = embed(x, names)
        f = np.asarray(f, dtype=np.float32)
        if f.ndim != 2 or len(f) != len(x):
            raise SystemExit(f"embed() must return (pixels, d); got {f.shape} for {len(x)} pixels")
        if result is None:
            result = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(rows, f.shape[1]))
        if np.nanmax(np.abs(f)) > 6.5e4:
            raise SystemExit("Values too large for float16; rescale them in embed()")
        result[start:start + len(f)] = f
        print(f"{start + len(f):>9,} / {rows:,} pixels", flush=True)
    result.flush()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    js.write_text(json.dumps({
        "subset": a.subset,
        "kind": "dense",
        "feature_names": feature_names,
        "rows": rows,
        "aligned_with": f"v1/{a.subset}-meta.npz",
        "v1_meta_sha256": ys.sha256(meta_path),
        "method": METHOD,
        "code": {"script": "tools/example_yieldsat_input.py", "commit": commit},
        "seconds": round(time.time() - t0, 1),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=1) + "\n")
    print(f"Wrote {npy} ({rows:,} x {len(feature_names)}) and {js}")


if __name__ == "__main__":
    main()
