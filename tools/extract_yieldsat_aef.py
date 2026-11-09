"""AlphaEarth Foundations (AEF) annual embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

AEF ("Satellite Embedding V1, annual", Google / Google DeepMind, CC-BY 4.0) is a 64-d, 10 m embedding per pixel
and calendar year (2017-2025), computed from that year's imagery. We read it anonymously from the Source
Cooperative mirror (tge-labs/aef), with HTTP range requests: per field, only the window of the 8192x8192 COG
that covers the field's pixels is fetched.

Year: the calendar year of the field's harvest date (the year that holds most of the growing season for every
crop here; for southern summer crops it holds the Jan-Apr peak). Fields harvested in 2016 (Germany) have no AEF
year and get NaN (the runner imputes them); the count is recorded.

    python -m tools.extract_yieldsat_aef --subset GER-W      (CPU job with internet; one subset per array task)

Needs features/yieldsat/latlon/. Writes features/yieldsat/aef/<subset>.npy float16 (pixels, 64) + <subset>.json.
Dequantisation (from the AEF README): v = (q / 127.5) ** 2 * sign(q); q = -128 is nodata.
"""
import argparse
import datetime as dt
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

for key, value in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
                   "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2", "VSI_CACHE": "TRUE",
                   "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES"}.items():
    os.environ.setdefault(key, value)

import rasterio  # noqa: E402  (after the GDAL settings)
from rasterio.warp import transform as warp  # noqa: E402

from tools import yieldsat as ys  # noqa: E402

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
HTTP_PREFIX = "https://data.source.coop/"
YEARS = range(2017, 2026)


def harvest_years():
    out = {}
    for path in glob.glob(str(ys.DATA / "Raw" / "*" / "*" / "*" / "metadata-*.json")):
        with open(path) as f:
            m = json.load(f)
        out[m["field_shared_name"]] = dt.datetime.strptime(m["harvesting_date"], "%d.%m.%Y").year
    return out


def dequantise(q):
    v = (q.astype(np.float32) / 127.5) ** 2 * np.sign(q)
    v[q == -128] = np.nan
    return v


def field_embeddings(lat, lon, year, tiles):
    """(pixels, 64) AEF values for one field's pixels in one year."""
    out = np.full((len(lat), 64), np.nan, np.float32)
    cand = tiles[(tiles.year == year) & (tiles.wgs84_west <= lon.max()) & (tiles.wgs84_east >= lon.min())
                 & (tiles.wgs84_south <= lat.max()) & (tiles.wgs84_north >= lat.min())]
    todo = np.ones(len(lat), bool)
    for _, t in cand.iterrows():
        x, y = warp("EPSG:4326", t.crs, lon[todo], lat[todo])
        x, y = np.asarray(x), np.asarray(y)
        inside = (x >= t.utm_west) & (x < t.utm_east) & (y > t.utm_south) & (y <= t.utm_north)
        if not inside.any():
            continue
        url = "/vsicurl/" + t.path.replace(S3_PREFIX, HTTP_PREFIX)
        with rasterio.open(url) as src:
            r, c = rasterio.transform.rowcol(src.transform, x[inside], y[inside])
            r, c = np.asarray(r), np.asarray(c)
            r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
            q = src.read(window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))  # (64, h, w) int8
        idx = np.flatnonzero(todo)[inside]
        out[idx] = dequantise(q[:, r - r0, c - c0].T)
        todo[idx] = False
        if not todo.any():
            break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "aef"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    index_path = out_dir / "_aef_index.parquet"
    if not index_path.exists():
        raise SystemExit(f"Download the tile index first: curl -o {index_path} {INDEX_URL}")
    tiles = pd.read_parquet(index_path, columns=["path", "year", "crs", "utm_west", "utm_south", "utm_east",
                                                 "utm_north", "wgs84_west", "wgs84_south", "wgs84_east",
                                                 "wgs84_north"])
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    latlon = np.load(ys.FEATURES / "latlon" / f"{args.subset}.npy")
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    years = harvest_years()
    out = np.full((len(fidx), 64), np.nan, np.float16)
    no_year = [f for f in fields if years[f] not in YEARS]
    t0 = time.time()

    def work(j):
        a, b = bounds[j], bounds[j + 1]
        f = fields[fidx[a]]
        if years[f] not in YEARS:
            return j, None
        return j, field_embeddings(latlon[a:b, 0], latlon[a:b, 1], years[f], tiles)

    done = 0
    with ThreadPoolExecutor(args.threads) as pool:
        for j, emb in pool.map(work, range(len(bounds) - 1)):
            if emb is not None:
                out[bounds[j]:bounds[j + 1]] = emb.astype(np.float16)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(bounds) - 1} fields, {time.time() - t0:.0f}s", flush=True)
    missing = int(np.isnan(out[:, 0]).sum())
    tmp = out_dir / f".{args.subset}.npy.tmp"
    with tmp.open("wb") as fh:
        np.save(fh, out)
    tmp.rename(target)
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)),
            "feature_names": [f"aef_{i:02d}" for i in range(64)],
            "year_rule": "calendar year of the harvest date", "fields_without_aef_year": len(no_year),
            "pixels_missing": missing, "seconds": round(time.time() - t0, 1),
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "source": "AlphaEarth Foundations Satellite Embedding V1 annual (Google, Google DeepMind), CC-BY 4.0, "
                      "via Source Cooperative tge-labs/aef", "index_sha256": ys.sha256(index_path)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "fields_without_aef_year", "pixels_missing",
                                           "seconds")}), flush=True)


if __name__ == "__main__":
    main()
