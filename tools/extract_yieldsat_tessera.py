"""TESSERA annual embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

TESSERA (Feng et al. 2025, arXiv:2506.20380; University of Cambridge) is a 128-d, 10 m embedding per pixel and
year (v1.1 "dclimate", global, 2017-2025), learned from that year's Sentinel-1 and Sentinel-2 time series. It is
read anonymously from the Source Cooperative Zarr store with geotessera's byte-range reader: per field, only the
window covering the field's pixels. Needs the layered env /lustre/shared/agriscience-benchmark/envs/tessera
(benchmark-v2 + geotessera 0.11.0).

Year: as for AlphaEarth, the calendar year of the harvest date; 2016 harvests (Germany) get NaN.

    envs/tessera/bin/python -m tools.extract_yieldsat_tessera --subset GER-W   (CPU job with internet)

Writes features/yieldsat/tessera/<subset>.npy float16 (pixels, 128) + <subset>.json.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.warp import transform as warp

from tools import yieldsat as ys
from tools.extract_yieldsat_aef import harvest_years

YEARS = range(2017, 2026)
PAD_DEG = 0.0005


def field_embeddings(gt, lat, lon, year):
    bbox = (lon.min() - PAD_DEG, lat.min() - PAD_DEG, lon.max() + PAD_DEG, lat.max() + PAD_DEG)
    patch, transform, crs = gt.read_region(bbox, year=year)
    patch = np.asarray(patch)
    if patch.ndim != 3 or patch.shape[-1] != 128:
        raise ValueError(f"Unexpected TESSERA patch shape {patch.shape}")
    x, y = warp("EPSG:4326", crs, lon, lat)
    r, c = rasterio.transform.rowcol(transform, np.asarray(x), np.asarray(y))
    r, c = np.asarray(r), np.asarray(c)
    out = np.full((len(lat), 128), np.nan, np.float32)
    ok = (r >= 0) & (r < patch.shape[0]) & (c >= 0) & (c < patch.shape[1])
    out[ok] = patch[r[ok], c[ok]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    from geotessera import GeoTesseraZarr

    out_dir = ys.FEATURES / "tessera"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    gt = GeoTesseraZarr()
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    latlon = np.load(ys.FEATURES / "latlon" / f"{args.subset}.npy")
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    years = harvest_years()
    out = np.full((len(fidx), 128), np.nan, np.float16)
    no_year = [f for f in fields if years[f] not in YEARS]
    t0 = time.time()

    def work(j):
        a, b = bounds[j], bounds[j + 1]
        f = fields[fidx[a]]
        if years[f] not in YEARS:
            return j, None
        for attempt in range(3):
            try:
                return j, field_embeddings(gt, latlon[a:b, 0], latlon[a:b, 1], years[f])
            except Exception as error:  # network hiccups: retry, then fail loudly
                if attempt == 2:
                    raise RuntimeError(f"{f}: {error}") from error
                time.sleep(5)

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
            "feature_names": [f"tessera_{i:03d}" for i in range(128)],
            "year_rule": "calendar year of the harvest date", "fields_without_tessera_year": len(no_year),
            "pixels_missing": missing, "seconds": round(time.time() - t0, 1),
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "source": f"TESSERA {gt.model_version} {gt.dataset} via geotessera GeoTesseraZarr ({gt.url})"}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "fields_without_tessera_year", "pixels_missing",
                                           "seconds")}), flush=True)


if __name__ == "__main__":
    main()
