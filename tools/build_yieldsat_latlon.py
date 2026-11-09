"""Per-pixel WGS84 latitude/longitude for the YieldSAT feature caches (protocols/yieldsat.md, 2.5).

The preprocessed file only stores transformed coordinates (coord_x/y/z) and each pixel's row/col inside its field's
raster. Location encoders, AlphaEarth and TESSERA need real coordinates, so this tool reads one georeferenced S2
GeoTIFF per field from Raw/, maps every pixel centre (row + 0.5, col + 0.5) through the raster transform, and
reprojects it to WGS84. It checks each field's pixel mean against the field centroid in the Raw metadata.

    python -m tools.build_yieldsat_latlon [--subsets ARG-S GER-W]

Writes features/yieldsat/latlon/<subset>.npy  float64 (pixels, 2) = (lat, lon), rows aligned with v1/<subset>.npy,
and <subset>.json with provenance and the check results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp

from tools import yieldsat as ys

MAX_CENTROID_KM = 1.0  # a field's mean pixel position must lie this close to the metadata centroid


def km(lat1, lon1, lat2, lon2):
    lat0 = np.radians((lat1 + lat2) / 2)
    return float(np.hypot((lon1 - lon2) * 111.32 * np.cos(lat0), (lat1 - lat2) * 110.57))


def raster_of(field, country):
    folder = ys.DATA / "Raw" / country / country / field / "s2_images"
    tifs = sorted(folder.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No S2 GeoTIFF for {field} in {folder}")
    return tifs[0]


def build(subset, out_dir, fields_table):
    meta = np.load(ys.FEATURES / "v1" / f"{subset}-meta.npz")
    names, fidx, row, col = meta["fields"], meta["field_idx"], meta["row"], meta["col"]
    latlon = np.full((len(fidx), 2), np.nan)
    worst = 0.0
    b = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    for j in range(len(b) - 1):
        a, z = b[j], b[j + 1]
        field = str(names[fidx[a]])
        info = fields_table.loc[field]
        with rasterio.open(raster_of(field, info.country)) as src:
            if row[a:z].max() >= src.height or col[a:z].max() >= src.width:
                raise ValueError(f"{field}: pixel row/col outside its raster ({src.height}x{src.width})")
            x, y = rasterio.transform.xy(src.transform, row[a:z], col[a:z], offset="center")
            lon, lat = warp(src.crs, "EPSG:4326", np.asarray(x), np.asarray(y))
        latlon[a:z, 0], latlon[a:z, 1] = lat, lon
        d = km(np.mean(lat), np.mean(lon), info.lat, info.lon)
        worst = max(worst, d)
        if d > MAX_CENTROID_KM:
            raise ValueError(f"{field}: pixel mean is {d:.2f} km from the metadata centroid")
    if not np.isfinite(latlon).all():
        raise RuntimeError(f"{subset}: some pixels have no coordinates")
    target = out_dir / f"{subset}.npy"
    tmp = out_dir / f".{subset}.npy.tmp"
    with tmp.open("wb") as f:
        np.save(f, latlon)
    tmp.rename(target)
    info = {"subset": subset, "kind": "dense", "feature_names": ["lat", "lon"], "rows": int(len(latlon)),
            "aligned_with": f"v1/{subset}-meta.npz", "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz"),
            "max_centroid_offset_km": round(worst, 3),
            "method": "pixel centre of the field's first Raw S2 GeoTIFF, reprojected to EPSG:4326"}
    (out_dir / f"{subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="*", default=ys.SUBSETS)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "latlon"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_table = ys.field_table().set_index("field_shared_name")
    for s in args.subsets:
        if (out_dir / f"{s}.npy").exists():
            print(f"{s}: exists, skipping", flush=True)
            continue
        build(s, out_dir, fields_table)


if __name__ == "__main__":
    main()
