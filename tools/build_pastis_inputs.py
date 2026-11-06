"""Build the PASTIS-R input table shared by frozen parcel encoders."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import pandas as pd

FOLD_COUNTS = {1: 17229, 2: 17559, 3: 17045, 4: 17478, 5: 16640}
REFERENCE = np.datetime64("2018-09-01")
START, WIDTH, FIRST, STOP = -200, 30, 7, 21
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def bins_for_patch(series):
    if isinstance(series, str):
        series = json.loads(series)
    days = [str(series[key]) for key in sorted(series, key=int)]
    dates = np.asarray([f"{day[:4]}-{day[4:6]}-{day[6:]}" for day in days],
                       dtype="datetime64[D]")
    offsets = (dates - REFERENCE).astype(np.int64)
    if not len(offsets) or np.isnat(dates).any() or (np.diff(offsets) <= 0).any():
        raise ValueError("Invalid Sentinel-2 dates in patch metadata")
    return (offsets - START) // WIDTH


def build(data, output):
    if output.exists():
        raise FileExistsError(output)
    parcel_csv = data / "metadata_parcel.csv"
    patch_geojson = data / "metadata.geojson"
    fields = ["ID_PARCEL", "ID_PATCH", "Fold"]
    parcels = pd.read_csv(parcel_csv, usecols=fields, dtype="Int64").sort_values("ID_PARCEL")
    if (parcels.isna().any().any() or parcels.ID_PARCEL.duplicated().any()
            or parcels.Fold.value_counts().to_dict() != FOLD_COUNTS):
        raise ValueError("Expected unique parcel IDs and the official five fold sizes")
    ids = parcels.ID_PARCEL.to_numpy(dtype=np.int64)
    patches = gpd.read_file(patch_geojson)
    patches = patches.set_index(patches.ID_PATCH.astype(int), verify_integrity=True)
    centres = patches.to_crs(2154).geometry.centroid.to_crs(4326)
    coords = pd.DataFrame({"lat": centres.y, "lon": centres.x}).loc[parcels.ID_PATCH]
    coords = coords.to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all():
        raise ValueError("Missing patch centroid coordinates")

    patch_bins = {patch: bins_for_patch(series) for patch, series in patches["dates-S2"].items()}
    s2 = np.full((len(ids), STOP - FIRST, len(BANDS)), np.nan, dtype=np.float32)
    outside = 0
    for i, (parcel, patch) in enumerate(zip(ids, parcels.ID_PATCH, strict=True)):
        bins = patch_bins[patch]
        values = np.load(data / "DATA_S2" / f"S2_{parcel}.npy", allow_pickle=False)
        if values.ndim != 3 or values.shape[:2] != (len(bins), len(BANDS)) or not values.shape[2]:
            raise ValueError(f"Unexpected Sentinel-2 array shape for parcel {parcel}: {values.shape}")
        keep = (bins >= FIRST) & (bins < STOP)
        outside += int((~keep).sum())
        pixel_medians = np.median(values[keep], axis=-1)
        retained_bins = bins[keep]
        for index in np.unique(retained_bins):
            s2[i, index - FIRST] = np.median(pixel_medians[retained_bins == index], axis=0)
        if (i + 1) % 10_000 == 0:
            print(f"Prepared {i + 1:,} / {len(ids):,} parcels", flush=True)

    finite = np.isfinite(s2)
    observed = finite.all(axis=-1)
    if (finite.any(axis=-1) != observed).any() or not observed.any(axis=1).all():
        raise ValueError("Found a partially observed bin or a parcel without observations")
    offsets = START + WIDTH * np.arange(FIRST, STOP) + WIDTH // 2
    months = (REFERENCE + offsets.astype("timedelta64[D]")).astype("datetime64[M]")
    months = months.astype(np.int64) % 12

    metadata_hashes = {}
    for path in (parcel_csv, patch_geojson):
        with path.open("rb") as file:
            metadata_hashes[path.name] = hashlib.file_digest(file, "sha256").hexdigest()
    provenance = {
        "reference_date": str(REFERENCE), "grid_start_days": START, "bin_width_days": WIDTH,
        "bins_inclusive": [FIRST, STOP - 1], "bands": BANDS,
        "composite": "per-band pixel median, then acquisition median within each bin",
        "coordinates": "parent patch centroid in EPSG:2154, returned as WGS84 latitude/longitude",
        "metadata_sha256": metadata_hashes,
        "out_of_window_acquisitions_counted_per_parcel": outside,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            np.savez(temporary, ids=ids, s2=s2, observed=observed, months=months,
                     coords=coords, provenance=json.dumps(provenance))
            temporary.flush()
            os.link(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    print(f"Saved {len(ids):,} parcels to {output}")
    print(f"Acquisitions outside the fixed window, counted per parcel: {outside:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.data, args.output)
