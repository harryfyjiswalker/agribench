"""Shared YieldSAT definitions: paths, subsets, band groups, the field table and the region rule.

Everything here follows protocols/yieldsat.md; section numbers refer to that page.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("AGRIBENCH_PROJECT", "/lustre/shared/agriscience-benchmark"))
DATA = Path(os.environ.get("AGRIBENCH_YIELDSAT", PROJECT / "datasets" / "yieldsat"))
SPLITS = PROJECT / "splits" / "yieldsat"
FEATURES = PROJECT / "features" / "yieldsat"

COUNTRIES = {"Argentina": "ARG", "Brazil": "BRA", "Germany": "GER", "Uruguay": "URG"}
CROPS = {"corn": "C", "rapeseed": "R", "soybean": "S", "wheat": "W"}
# The nine country-crop subsets the paper reports (2.1).
SUBSETS = ["ARG-C", "ARG-S", "ARG-W", "BRA-C", "BRA-S", "BRA-W", "GER-R", "GER-W", "URG-S"]
# Crops grown in more than one country, for leave-one-country-out (2.3b).
LOCO_CROPS = ["corn", "soybean", "wheat"]

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
WEATHER_BANDS = ["temp_mean", "temp_max", "temp_min", "total_prec"]
TEMPORAL_BANDS = S2_BANDS + WEATHER_BANDS  # everything else in the file is static over time
N_STEPS = 24
S2_NODATA = 65535  # saturated / nodata reflectance, treated as missing (1, item 4)

# Region rule for leave-one-region-out (2.3).
MAX_REGIONS = 8
MIN_REGION_FIELDS = 10


def subset_name(country, crop):
    return f"{COUNTRIES[country]}-{CROPS[crop]}"


def preprocessed_file(country):
    return DATA / "Preprocessed" / country / country / "merge_s2-soil-dem-weather-coords.nc"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def field_table():
    """One row per field from the Raw metadata JSONs (country, crop, year, farm, centroid, quality)."""
    rows = []
    for path in sorted(glob.glob(str(DATA / "Raw" / "*" / "*" / "*" / "metadata-*.json"))):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "field_shared_name": m["field_shared_name"],
            "country": m["adm_units"]["country"],
            "crop": m["crop"],
            "year": int(m["year"]),
            "farm": m["farm_identifier"],
            "quality": m.get("yieldmap_quality"),
            "lat": float(m["centroid_latitude_wgs84"]),
            "lon": float(m["centroid_longitude_wgs84"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No field metadata found under {DATA / 'Raw'}")
    if df.field_shared_name.duplicated().any():
        raise ValueError("Duplicate field names in the metadata")
    unknown = set(df.country) - COUNTRIES.keys() | set(df.crop) - CROPS.keys()
    if unknown:
        raise ValueError(f"Unexpected countries or crops: {sorted(unknown)}")
    df["subset"] = [subset_name(c, k) for c, k in zip(df.country, df.crop)]
    return df.sort_values("field_shared_name").reset_index(drop=True)


def regions(fields):
    """Group a subset's farms into regions of neighbouring farms (2.3). Returns farm -> region id.

    Ward clustering of farm centroids (km) into at most MAX_REGIONS groups, then repeatedly merge the smallest
    region with fewer than MIN_REGION_FIELDS fields into the region with the nearest centroid. Farms are never
    split. Region ids are renumbered 0.. by decreasing field count (ties by first farm name).
    """
    from sklearn.cluster import AgglomerativeClustering

    farms = fields.groupby("farm").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("field_shared_name", "size"))
    farms = farms.sort_index()
    lat0 = np.radians(farms.lat.mean())
    xy = np.c_[farms.lon * 111.32 * np.cos(lat0), farms.lat * 110.57]
    if len(farms) > MAX_REGIONS:
        lab = AgglomerativeClustering(n_clusters=MAX_REGIONS, linkage="ward").fit_predict(xy)
    else:
        lab = np.arange(len(farms))
    lab = pd.Series(lab, index=farms.index)
    while True:
        size = farms.n.groupby(lab).sum()
        small = size[size < MIN_REGION_FIELDS]
        if small.empty or len(size) <= 2:
            break
        r = small.idxmin()
        centre = {g: xy[(lab == g).to_numpy()].mean(0) for g in size.index}
        others = [g for g in size.index if g != r]
        lab[lab == r] = min(others, key=lambda g: (np.linalg.norm(centre[g] - centre[r]), g))
    size = farms.n.groupby(lab).sum()
    first = pd.Series(lab.index, index=lab.index).groupby(lab).min()
    order = sorted(size.index, key=lambda g: (-size[g], first[g]))
    renumber = {g: i for i, g in enumerate(order)}
    return lab.map(renumber)
