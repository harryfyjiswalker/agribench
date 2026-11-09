"""Shared EarthShift definitions: paths, the four tasks, the chip table and the leak-free split rule.

Everything here follows protocols/earthshift.md; section numbers refer to that page. The raw data are the
Fields of The World (FTW) country zips; chips are read straight from them, never unpacked (section 0).
"""
import hashlib
import io
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("AGRIBENCH_PROJECT", "/lustre/shared/agriscience-benchmark"))
ZIPS = Path(os.environ.get("AGRIBENCH_EARTHSHIFT", PROJECT / "datasets" / "earthshift" / "ftw-zips"))
SPLITS = PROJECT / "splits" / "earthshift"
INPUTS = PROJECT / "inputs" / "earthshift"

COUNTRIES = ["germany", "cambodia", "denmark", "south_africa"]
WINDOWS = ("a", "b")
CHIP_PX = 256
N_BANDS = 4  # R, G, B, NIR
CLASSES = (0, 1, 2)  # background, field interior, boundary
TOUCH_M = 50.0  # chips closer than this (edges and corners) count as touching (2.1b)

# Germany grid g2 is cut into 14 latitude bands of 24 chips, band 1 the northernmost (2.1b).
DE_BANDS = 14
DE_CHIPS_PER_BAND = 24
DE_TRAIN_BANDS = range(1, 8)
DE_GAP_BANDS = (8, 10)
DE_VAL_BANDS = (9,)
DE_LATE_BANDS = range(11, 15)  # extra train in the location tasks, ID test in the year task

# The four tasks (2.1). Every role lists (country, windows); the chips come from build_splits().
TASKS = {
    "de-kh": "location: Germany -> Cambodia",
    "de-dk": "location: Germany -> Denmark",
    "za-season": "temporal: South Africa window A -> window B",
    "de-year": "temporal: Germany 2018 (grid g2) -> 2019 (grid g1)",
}
ROLES = ["train", "val", "id_test", "ood_test", "gap"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def zip_path(country):
    return ZIPS / f"{country}.zip"


def image_name(aoi_id, window):
    return f"s2_images/window_{window}/{aoi_id}.tif"


def label_name(aoi_id):
    return f"label_masks/semantic_3class/{aoi_id}.tif"


def chip_table(country):
    """One row per chip: aoi_id, FTW split, footprint, and whether each window and the label exist."""
    with zipfile.ZipFile(zip_path(country)) as z:
        names = set(z.namelist())
        chips = gpd.read_parquet(io.BytesIO(z.read(f"chips_{country}.parquet")))
    chips = chips[["aoi_id", "split", "geometry"]].copy()
    chips["country"] = country
    chips["grid"] = chips.aoi_id.str.split("_").str[0]
    chips["label"] = [label_name(a) in names for a in chips.aoi_id]
    for w in WINDOWS:
        chips[f"has_{w}"] = [image_name(a, w) in names for a in chips.aoi_id]
    return chips.reset_index(drop=True)


def to_utm(chips):
    return chips.to_crs(chips.estimate_utm_crs())


def touching(a, b):
    """Boolean per chip in `a`: within TOUCH_M metres of any chip in `b` (both GeoDataFrames in one UTM CRS)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    tree = b.sindex
    hits = tree.query(a.geometry, predicate="dwithin", distance=TOUCH_M)
    out = np.zeros(len(a), dtype=bool)
    out[np.unique(hits[0])] = True
    return out


def germany_bands(de):
    """Latitude band (1 = north) for each Germany grid-g2 chip, from footprint centroids (2.1b)."""
    g2 = to_utm(de[de.grid == "g2"])
    y = g2.geometry.centroid.y.to_numpy()
    order = np.argsort(-y)  # north first
    band = np.empty(len(y), dtype=int)
    current, last = 1, None
    for i in order:
        if last is not None and last - y[i] > 300:  # bands are 1,535 m apart; chips in one band within metres
            current += 1
        band[i] = current
        last = y[i]
    counts = np.bincount(band)[1:]
    if len(counts) != DE_BANDS or (counts != DE_CHIPS_PER_BAND).any():
        raise RuntimeError(f"Expected {DE_BANDS} bands of {DE_CHIPS_PER_BAND} chips in grid g2, got {counts.tolist()}")
    return pd.Series(band, index=g2.index)


def usable(chips, window):
    return chips.label & chips[f"has_{window}"]


def samples(chips, windows, role):
    """Rows (country, aoi_id, window, role) for chips whose image in that window and label both exist."""
    rows = [(c, a, w, role) for w in windows for c, a, ok in zip(chips.country, chips.aoi_id, usable(chips, w)) if ok]
    return pd.DataFrame(rows, columns=["country", "aoi_id", "window", "role"])


def gap_rows(chips):
    return pd.DataFrame({"country": chips.country, "aoi_id": chips.aoi_id, "window": "-", "role": "gap"})


def build_splits(tables):
    """The four task tables (2.1, 2.1b), from {country: chip_table(country)}."""
    de, kh, dk, za = (tables[c] for c in COUNTRIES)
    band = germany_bands(de)
    de = de.assign(band=band.reindex(de.index))
    in_bands = lambda bands: de[de.band.isin(list(bands))]
    de_test = de[de.grid == "g1"]
    if set(de_test.split) != {"test"} or set(de[de.grid == "g2"].split) - {"train", "val"}:
        raise RuntimeError("Germany: expected FTW test = grid g1, train/val = grid g2")

    out = {}
    for task, ood in (("de-kh", kh), ("de-dk", dk)):
        out[task] = pd.concat([
            samples(in_bands([*DE_TRAIN_BANDS, *DE_LATE_BANDS]), WINDOWS, "train"),
            samples(in_bands(DE_VAL_BANDS), WINDOWS, "val"),
            samples(de_test, WINDOWS, "id_test"),
            samples(ood[ood.split == "test"], WINDOWS, "ood_test"),
            gap_rows(in_bands(DE_GAP_BANDS)),
        ])
    out["de-year"] = pd.concat([
        samples(in_bands(DE_TRAIN_BANDS), WINDOWS, "train"),
        samples(in_bands(DE_VAL_BANDS), WINDOWS, "val"),
        samples(in_bands(DE_LATE_BANDS), WINDOWS, "id_test"),
        samples(de_test, WINDOWS, "ood_test"),
        gap_rows(in_bands(DE_GAP_BANDS)),
    ])

    # South Africa: keep FTW's val/test blocks, drop train chips touching val or test and val chips touching test.
    zu = to_utm(za)
    tr, va, te = (zu[zu.split == s] for s in ("train", "val", "test"))
    keep_tr = ~touching(tr, pd.concat([va, te]))
    keep_va = ~touching(va, te)
    out["za-season"] = pd.concat([
        samples(za.loc[tr.index[keep_tr]], ("a",), "train"),
        samples(za.loc[va.index[keep_va]], ("a",), "val"),
        samples(za.loc[te.index], ("a",), "id_test"),
        samples(za.loc[te.index], ("b",), "ood_test"),
        gap_rows(za.loc[np.r_[tr.index[~keep_tr], va.index[~keep_va]]]),
    ])
    return {t: d.reset_index(drop=True) for t, d in out.items()}


def check_splits(splits, tables):
    """Enforce the 2.1b rule; raise on any violation."""
    for task, d in splits.items():
        used = d[d.role != "gap"]
        if used.duplicated(["country", "aoi_id", "window"]).any():
            raise RuntimeError(f"{task}: a sample appears twice")
        dev = used[used.role.isin(["train", "val", "id_test"])]
        # Both windows of a chip on one side: a chip may hold only one of train / val / id_test.
        per_chip = dev.groupby(["country", "aoi_id"]).role.nunique()
        if (per_chip > 1).any():
            raise RuntimeError(f"{task}: chips in two roles: {per_chip[per_chip > 1].index.tolist()[:5]}")
        # OOD chips never used for training or validation.
        ood = set(map(tuple, used[used.role == "ood_test"][["country", "aoi_id"]].to_numpy()))
        fit = set(map(tuple, used[used.role.isin(["train", "val"])][["country", "aoi_id"]].to_numpy()))
        if ood & fit:
            raise RuntimeError(f"{task}: OOD chips also used for train/val")
        # No touching between train, val and id_test chips (per country, in UTM).
        for country, g in dev.groupby("country"):
            chips = to_utm(tables[country]).set_index("aoi_id")
            roles = g.drop_duplicates("aoi_id").set_index("aoi_id").role
            for r1, r2 in (("train", "val"), ("train", "id_test"), ("val", "id_test")):
                a = chips.loc[roles.index[roles == r1]]
                b = chips.loc[roles.index[roles == r2]]
                if touching(a, b).any():
                    raise RuntimeError(f"{task}: {country} {r1} chips touch {r2} chips")
