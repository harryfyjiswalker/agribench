"""Raw baseline: flattened monthly Sentinel-2 parcel medians (unobserved months are NaN)."""
from pathlib import Path

import numpy as np


def extract(*, inputs: Path, weights_dir=None, source_dir=None, device="cpu", batch_size=0):
    with np.load(inputs, allow_pickle=False) as table:
        ids, raw, observed = table["ids"], table["s2"], table["observed"]
    if raw.ndim != 3 or observed.shape != raw.shape[:2] or observed.dtype != np.bool_:
        raise ValueError("Expected s2 shaped (parcels, months, bands) and a boolean (parcels, months) mask")
    features = np.where(observed[:, :, None], raw, np.nan).astype(np.float32)
    return ids, features.reshape(len(ids), -1)
