"""Frozen Presto embeddings (pinned upstream revision, default_model.pt) from prepared PASTIS-R inputs.

Follows Presto's own crop-type evaluation (presto/eval/croptype_france_eval.py): Sentinel-2 bands and
NDVI are visible; S1, ERA5, SRTM and Dynamic World are masked; location is (lat, lon) in degrees.
Presto needs the same number of masked tokens for every item in a batch, so parcels are batched by
their number of observed months (the result equals running them one by one).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

WEIGHTS_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"
NUM_CHANNELS = 17          # Presto: 0-1 S1, 2-11 S2 (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), 12-13 ERA5, 14-15 SRTM, 16 NDVI
S2_SLICE = slice(2, 12)    # the ten PASTIS-R bands, same order as Presto's S2 channels
NDVI = 16
B4_POS, B8_POS = 2, 6      # positions of B4 and B8 within those ten bands
DW_MISSING = 9             # Presto's "no Dynamic World class" value


def extract(*, inputs, weights_dir=None, source_dir=None, device="cpu", batch_size=512):
    if source_dir is None:
        raise ValueError("Presto requires the pinned source checkout (--source-dir)")
    source_dir = Path(source_dir)
    weights = (Path(weights_dir) if weights_dir else source_dir / "data") / "default_model.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"{weights} is not the pinned Presto default_model.pt")

    with np.load(inputs, allow_pickle=False) as table:
        ids, raw, observed = table["ids"], table["s2"], table["observed"]
        months, coords = table["months"], table["coords"]
    if raw.ndim != 3 or raw.shape[2] != 10:
        raise ValueError("Expected s2 shaped (parcels, months, 10 bands)")
    n, t, _ = raw.shape
    if observed.shape != (n, t) or observed.dtype != np.bool_:
        raise ValueError("Expected a boolean observed mask shaped (parcels, months)")
    if coords.shape != (n, 2) or not np.isfinite(coords).all():
        raise ValueError("Expected finite (lat, lon) coords for every parcel")
    if months.shape != (t,) or months.dtype.kind not in "iu" or months.min() < 0 or months.max() > 11:
        raise ValueError("Expected zero-indexed months 0-11, one per timestep")
    if not observed.any(axis=1).all() or not np.isfinite(raw[observed]).all():
        raise ValueError("Every parcel needs at least one valid observed month")

    sys.path.insert(0, str(source_dir.resolve()))
    import single_file_presto as presto

    model = presto.Presto.construct()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    encoder = model.encoder.to(device).eval()

    s2 = np.where(observed[:, :, None], raw, 0.0).astype(np.float32) / 1e4
    b8, b4 = s2[:, :, B8_POS], s2[:, :, B4_POS]
    total = b8 + b4
    x = np.zeros((n, t, NUM_CHANNELS), dtype=np.float32)
    x[:, :, S2_SLICE] = s2
    x[:, :, NDVI] = np.where(total > 0, (b8 - b4) / np.where(total > 0, total, 1.0), 0.0)
    mask = np.ones((n, t, NUM_CHANNELS), dtype=np.float32)    # 1 = masked
    mask[:, :, S2_SLICE] = 0
    mask[:, :, NDVI] = 0
    mask[~observed] = 1                                       # unobserved months: everything masked

    counts = observed.sum(axis=1)
    features = np.empty((n, encoder.embedding_size), dtype=np.float32)
    month = torch.as_tensor(months, dtype=torch.long, device=device)
    with torch.inference_mode():
        for k in np.unique(counts):
            rows = np.flatnonzero(counts == k)
            for start in range(0, len(rows), batch_size):
                idx = rows[start:start + batch_size]
                out = encoder(
                    torch.from_numpy(x[idx]).to(device),
                    torch.full((len(idx), t), DW_MISSING, dtype=torch.long, device=device),
                    torch.from_numpy(coords[idx].astype(np.float32)).to(device),
                    torch.from_numpy(mask[idx]).to(device),
                    month.expand(len(idx), t),
                    eval_task=True,
                )
                features[idx] = out.float().cpu().numpy()
    return ids, features
