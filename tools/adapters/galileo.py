"""Extract frozen Galileo Base embeddings from prepared PASTIS-R inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from tools.extract_galileo import encode


def extract(
    *,
    inputs: Path,
    weights_dir: Path | None,
    source_dir: Path | None,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return parcel IDs and their Galileo embeddings in the same row order."""
    if weights_dir is None or source_dir is None:
        raise ValueError("Galileo requires checkpoint and source directories")

    with np.load(inputs, allow_pickle=False) as table:
        ids = table["ids"]
        raw = table["s2"]
        observed = table["observed"]
        months = table["months"]

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    if raw.ndim != 3 or raw.shape != (len(ids), len(months), len(bands)):
        raise ValueError("Expected Sentinel-2 inputs shaped (parcels, months, bands)")
    if observed.shape != raw.shape[:2] or observed.dtype != np.bool_:
        raise ValueError("Expected a boolean observation mask shaped (parcels, months)")
    if months.ndim != 1 or months.dtype.kind not in "iu" or np.any((months < 0) | (months > 11)):
        raise ValueError("Expected zero-indexed month values from 0 to 11")
    if not observed.any(axis=1).all() or not np.isfinite(raw[observed]).all():
        raise ValueError("Every parcel must have at least one valid observation")

    sys.path.insert(0, str(source_dir.resolve()))
    import single_file_galileo as galileo

    if galileo.S2_BANDS != bands:
        raise ValueError("Galileo and PASTIS-R Sentinel-2 band orders disagree")

    indices = [galileo.SPACE_TIME_BANDS.index(band) for band in bands]
    groups = [
        index
        for index, name in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX)
        if name.startswith("S2_")
    ]

    stats_path = source_dir / "config" / "normalization.json"
    stats = json.loads(stats_path.read_text())[str(len(galileo.SPACE_TIME_BANDS))]
    mean = np.asarray(stats["mean"], dtype=np.float32)[indices]
    std = np.asarray(stats["std"], dtype=np.float32)[indices]

    values = np.zeros(raw.shape, dtype=np.float32)
    values[observed] = (raw[observed] - (mean - 2 * std)) / (4 * std)

    encoder = galileo.Encoder.load_from_folder(weights_dir, torch.device("cpu"))
    encoder = encoder.to(device).eval()
    month_tensor = torch.as_tensor(months, dtype=torch.long, device=device)

    features = np.empty((len(ids), 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            features[start:end] = encode(
                encoder,
                galileo,
                values[start:end],
                observed[start:end],
                month_tensor,
                indices,
                groups,
                device,
            )

    return ids, features