"""Copy this module when adding an architecture."""

from pathlib import Path

import numpy as np


def extract(
    *, inputs: Path, weights_dir: Path | None, source_dir: Path | None,
    device: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load prepared inputs, run the frozen model, and return (parcel_ids, features).

    The two arrays share their row order. The runner checks IDs and writes the files.
    The checkpoint and source directories are supplied when the preset declares them.
    """
    raise NotImplementedError("Implement model-specific feature extraction here")
