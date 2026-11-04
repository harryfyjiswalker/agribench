"""Extract frozen Galileo Base embeddings from the verified RF Sentinel-2 cache."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

WINDOW = slice(7, 21)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_s2(run):
    tables = json.loads((run / "feature-tables.json").read_text())
    if len(tables) != 1:
        raise ValueError("Expected one feature table from the full S2 RF run")
    table = tables[0]
    if (
        table["recipe"] != "pixel_median_then_30day_median_v1"
        or table["sensor"] != "S2"
        or table["folds"] != [1, 2, 3, 4]
        or table["bins"] != [-200, 600, 30]
        or table["reference_date"] != "2018-09-01"
    ):
        raise ValueError("The source cache does not match the PASTIS-R S2 protocol")
    with Path(table["file"]).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != table["sha256"]:
            raise ValueError("S2 cache hash differs from the RF run")
        file.seek(0)
        with np.load(file, allow_pickle=False) as cache:
            ids, values = cache["ids"], cache["features"]
    if (
        ids.shape != (69311,)
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or values.shape != (len(ids), 27 * 10)
    ):
        raise ValueError("Unexpected parcel IDs or S2 feature shape")
    values = values.reshape(len(ids), 27, 10)
    if np.isfinite(values[:, :WINDOW.start]).any() or np.isfinite(values[:, WINDOW.stop:]).any():
        raise ValueError("The verified Galileo window would drop S2 observations")
    values = values[:, WINDOW, :]
    finite = np.isfinite(values)
    observed = finite.all(axis=-1)
    if not observed.any(axis=1).all() or (finite.any(axis=-1) != observed).any():
        raise ValueError("Found a parcel without observations or a partially missing S2 bin")
    return ids, values, observed, table


def encode(encoder, galileo, values, observed, months, indices, groups, device):
    batch, time, _ = values.shape

    def tensor(shape, masked=False):
        return (torch.ones if masked else torch.zeros)(shape, device=device)

    space_time = tensor((batch, 1, 1, time, len(galileo.SPACE_TIME_BANDS)))
    space_time[..., indices] = torch.as_tensor(values[:, None, None, :, :], device=device)
    space_time_mask = tensor(
        (batch, 1, 1, time, len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)), masked=True
    )
    missing = torch.as_tensor(~observed, dtype=space_time_mask.dtype, device=device)
    for group in groups:
        space_time_mask[:, 0, 0, :, group] = missing
    encoded = encoder(
        space_time,
        tensor((batch, 1, 1, len(galileo.SPACE_BANDS))),
        tensor((batch, time, len(galileo.TIME_BANDS))),
        tensor((batch, len(galileo.STATIC_BANDS))),
        space_time_mask,
        tensor((batch, 1, 1, len(galileo.SPACE_BAND_GROUPS_IDX)), masked=True),
        tensor((batch, time, len(galileo.TIME_BAND_GROUPS_IDX)), masked=True),
        tensor((batch, len(galileo.STATIC_BAND_GROUPS_IDX)), masked=True),
        months.expand(batch, -1),
        patch_size=1,
    )
    embedding = encoder.average_tokens(*encoded[:8])
    if embedding.shape != (batch, 768) or not embedding.isfinite().all():
        raise ValueError("Expected one finite 768-dimensional Galileo Base embedding per parcel")
    return embedding.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rf_run", type=Path)
    parser.add_argument("galileo", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.suffix != ".npz" or args.output.exists():
        parser.error("Provide a positive batch size and a new .npz output path")

    sys.path.insert(0, str(args.galileo.resolve()))
    import single_file_galileo as galileo

    if not torch.cuda.is_available():
        raise RuntimeError("Run extraction in an NVIDIA GPU job")
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    if galileo.S2_BANDS != bands:
        raise ValueError("Galileo and PASTIS S2 band orders disagree")
    indices = [galileo.SPACE_TIME_BANDS.index(band) for band in bands]
    groups = [i for i, name in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX)
              if name.startswith("S2_")]

    stats_path = args.galileo / "config/normalization.json"
    stats = json.loads(stats_path.read_text())[str(len(galileo.SPACE_TIME_BANDS))]
    mean = np.asarray(stats["mean"], dtype=np.float32)[indices]
    std = np.asarray(stats["std"], dtype=np.float32)[indices]
    ids, raw, observed, table = load_s2(args.rf_run)
    values = np.zeros(raw.shape, dtype=np.float32)
    values[observed] = (raw[observed] - (mean - 2 * std)) / (4 * std)

    device = torch.device("cuda")
    encoder_hash = digest(args.checkpoint / "encoder.pt")
    if encoder_hash != "935f4912f08483d052d69d8e96518f52ef6827fe718a435cb5536fd9bb7737a7":
        raise ValueError("Expected the verified Galileo Base checkpoint")
    encoder = galileo.Encoder.load_from_folder(args.checkpoint, device).to(device).eval()
    grid_start, _, width = table["bins"]
    midpoint_offsets = grid_start + width * np.arange(WINDOW.start, WINDOW.stop) + width // 2
    midpoints = np.datetime64(table["reference_date"]) + midpoint_offsets.astype("timedelta64[D]")
    months = torch.as_tensor(midpoints.astype("datetime64[M]").astype(np.int64) % 12, device=device)
    print(f"Verified RF cache {table['sha256']}; extracting {len(ids):,} parcels", flush=True)

    features = np.empty((len(ids), 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            features[start:end] = encode(
                encoder, galileo, values[start:end], observed[start:end], months,
                indices, groups, device,
            )
            if start == 0 or start // args.batch_size % 100 == 0 or end == len(ids):
                print(f"{end:,}/{len(ids):,} parcels", flush=True)

    provenance = {
        "representation": "galileo_base_s2_parcel_30day",
        "source_feature_sha256": table["sha256"],
        "folds": table["folds"],
        "reference_date": table["reference_date"],
        "bins": table["bins"],
        "retained_bins_inclusive": [WINDOW.start, WINDOW.stop - 1],
        "bands": bands,
        "normalization_sha256": digest(stats_path),
        "normalization": "(x - (mean - 2*std)) / (4*std)",
        "missing_bins": "masked",
        "month": "calendar month of each bin midpoint",
        "pooling": "upstream average_tokens",
        "galileo_commit": subprocess.check_output(
            ["git", "-C", str(args.galileo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "encoder_sha256": encoder_hash,
        "encoder_config_sha256": digest(args.checkpoint / "config.json"),
        "extractor_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".npz", delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            np.savez(tmp, ids=ids, features=features, provenance=json.dumps(provenance))
            tmp.flush()
            staged.chmod(0o644)
            os.link(staged, args.output)
        finally:
            staged.unlink(missing_ok=True)
    print(f"Saved {features.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
