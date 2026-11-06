"""Check frozen Galileo Base inference on three PASTIS-R PixelSet parcels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("pastis", type=Path)
    parser.add_argument("galileo", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                    str(args.pastis / "code"), str(args.galileo)]
    from tabular import START, STOP, WIDTH, summarize
    from dataloader_pixelset import PASTIS_Dataset_PixelSet
    import single_file_galileo as galileo

    # PASTIS S2 order, with the leading zero removed from single-digit band names.
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    if galileo.S2_BANDS != bands:
        raise ValueError("Galileo and PASTIS S2 band orders disagree")
    indices = [galileo.SPACE_TIME_BANDS.index(band) for band in bands]
    groups = [i for i, name in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX)
              if name.startswith("S2_")]
    stats = json.loads((args.galileo / "config/normalization.json").read_text())
    s2_stats = stats[str(len(galileo.SPACE_TIME_BANDS))]
    mean = np.asarray(s2_stats["mean"], dtype=np.float32)[indices]
    std = np.asarray(s2_stats["std"], dtype=np.float32)[indices]

    ds = PASTIS_Dataset_PixelSet(
        folder=str(args.data), sats=["S2"], folds=[1, 2, 3, 4], norm=False,
        reference_date="2018-09-01", cache=False, geom_features=False,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Run the smoke test in an allocated NVIDIA GPU job")
    encoder = galileo.Encoder.load_from_folder(args.checkpoint, device).to(device).eval()

    patch_ids = {ds.id_patches[parcel] for parcel in ds.id_parcels}
    offsets = np.concatenate([ds.get_dates(patch, "S2") for patch in patch_ids])
    if not offsets.size or np.any((offsets < START) | (offsets >= STOP)):
        raise ValueError("Acquisition dates are empty or outside the existing S2 grid")
    first, last = (np.array([offsets.min(), offsets.max()]) - START) // WIDTH
    t = int(last - first + 1)
    limit = encoder.pos_embed.shape[0]
    print(f"Folds 1–4: occupied span is bins {first}–{last} ({t} slots, encoder limit {limit})")
    if t > limit:
        raise ValueError("Full observation window does not fit. Choose and match a window before scoring.")
    left = START + WIDTH * np.arange(first, last + 1)
    reference = np.datetime64("2018-09-01")
    midpoints = reference + ((left + np.minimum(left + WIDTH, STOP)) // 2).astype("timedelta64[D]")
    month = midpoints.astype("datetime64[M]").astype(np.int64) % 12

    patches = set()
    with torch.inference_mode():
        for parcel in ds.id_parcels:
            patch = ds.id_patches[parcel]
            if patch in patches:
                continue
            values = np.load(args.data / "DATA_S2" / f"S2_{parcel}.npy", allow_pickle=False)
            dates = np.asarray(ds.get_dates(patch, "S2"))
            if values.ndim != 3 or values.shape[:2] != (len(dates), len(bands)):
                raise ValueError(f"Unexpected S2 shape or acquisition count for parcel {parcel}")
            if not values.size or not np.isfinite(values).all():
                raise ValueError(f"Empty or non-finite S2 values for parcel {parcel}")
            raw = summarize(values, dates, len(bands)).reshape(-1, len(bands))[first:last + 1]
            observed = np.isfinite(raw).all(axis=1)
            if not observed.any():
                raise ValueError(f"No valid S2 bins for parcel {parcel}")
            normalized = np.zeros_like(raw)
            normalized[observed] = (raw[observed] - (mean - 2 * std)) / (4 * std)

            def tensor(shape, masked=False):
                return (torch.ones if masked else torch.zeros)(shape, device=device)

            space_time = tensor((1, 1, 1, t, len(galileo.SPACE_TIME_BANDS)))
            space_time[..., indices] = torch.as_tensor(normalized, device=device)
            space_time_mask = tensor(
                (1, 1, 1, t, len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)), masked=True
            )
            for group in groups:
                space_time_mask[0, 0, 0, :, group] = torch.as_tensor(
                    ~observed, dtype=space_time_mask.dtype, device=device
                )
            encoded = encoder(
                space_time,
                tensor((1, 1, 1, len(galileo.SPACE_BANDS))),
                tensor((1, t, len(galileo.TIME_BANDS))),
                tensor((1, len(galileo.STATIC_BANDS))),
                space_time_mask,
                tensor((1, 1, 1, len(galileo.SPACE_BAND_GROUPS_IDX)), masked=True),
                tensor((1, t, len(galileo.TIME_BAND_GROUPS_IDX)), masked=True),
                tensor((1, len(galileo.STATIC_BAND_GROUPS_IDX)), masked=True),
                torch.as_tensor(month[None, :], device=device),
                patch_size=1,
            )
            embedding = encoder.average_tokens(*encoded[:8])
            if embedding.ndim != 2 or embedding.shape[0] != 1 or not embedding.isfinite().all():
                raise ValueError(f"Invalid embedding for parcel {parcel}")
            print(f"Parcel {parcel}, patch {patch}: {len(dates)} acquisitions, "
                  f"{observed.sum()}/{t} observed bins, embedding {tuple(embedding.shape)}")
            patches.add(patch)
            if len(patches) == 3:
                print("Galileo Base smoke check passed")
                return
    raise ValueError("Expected parcels from three distinct patches")


if __name__ == "__main__":
    main()
