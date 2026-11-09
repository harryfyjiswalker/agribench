"""Frozen Prithvi-EO-2.0-300M-TL embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

Multi-temporal patch input, in-season. Prithvi-EO-2.0 is a ViT with 3D (time x 16 x 16 px) patches, pretrained on
4-frame HLS stacks with each frame's (year, day of year) as a temporal encoding; the TL variant also takes a location,
which is left out here (location only enters through the location encoders).

- Space: the field's Raw S2 raster (its bounding box with the real surroundings) via tools/yieldsat_patches.py,
  224 x 224 windows at native 10 m, reflection outside the raster. Each labelled pixel uses the window whose centre
  is nearest; its embedding is the token grid (14 x 14, one token per 160 m) averaged over the 4 frames and sampled
  bilinearly at the pixel centre (`sample_tokens`).
- Time: 4 dates from the field's own season, label-free, sorted: the first clear in-season acquisition, and the
  helper's green-up, peak and senescence dates (`phenology_dates`; clear = SCL vegetation/soil/water/unclassified on
  at least 90% of the field's labelled pixels, relaxed to 60% if needed). A date can repeat when the season has
  fewer clear acquisitions. Prithvi adds its temporal encoding only for its pretraining frame count, 4.
- Bands: Prithvi's 6 HLS bands from S2: B02, B03, B04, B8A (HLS "B05" narrow NIR), B11, B12, in DN, normalised with
  the checkpoint's mean and std; NaN pixels take the band's median over the image.

    python -m tools.embed_yieldsat_prithvi --subset GER-W     (GPU job; pinned model code put on sys.path)

Writes features/yieldsat/prithvi/<subset>.npy float32 (pixels, 1024), rows aligned with v1/<subset>.npy, plus
<subset>.json.
"""
import argparse
import inspect
import json
import sys
import time

import numpy as np
import torch

from tools import yieldsat as ys
from tools import yieldsat_patches as yp

HF_REPO, HF_REVISION = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", "63adbd39c271da4c42f447e69b1a7c91a338cdc9"
SOURCE = ys.PROJECT / "upstream" / "prithvi" / f"Prithvi-EO-2.0-300M-TL@{HF_REVISION}"
WEIGHTS = SOURCE / "Prithvi_EO_V2_300M_TL.pt"
S2_FOR_HLS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # HLS B02, B03, B04, B05, B06, B07
SIZE, FRAMES = 224, 4


def load_model(device):
    sys.path.insert(0, str(SOURCE))
    from prithvi_mae import PrithviMAE

    cfg = json.loads((SOURCE / "config.json").read_text())["pretrained_cfg"]
    params = inspect.signature(PrithviMAE.__init__).parameters
    model = PrithviMAE(**{k: v for k, v in cfg.items() if k in params})
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if "pos_embed" not in k}  # fixed sin-cos, rebuilt by the model
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("pos_embed" not in k for k in missing):
        raise ValueError(f"Prithvi checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    if cfg["bands"] != ["B02", "B03", "B04", "B05", "B06", "B07"] or cfg["num_frames"] != FRAMES:
        raise ValueError("Unexpected Prithvi band list or frame count")
    return model.encoder.to(device).eval(), cfg


def season_dates(field):
    curve = yp.greenness_curve(field)
    usable, threshold = yp._usable(curve)
    first = min(c[0] for c in usable)
    return sorted([first, *yp.phenology_dates(field, curve)]), threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--windows-per-batch", type=int, default=8)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "prithvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    if not torch.cuda.is_available():
        raise SystemExit("Run this in a GPU job")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    encoder, cfg = load_model(device)
    band_idx = [yp.RAW_S2_BANDS.index(b) for b in S2_FOR_HLS]
    mean = np.asarray(cfg["mean"], np.float32)[:, None, None]
    std = np.asarray(cfg["std"], np.float32)[:, None, None]
    grid = SIZE // cfg["patch_size"][1]

    fields = list(yp.fields(args.subset))
    n = fields[-1].rows.stop
    out = np.lib.format.open_memmap(out_dir / f".{args.subset}.npy.tmp", mode="w+", dtype=np.float32,
                                    shape=(n, cfg["embed_dim"]))
    t0, n_windows, thresholds, distinct, offsets = time.time(), 0, [], [], []
    with torch.inference_mode():
        for j, f in enumerate(fields):
            dates, threshold = season_dates(f)
            thresholds.append(threshold)
            distinct.append(len(set(dates)))
            offsets.append([(d - f.sowing).days for d in dates])
            stack = np.stack([(yp.fill_nan(f.s2(d))[band_idx] - mean) / std for d in dates])  # (4, 6, H, W)
            wins = yp.windows(*stack.shape[2:], SIZE)
            which = yp.assign(f.row, f.col, wins, SIZE)
            coords = torch.tensor([[d.year, d.timetuple().tm_yday] for d in dates], dtype=torch.float32,
                                  device=device)
            emb = np.empty((len(f.row), cfg["embed_dim"]), np.float32)
            for w0 in range(0, len(wins), args.windows_per_batch):
                ws = wins[w0:w0 + args.windows_per_batch]
                x = np.stack([np.stack([yp.crop(fr, r0, c0, SIZE) for fr in stack], 1) for r0, c0 in ws])  # B,C,T,H,W
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
                tokens = encoder.forward_features(x, coords.expand(len(ws), FRAMES, 2), None)[-1][:, 1:]
                tokens = tokens.reshape(len(ws), FRAMES, grid, grid, -1).mean(1).permute(0, 3, 1, 2)  # B,D,g,g
                for k, (r0, c0) in enumerate(ws):
                    sel = which == w0 + k
                    if sel.any():
                        emb[sel] = yp.sample_tokens(tokens[k], f.row[sel], f.col[sel], r0, c0, SIZE)
            n_windows += len(wins)
            if not np.isfinite(emb).all():
                raise RuntimeError(f"Non-finite Prithvi output for field {j}")
            out[f.rows] = emb
            if j % 25 == 0:
                rate = f.rows.stop / max(time.time() - t0, 1e-6)
                print(f"field {j}/{len(fields)}, {f.rows.stop}/{n} pixels, ETA {(n - f.rows.stop) / rate / 60:.1f} min",
                      flush=True)
    out.flush()
    del out
    (out_dir / f".{args.subset}.npy.tmp").rename(target)
    offsets = np.array(offsets)
    meta_info = {"subset": args.subset, "kind": "dense", "rows": int(n),
                 "feature_names": [f"prithvi_{i:04d}" for i in range(cfg["embed_dim"])],
                 "aligned_with": f"v1/{args.subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
                 "model": "Prithvi-EO-2.0-300M-TL encoder", "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
                 "weights_sha256": ys.sha256(WEIGHTS),
                 "design": {"temporal": "in-season, 4 dates", "spatial": "patch (Raw raster, 224 px windows)"},
                 "input": "S2 B02,B03,B04,B8A,B11,B12 (as HLS B02-B07) DN, checkpoint mean/std; field's Raw raster "
                          "via tools/yieldsat_patches.py, 224x224 windows at 10 m, reflection padding; 4 dates: "
                          "first clear in-season acquisition + green-up, peak, senescence (phenology_dates); "
                          "temporal coords (year, doy); no location coords",
                 "pooling": "last-layer patch tokens averaged over the 4 frames, bilinear at the pixel centre",
                 "token_size_m": 160, "windows": n_windows,
                 "clear_threshold_used": {str(t): int(sum(1 for x in thresholds if x == t)) for t in set(thresholds)},
                 "distinct_dates": {str(k): int(sum(1 for x in distinct if x == k)) for k in sorted(set(distinct))},
                 "median_days_since_sowing": np.median(offsets, 0).tolist(),
                 "helper": "tools/yieldsat_patches.py", "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    print(json.dumps({k: meta_info[k] for k in ("subset", "rows", "windows", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
