"""Frozen AnySat-Base embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

In-season patch input. AnySat's Sentinel-2 branch runs a temporal attention encoder (L-TAE) on every pixel's series
and a spatial transformer over patches, so each pixel is embedded with its neighbours as context:

- Space: the field's own raster, rebuilt from the v1 rows (row, col), cut into TILE x TILE pixel tiles (240 m) with
  AnySat patch size 10 m, i.e. one patch per pixel. Output mode "patch": the pixel's patch token after the spatial
  transformer (768-d). Cells of a tile outside the yield map get the field's mean series. The context is therefore
  the field itself, not the surrounding landscape (the preprocessed file holds only yield-map pixels).
- Time: the field's own season (the 24 slots of the preprocessed file; 1, note 5) at real dates (file time
  reference + slot offset, as for `galileo`). AnySat takes one date list per tile and has no per-timestep mask, so
  slots with a valid observation in fewer than MIN_SLOT_COVER of the field's pixels are dropped, and each pixel's
  remaining gaps are filled by linear interpolation in time (nearest observation at the ends). Dates are days since
  the first day of the month of the field's first kept slot, as AnySat's PASTIS loader counts days from a season
  reference.
- Bands: AnySat's 10 S2 bands (B2-B8, B8A, B11, B12). AnySat normalises each dataset with its own band mean and std;
  here one fixed set computed from a seeded sample of valid S2 values pooled over all nine subsets (no labels).

    python -m tools.embed_yieldsat_anysat --subset GER-W     (GPU job; pinned source put on sys.path)

Writes features/yieldsat/anysat/<subset>.npy float32 (pixels, 768), rows aligned with v1/<subset>.npy, plus
<subset>.json.
"""
import argparse
import datetime as dt
import json
import sys
import time

import numpy as np
import torch

from tools import yieldsat as ys
from tools.build_yieldsat_s2_aligned import fill, interpolate
from tools.embed_yieldsat_galileo import check_dates, slot_months

SOURCE = ys.PROJECT / "upstream" / "anysat" / "AnySat"
SOURCE_REVISION = "5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a"
HF_REPO, HF_REVISION = "g-astruc/AnySat", "63f252119babfe896d200f88631496f86eddacd3"
WEIGHTS = ys.PROJECT / "upstream" / "anysat" / f"AnySat@{HF_REVISION}" / "models" / "AnySat.pth"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # AnySat S2 order
TILE = 24              # pixels per tile side (240 m); 576 patches, under AnySat's advised 1024
PATCH_M = 10           # AnySat patch size in metres: one S2 pixel per patch
MIN_SLOT_COVER = 0.1   # a slot is kept if at least this share of the field's pixels is valid in it
NORM_ROWS = 20000      # rows sampled per subset for the band statistics


def band_stats():
    """Per-band mean and std of valid S2 values, pooled over a seeded row sample of all nine subsets."""
    rng = np.random.default_rng(0)
    values = []
    for s in ys.SUBSETS:
        names = json.loads((ys.FEATURES / "v1" / f"{s}.json").read_text())["feature_names"]
        cols = [names.index(f"{b}_t{t:02d}") for t in range(ys.N_STEPS) for b in BANDS]
        cache = np.load(ys.FEATURES / "v1" / f"{s}.npy", mmap_mode="r")
        rows = np.sort(rng.choice(len(cache), min(NORM_ROWS, len(cache)), replace=False))
        v = np.asarray(cache[rows])[:, cols].reshape(-1, len(BANDS))
        values.append(v[np.isfinite(v).all(1)])
    v = np.concatenate(values)
    return v.mean(0).astype(np.float32), v.std(0).astype(np.float32), int(len(v))


def field_series(values, days):
    """values (pixels, slots, bands) with NaN gaps -> gap-free series: linear in time, nearest at the ends."""
    x = interpolate(values, days, days)
    fv, _ = fill(x, days)
    bv, _ = fill(x, days, reverse=True)
    x = np.where(np.isfinite(x), x, np.where(np.isfinite(fv), fv, bv))
    mean = np.nanmean(x, axis=0)                     # field mean series for pixels with no observation at all
    return np.where(np.isfinite(x), x, mean[None]), int((~np.isfinite(values).all(-1)).all(1).sum())


def load_model(device):
    sys.path.insert(0, str(SOURCE))
    from hubconf import AnySat

    model = AnySat(model_size="base", flash_attn=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["state_dict"]
    model.model.load_state_dict(state)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--tiles-per-batch", type=int, default=32)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "anysat"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    if not torch.cuda.is_available():
        raise SystemExit("Run this in a GPU job")
    device = torch.device("cuda")
    model = load_model(device)
    mean, std, norm_n = band_stats()

    info = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())
    names = info["feature_names"]
    s2_cols = [[names.index(f"{b}_t{s:02d}") for s in range(ys.N_STEPS)] for b in BANDS]
    cols = np.array(s2_cols).T  # (24, 10)
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx, prow, pcol = [str(f) for f in meta["fields"]], meta["field_idx"], meta["row"], meta["col"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    _, exists, dates, country = slot_months(args.subset, fields)
    date_check = check_dates(args.subset, fields, dates, country, cache, bounds[:-1], s2_cols)
    print(json.dumps({"subset": args.subset, "date_check": date_check, "norm_values": norm_n}), flush=True)

    n = len(fidx)
    out = np.lib.format.open_memmap(out_dir / f".{args.subset}.npy.tmp", mode="w+", dtype=np.float32,
                                    shape=(n, 768))
    t0, kept_slots, no_obs, n_tiles = time.time(), [], 0, 0
    with torch.inference_mode():
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            i = fidx[a]
            raw = np.asarray(cache[a:b])[:, cols.ravel()].reshape(b - a, ys.N_STEPS, len(BANDS))
            valid = np.isfinite(raw).all(-1) & exists[i][None]
            slots = np.flatnonzero(valid.mean(0) >= MIN_SLOT_COVER)
            if len(slots) == 0:
                slots = np.flatnonzero(valid.any(0))
            d = [dates[fields[i]][s] for s in slots]
            ref = d[0].replace(day=1)
            days = np.array([(x - ref).days for x in d], np.float64)
            v = np.where(valid[:, slots, None], raw[:, slots], np.nan).astype(np.float32)
            series, missing = field_series(v, days)          # (pixels, T, 10)
            no_obs += missing
            kept_slots.append(len(slots))
            series = (series - mean) / std
            r, c = prow[a:b] - prow[a:b].min(), pcol[a:b] - pcol[a:b].min()
            H, W = int(np.ceil((r.max() + 1) / TILE) * TILE), int(np.ceil((c.max() + 1) / TILE) * TILE)
            T = len(slots)
            grid = np.broadcast_to(series.mean(0)[:, :, None, None], (T, len(BANDS), H, W)).copy()
            grid[:, :, r, c] = series.transpose(1, 2, 0)
            tiles = grid.reshape(T, len(BANDS), H // TILE, TILE, W // TILE, TILE).transpose(2, 4, 0, 1, 3, 5)
            ty, tx = tiles.shape[:2]
            tiles = tiles.reshape(ty * tx, T, len(BANDS), TILE, TILE)
            n_tiles += len(tiles)
            dates_t = torch.as_tensor(days, dtype=torch.float32, device=device)
            emb = np.empty((ty * tx, TILE, TILE, 768), np.float32)
            for t_a in range(0, len(tiles), args.tiles_per_batch):
                x = torch.as_tensor(tiles[t_a:t_a + args.tiles_per_batch], device=device)
                o = model({"s2": x, "s2_dates": dates_t.expand(len(x), T)}, patch_size=PATCH_M, output="patch")
                emb[t_a:t_a + len(x)] = o.float().cpu().numpy()
            emb = emb.reshape(ty, tx, TILE, TILE, 768).transpose(0, 2, 1, 3, 4).reshape(H, W, 768)
            e = emb[r, c]
            if not np.isfinite(e).all():
                raise RuntimeError(f"Non-finite AnySat output for field {j}")
            out[a:b] = e
            if j % 50 == 0:
                rate = b / max(time.time() - t0, 1e-6)
                print(f"field {j}/{len(bounds) - 1}, {b}/{n} pixels, {rate:.0f} px/s, "
                      f"ETA {(n - b) / rate / 60:.1f} min", flush=True)
    out.flush()
    del out
    (out_dir / f".{args.subset}.npy.tmp").rename(target)
    kept_slots = np.array(kept_slots)
    meta_info = {"subset": args.subset, "kind": "dense", "rows": int(n),
                 "feature_names": [f"anysat_{i:03d}" for i in range(768)],
                 "aligned_with": f"v1/{args.subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
                 "model": "AnySat-Base", "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
                 "source": "https://github.com/gastruc/AnySat", "source_revision": SOURCE_REVISION,
                 "weights_sha256": ys.sha256(WEIGHTS),
                 "design": {"temporal": "in-season", "spatial": "patch (field raster tiles)"},
                 "input": f"S2 10 bands; field raster from v1 row/col in {TILE}x{TILE}-pixel tiles, AnySat patch "
                          f"size {PATCH_M} m (one pixel per patch), off-field cells = field mean series; the "
                          f"field's own season slots with >= {MIN_SLOT_COVER:.0%} valid pixels, per-pixel gaps "
                          "linearly interpolated in time; dates = days since the 1st of the first kept slot's month",
                 "normalization": {"kind": "(x - mean) / std, pooled over a seeded sample of all nine subsets",
                                   "mean": mean.round(2).tolist(), "std": std.round(2).tolist(),
                                   "values": norm_n},
                 "pooling": "AnySat output='patch': the pixel's own patch token",
                 "kept_slots": {"median": float(np.median(kept_slots)), "min": int(kept_slots.min()),
                                "max": int(kept_slots.max())},
                 "pixels_without_s2": no_obs, "tiles": n_tiles,
                 "date_check": date_check, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    print(json.dumps({k: meta_info[k] for k in ("subset", "rows", "pixels_without_s2", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
