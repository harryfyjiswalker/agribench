"""Frozen Presto embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

In-season input: each pixel's own season of Sentinel-2 observations (the 24 slots of the preprocessed file, which
hold one field's season; 1, note 5), placed on real calendar dates (file time reference + slot offset, as for
`galileo`). Presto was pretrained on monthly composites, so the observations are averaged per calendar month over
the field's span of months (first to last slot that exists), each month gets its calendar month token, and months
without a valid observation are masked. Bands: the 10 S2 bands Presto knows plus NDVI from the composite. S1, ERA5,
SRTM and Dynamic World are masked. Presto always adds a location token and cannot mask it, so latitude and longitude
are set to (0, 0) for every pixel: location only enters through the location encoders, as for `galileo`.

    python -m tools.embed_yieldsat_presto --subset GER-W     (GPU job; one subset per array task)

Writes features/yieldsat/presto/<subset>.npy float32 (pixels, 128), rows aligned with v1/<subset>.npy, plus
<subset>.json. Pixels with no valid S2 observation get NaN (the runner imputes them).
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

from tools import yieldsat as ys
from tools.embed_yieldsat_galileo import check_dates, slot_months

SOURCE = ys.PROJECT / "upstream" / "presto"
SOURCE_REVISION = "11e207a668a34336ced1d8e492a1bd5849b96c4a"
WEIGHTS = SOURCE / "data" / "default_model.pt"
WEIGHTS_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"
# Presto channels: 0-1 S1, 2-11 S2 (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), 12-13 ERA5, 14-15 SRTM, 16 NDVI
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
NUM_CHANNELS, S2_SLICE, NDVI = 17, slice(2, 12), 16
B4, B8 = BANDS.index("B04"), BANDS.index("B08")
DW_MISSING = 9
MAX_MONTHS = 24  # Presto's max_sequence_length


def month_grid(dates):
    """Per field: (24 slots,) month position on the field's grid (-1 = slot has no date), first calendar month."""
    pos = np.full((len(dates), ys.N_STEPS), -1, np.int64)
    first = np.zeros(len(dates), np.int64)
    spans = np.zeros(len(dates), np.int64)
    for i, d in enumerate(dates):
        key = np.array([x.year * 12 + x.month - 1 if x else -1 for x in d])
        ok = key >= 0
        lo, hi = key[ok].min(), key[ok].max()
        spans[i] = hi - lo + 1
        lo = max(lo, hi - MAX_MONTHS + 1)  # keep the last 24 months if a span were longer (none expected)
        keep = ok & (key >= lo)
        pos[i, keep] = key[keep] - lo
        first[i] = lo
    return pos, first, spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "presto"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    if not torch.cuda.is_available():
        raise SystemExit("Run this in a GPU job")
    device = torch.device("cuda")
    if ys.sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError(f"{WEIGHTS} is not the pinned Presto default_model.pt")

    sys.path.insert(0, str(SOURCE))
    import single_file_presto as presto

    info = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())
    names = info["feature_names"]
    s2_cols = [[names.index(f"{b}_t{s:02d}") for s in range(ys.N_STEPS)] for b in BANDS]
    cols = np.array(s2_cols).T  # (24, 10) slot-major
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    start = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")

    _, exists, dates, country = slot_months(args.subset, fields)
    date_check = check_dates(args.subset, fields, dates, country, cache, start, s2_cols)
    pos, first, spans = month_grid([dates[f] for f in fields])
    onehot = np.zeros((len(fields), ys.N_STEPS, MAX_MONTHS), np.float32)  # slot -> month position
    for i in range(len(fields)):
        s = np.flatnonzero(pos[i] >= 0)
        onehot[i, s, pos[i, s]] = 1
    month_tok = (first[:, None] + np.arange(MAX_MONTHS)[None]) % 12  # calendar month 0-11 per grid position
    print(json.dumps({"subset": args.subset, "date_check": date_check,
                      "month_span": {"median": float(np.median(spans)), "max": int(spans.max())}}), flush=True)

    model = presto.Presto.construct()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    encoder = model.encoder.to(device).eval()
    width = encoder.embedding_size

    n, T = len(fidx), ys.N_STEPS
    out = np.lib.format.open_memmap(out_dir / f".{args.subset}.npy.tmp", mode="w+", dtype=np.float32,
                                    shape=(n, width))
    t0, empty, obs_months = time.time(), 0, []
    chunk = args.batch_size * 16
    with torch.inference_mode():
        for a in range(0, n, chunk):
            z = min(a + chunk, n)
            f = fidx[a:z]
            raw = np.asarray(cache[a:z])[:, cols.ravel()].reshape(z - a, T, len(BANDS))
            ok = np.isfinite(raw).all(-1) & exists[f] & (pos[f] >= 0)
            vals = np.where(ok[..., None], raw, 0).astype(np.float32) / 1e4
            w = onehot[f] * ok[..., None]                                   # (b, slot, month)
            count = w.sum(1)                                                # (b, month)
            comp = np.einsum("bsm,bsc->bmc", w, vals) / np.maximum(count, 1)[..., None]
            observed = count > 0
            tot = comp[..., B8] + comp[..., B4]
            x = np.zeros((z - a, MAX_MONTHS, NUM_CHANNELS), np.float32)
            x[..., S2_SLICE] = comp
            x[..., NDVI] = np.where(tot > 0, (comp[..., B8] - comp[..., B4]) / np.where(tot > 0, tot, 1), 0)
            mask = np.ones_like(x)
            mask[..., S2_SLICE] = 0
            mask[..., NDVI] = 0
            mask[~observed] = 1
            k_obs = observed.sum(1)
            obs_months.append(k_obs)
            emb = np.full((z - a, width), np.nan, np.float32)
            empty += int((k_obs == 0).sum())
            for k in np.unique(k_obs[k_obs > 0]):
                rows = np.flatnonzero(k_obs == k)
                for b0 in range(0, len(rows), args.batch_size):
                    idx = rows[b0:b0 + args.batch_size]
                    e = encoder(
                        torch.from_numpy(x[idx]).to(device),
                        torch.full((len(idx), MAX_MONTHS), DW_MISSING, dtype=torch.long, device=device),
                        torch.zeros((len(idx), 2), device=device),
                        torch.from_numpy(mask[idx]).to(device),
                        torch.from_numpy(month_tok[f[idx]]).to(device),
                        eval_task=True,
                    ).float().cpu().numpy()
                    if not np.isfinite(e).all():
                        raise RuntimeError(f"Non-finite Presto output in rows {a}:{z}")
                    emb[idx] = e
            out[a:z] = emb
            rate = z / max(time.time() - t0, 1e-6)
            print(f"{z}/{n} pixels, {rate:.0f} px/s, ETA {(n - z) / rate / 60:.1f} min", flush=True)
    out.flush()
    del out
    (out_dir / f".{args.subset}.npy.tmp").rename(target)
    obs_months = np.concatenate(obs_months)
    meta_info = {"subset": args.subset, "kind": "dense", "rows": int(n),
                 "feature_names": [f"presto_{i:03d}" for i in range(width)],
                 "aligned_with": f"v1/{args.subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
                 "model": "Presto (default_model.pt)", "source": "https://github.com/nasaharvest/presto",
                 "source_revision": SOURCE_REVISION, "weights_sha256": WEIGHTS_SHA256,
                 "design": {"temporal": "in-season", "spatial": "pixel"},
                 "input": "S2 10 bands + NDVI; the field's own season: per-pixel mean composite per calendar month "
                          "over the field's span of slot dates (file time reference + slot offset), months without "
                          "a valid observation masked, calendar month tokens; S1/ERA5/SRTM/Dynamic World masked; "
                          "lat/lon fixed at (0, 0)",
                 "normalization": "S2 reflectance / 10000, NDVI from the composite (Presto upstream convention)",
                 "pooling": "Presto encoder eval_task=True (mean of tokens incl. location token, then LayerNorm)",
                 "pixels_without_s2": empty,
                 "observed_months": {"median": float(np.median(obs_months)), "max": int(obs_months.max())},
                 "month_span": {"median": float(np.median(spans)), "max": int(spans.max())},
                 "date_check": date_check, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    print(json.dumps({k: meta_info[k] for k in ("subset", "rows", "pixels_without_s2", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
