"""Frozen Galileo-Base embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

Input per pixel: its 24-slot Sentinel-2 series (the 10 bands Galileo knows, plus NDVI computed from B8 and B4) as
a 1x1 pixel time series, with each slot's calendar month as Galileo's month token. YieldSAT stores each slot as a
day offset from the field's seeding date, so a slot's month is month(seeding_date + offset). Slots without an S2
observation are masked. Weather, topography and location are deliberately left out, so "galileo" is comparable to
the raw `s2` input and location only enters through the location encoders.

    python -m tools.embed_yieldsat_galileo --subset GER-W     (GPU job; one subset per array task)

Writes features/yieldsat/galileo/<subset>.npy  float16 (pixels, 768), rows aligned with v1/<subset>.npy, plus
<subset>.json. Pixels with no S2 observation at all get NaN (the runner imputes them).
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from tools import yieldsat as ys
from tools.build_yieldsat_inputs import code_labels

SOURCE = ys.PROJECT / "upstream" / "galileo"
SOURCE_REVISION = "0f0b5b95ac81acef4b74cf4686dc877202f4541b"
WEIGHTS = ys.PROJECT / "checkpoints" / "galileo" / "base"
OURS = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05", "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
        "B11": "B11", "B12": "B12"}
MAX_SLOT_DATE_ERROR_DAYS = 3  # observed slots must sit on a Raw S2 acquisition date (checked on sample fields)


def slot_months(subset, fields):
    """(fields, 24) month index 0-11 per slot and (fields, 24) bool 'slot exists', from the preprocessed file."""
    country = next(c for c, code in ys.COUNTRIES.items() if code == subset.split("-")[0])
    with h5py.File(ys.preprocessed_file(country), "r") as h:
        code = h["field_shared_name"][:]
        labels = code_labels(h, "field_shared_name")
        seeding = code_labels(h, "seeding_date")
        starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
        stops = np.r_[starts[1:], len(code)]
        rows = {labels[int(code[a])]: (int(a), int(b)) for a, b in zip(starts, stops)}
        months = np.zeros((len(fields), ys.N_STEPS), np.int64)
        exists = np.zeros((len(fields), ys.N_STEPS), bool)
        dates = {}
        for i, f in enumerate(fields):
            a, b = rows[f]
            t_first, t_last = h["times"][a], h["times"][b - 1]
            if not np.array_equal(t_first, t_last, equal_nan=True):
                raise ValueError(f"{f}: slot offsets differ between pixels of one field")
            seed = dt.date.fromisoformat(seeding[int(h["seeding_date"][a])])
            ok = np.isfinite(t_first)
            d = [seed + dt.timedelta(days=float(t)) if o else None for t, o in zip(t_first, ok)]
            months[i] = [x.month - 1 if x else 0 for x in d]
            exists[i] = ok
            dates[f] = d
    return months, exists, dates, country


def check_dates(subset, fields, dates, country, cache, start, s2_cols, n=25):
    """Slots holding an S2 value should fall on a Raw S2 acquisition date; otherwise the months are wrong."""
    errors = []
    for i in np.linspace(0, len(fields) - 1, min(n, len(fields))).astype(int):
        f = fields[i]
        raw = sorted(ys.DATA.glob(f"Raw/{country}/{country}/{f}/s2_images/S2_L2A_*.tif"))
        acq = np.array([dt.date.fromisoformat(p.stem[-8:-4] + "-" + p.stem[-4:-2] + "-" + p.stem[-2:]).toordinal()
                        for p in raw])
        row = np.asarray(cache[start[i]])
        for s in range(ys.N_STEPS):
            if dates[f][s] is None or not np.isfinite(row[s2_cols[0][s]]):
                continue
            errors.append(int(np.min(np.abs(acq - dates[f][s].toordinal()))) if len(acq) else 999)
    errors = np.array(errors)
    if len(errors) == 0 or np.median(errors) > MAX_SLOT_DATE_ERROR_DAYS:
        raise ValueError(f"{subset}: S2 slot dates don't match Raw acquisitions (median error "
                         f"{np.median(errors) if len(errors) else 'n/a'} days); the month tokens would be wrong")
    return {"slots_checked": int(len(errors)), "median_days": float(np.median(errors)),
            "share_exact": float(np.mean(errors == 0)), "max_days": int(errors.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "galileo"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    if not torch.cuda.is_available():
        raise SystemExit("Run this in a GPU job")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    sys.path.insert(0, str(SOURCE))
    import single_file_galileo as galileo

    bands = list(OURS)
    if galileo.S2_BANDS != bands:
        raise ValueError("Galileo's S2 band list changed")
    st_bands = galileo.SPACE_TIME_BANDS
    idx = [st_bands.index(b) for b in bands] + [st_bands.index("NDVI")]
    groups = [i for i, g in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX) if g.startswith("S2_") or g == "NDVI"]
    stats = json.loads((SOURCE / "config" / "normalization.json").read_text())[str(len(st_bands))]
    mean = np.asarray(stats["mean"], np.float32)[idx]
    std = np.asarray(stats["std"], np.float32)[idx]
    shift, scale = mean - 2 * std, 4 * std

    info = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())
    names = info["feature_names"]
    s2_cols = [[names.index(f"{OURS[b]}_t{s:02d}") for s in range(ys.N_STEPS)] for b in bands]
    cols = np.array(s2_cols).T  # (24, 10): slot-major
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    start = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")

    months, exists, dates, country = slot_months(args.subset, fields)
    date_check = check_dates(args.subset, fields, dates, country, cache, start, s2_cols)
    print(json.dumps({"subset": args.subset, "date_check": date_check}), flush=True)

    encoder = galileo.Encoder.load_from_folder(WEIGHTS, torch.device("cpu")).to(device).eval()
    n, T = len(fidx), ys.N_STEPS
    out = np.lib.format.open_memmap(out_dir / f".{args.subset}.npy.tmp", mode="w+", dtype=np.float16,
                                    shape=(n, 768))
    t0, empty = time.time(), 0
    with torch.inference_mode():
        for a in range(0, n, args.batch_size):
            z = min(a + args.batch_size, n)
            raw = np.asarray(cache[a:z])[:, cols.ravel()].reshape(z - a, T, len(bands))
            observed = np.isfinite(raw).all(-1) & exists[fidx[a:z]]
            b4, b8 = raw[..., bands.index("B4")], raw[..., bands.index("B8")]
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (b8 - b4) / (b8 + b4)
            x = np.concatenate([raw, ndvi[..., None]], -1)
            x = np.where(observed[..., None], (x - shift) / scale, 0).astype(np.float32)
            has = observed.any(1)
            empty += int((~has).sum())
            emb = np.full((z - a, 768), np.nan, np.float32)
            if has.any():
                x, observed_h = x[has], observed[has]
                b = len(x)
                s_t = torch.zeros((b, 1, 1, T, len(st_bands)), device=device)
                s_t[..., idx] = torch.as_tensor(x[:, None, None], device=device)
                s_t_m = torch.ones((b, 1, 1, T, len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)), device=device)
                miss = torch.as_tensor(~observed_h, dtype=s_t_m.dtype, device=device)
                for g in groups:
                    s_t_m[:, 0, 0, :, g] = miss
                encoded = encoder(
                    s_t,
                    torch.zeros((b, 1, 1, len(galileo.SPACE_BANDS)), device=device),
                    torch.zeros((b, T, len(galileo.TIME_BANDS)), device=device),
                    torch.zeros((b, len(galileo.STATIC_BANDS)), device=device),
                    s_t_m,
                    torch.ones((b, 1, 1, len(galileo.SPACE_BAND_GROUPS_IDX)), device=device),
                    torch.ones((b, T, len(galileo.TIME_BAND_GROUPS_IDX)), device=device),
                    torch.ones((b, len(galileo.STATIC_BAND_GROUPS_IDX)), device=device),
                    torch.as_tensor(months[fidx[a:z]][has], device=device),
                    patch_size=1,
                )
                e = encoder.average_tokens(*encoded[:8]).float().cpu().numpy()
                if not np.isfinite(e).all():
                    raise RuntimeError(f"Non-finite Galileo output in rows {a}:{z}")
                emb[has] = e
            out[a:z] = emb
            if (a // args.batch_size) % 100 == 0:
                rate = z / max(time.time() - t0, 1e-6)
                print(f"{z}/{n} pixels, {rate:.0f} px/s, ETA {(n - z) / rate / 60:.1f} min", flush=True)
    out.flush()
    del out
    (out_dir / f".{args.subset}.npy.tmp").rename(target)
    meta_info = {"subset": args.subset, "kind": "dense", "rows": int(n),
                 "feature_names": [f"galileo_{i:03d}" for i in range(768)],
                 "pixels_without_s2": empty, "seconds": round(time.time() - t0, 1),
                 "pixels_per_second": round(n / (time.time() - t0), 1), "date_check": date_check,
                 "aligned_with": f"v1/{args.subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
                 "input": "S2 10 bands + NDVI, 24 slots, months from seeding date + slot offset, 1x1 pixel",
                 "normalization": "(x - (mean - 2 std)) / (4 std), Galileo upstream statistics",
                 "pooling": "Galileo average_tokens", "source_revision": SOURCE_REVISION,
                 "weights_sha256": {p.name: ys.sha256(p) for p in sorted(WEIGHTS.iterdir())}}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    print(json.dumps({k: meta_info[k] for k in ("subset", "rows", "pixels_without_s2", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
