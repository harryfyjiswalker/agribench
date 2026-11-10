"""Shared driver for single-date patch foundation models on YieldSAT (protocols/yieldsat.md, 2.5).

Two steps per subset:

1. Dates (CPU job, once per subset, shared by every model):
       python -m tools.patch_embed --subset GER-W
   writes features/yieldsat/patch-dates/<subset>.json: per field the label-free peak-greenness date and the
   green-up / senescence dates of tools/yieldsat_patches.py, with each date's clear share. This file holds field
   names, so it stays on the cluster (never commit it).

2. Embeddings: each tools/embed_yieldsat_<model>.py defines a `Model` (input size, token size, a `prepare` that
   turns a field's (12, H, W) Raw S2 scene into the model's normalised input, and an `encode` that returns the
   patch-token grid) and calls `run`. Per field: the peak date's scene -> NaN filled with the band's scene median ->
   `prepare` -> S x S windows (yieldsat_patches.windows, reflection padding, native 10 m) -> tokens -> each labelled
   pixel gets the bilinear sample of its window's token grid at the pixel centre.

Output: features/yieldsat/<name>/<subset>.npy float32 (pixels, D), rows aligned with v1/<subset>-meta.npz, plus
<subset>.json (feature_names, kind dense, aligned_with, v1_meta_sha256, model, revision, settings).
"""
import argparse
import datetime as dt
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field as dfield
from typing import Callable

import numpy as np

from tools import yieldsat as ys
from tools import yieldsat_patches as yp

DATES = ys.FEATURES / "patch-dates"
DATE_RULE = (f"peak greenness: the in-season (sowing..harvest) Raw S2 acquisition with the highest median EVI2 over "
             f"the field's labelled pixels, among acquisitions whose SCL is clear on >= {yp.MIN_CLEAR:.0%} of them "
             f"(fallback {yp.FALLBACK_CLEAR:.0%}, then the clearest); label-free")

# SSL4EO-S12 Sentinel-2 statistics (DN) as used by Copernicus-Bench for every S2 model, in RAW_S2_BANDS order
# (B10 is not in L2A). Source: upstream/copernicus-fm/src/Copernicus-Bench/src/configs/dataset/cobench_eurosat_s2.yaml
SSL4EO_MEAN = np.array([1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2, 2599.7, 732.1, 1820.6,
                        1118.2], np.float32)
SSL4EO_STD = np.array([897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5, 1750.1, 475.1, 1216.5, 736.7],
                      np.float32)
SSL4EO_B10 = (12.1, 98.3)  # cirrus band mean / std, for models that need all 13 L1C bands
# Centre wavelength and bandwidth (nm) per RAW_S2_BANDS band (Copernicus-Bench configs).
WAVELENGTH_NM = [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1610, 2190]
BANDWIDTH_NM = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]


def zscore(img):
    return (img - SSL4EO_MEAN[:, None, None]) / SSL4EO_STD[:, None, None]


def minmax2std(img, mean=SSL4EO_MEAN, std=SSL4EO_STD):
    """(x - (mean - 2 std)) / (4 std), clipped to [0, 1] (SoftCon / CROMA recipe with fixed dataset statistics)."""
    lo = (mean - 2 * std)[:, None, None]
    return np.clip((img - lo) / (4 * std)[:, None, None], 0, 1)


# ---- step 1: dates -----------------------------------------------------------------------------------------------

def _field_dates(f):
    curve = yp.greenness_curve(f)
    up, peak, down = yp.phenology_dates(f, curve)
    clear = {c[0]: c[1] for c in curve}
    return f.name, {"peak": peak.isoformat(), "greenup": up.isoformat(), "senescence": down.isoformat(),
                    "peak_clear": round(clear[peak], 4), "acquisitions_in_season": len(curve),
                    "peak_days_after_sowing": (peak - f.sowing).days}


def build_dates(subset, workers):
    DATES.mkdir(parents=True, exist_ok=True)
    target = DATES / f"{subset}.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    t0 = time.time()
    fs = list(yp.fields(subset))
    with ProcessPoolExecutor(workers) as pool:
        out = dict(pool.map(_field_dates, fs, chunksize=4))
    clear = np.array([v["peak_clear"] for v in out.values()])
    info = {"subset": subset, "rule": DATE_RULE, "fields": out,
            "summary": {"fields": len(out), "share_peak_clear_ge_min": float(np.mean(clear >= yp.MIN_CLEAR)),
                        "median_peak_days_after_sowing": float(np.median([v["peak_days_after_sowing"]
                                                                          for v in out.values()]))},
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    tmp = DATES / f".{subset}.json.tmp"
    tmp.write_text(json.dumps(info, indent=1) + "\n")
    tmp.rename(target)
    print(json.dumps(info["summary"]), flush=True)


def load_dates(subset):
    path = DATES / f"{subset}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python -m tools.patch_embed --subset {subset}` first")
    info = json.loads(path.read_text())
    if info["v1_meta_sha256"] != ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz"):
        raise ValueError(f"{path} was built for another v1 cache")
    return info


# ---- step 2: embeddings ------------------------------------------------------------------------------------------

@dataclass
class Model:
    name: str                       # cache folder under features/yieldsat/
    dim: int                        # embedding width
    size: int                       # model input size S (pixels)
    token: int                      # token (patch) size in pixels
    prepare: Callable               # (12, H, W) DN, NaN filled -> (C, H, W) float32 model input
    encode: Callable                # (B, C, S, S) torch tensor on device, list of dates -> (B, D, S/token, S/token)
    description: dict = dfield(default_factory=dict)   # model, revision, weights sha256, settings for the json


def _load_field(f, date, prepare, size):
    img = prepare(yp.fill_nan(f.s2(date)))
    wins = yp.windows(*img.shape[1:], size)
    k = yp.assign(f.row, f.col, wins, size)
    crops = np.stack([yp.crop(img, r0, c0, size) for r0, c0 in wins]).astype(np.float32)
    return f, date, wins, k, crops


def run(model, subset, batch_windows=32, workers=6):
    import torch

    out_dir = ys.FEATURES / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dates = load_dates(subset)
    fs = list(yp.fields(subset))
    n = fs[-1].rows.stop
    tmp = out_dir / f".{subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, model.dim))
    filled = np.zeros(n, bool)
    t0, n_windows = time.time(), 0
    pending = []  # (field, date, window index, crop) waiting for a batch

    def flush():
        nonlocal pending, n_windows
        if not pending:
            return
        x = torch.as_tensor(np.stack([p[3] for p in pending]), device=device)
        with torch.inference_mode():
            tok = model.encode(x, [p[1] for p in pending]).float()
        g = model.size // model.token
        if tok.shape[1:] != (model.dim, g, g):
            raise ValueError(f"encode returned {tuple(tok.shape)}, expected (B, {model.dim}, {g}, {g})")
        if not torch.isfinite(tok).all():
            raise RuntimeError("non-finite tokens")
        for (f, _, (r0, c0, sel), _), t in zip(pending, tok):
            rows = np.arange(f.rows.start, f.rows.stop)[sel]
            out[rows] = yp.sample_tokens(t, f.row[sel], f.col[sel], r0, c0, model.size)
            filled[rows] = True
        n_windows += len(pending)
        pending = []

    with ThreadPoolExecutor(workers) as pool:
        jobs = pool.map(lambda f: _load_field(f, dt.date.fromisoformat(dates["fields"][f.name]["peak"]),
                                              model.prepare, model.size), fs)
        for i, (f, date, wins, k, crops) in enumerate(jobs):
            for j, (r0, c0) in enumerate(wins):
                sel = k == j
                if sel.any():
                    pending.append((f, date, (r0, c0, sel), crops[j]))
                if len(pending) >= batch_windows:
                    flush()
            if i % 50 == 0:
                print(f"{i}/{len(fs)} fields, {n_windows} windows, {time.time() - t0:.0f}s", flush=True)
        flush()
    out.flush()
    del out
    if not filled.all():
        raise RuntimeError(f"{(~filled).sum()} pixels got no embedding")
    tmp.rename(target)
    info = {"subset": subset, "kind": "dense", "rows": int(n),
            "feature_names": [f"{model.name}_{i:04d}" for i in range(model.dim)],
            "aligned_with": f"v1/{subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{subset}-meta.npz"),
            **model.description,
            "settings": {**model.description.get("settings", {}), "input_size": model.size,
                         "token_size_px": model.token, "date_rule": DATE_RULE,
                         "windows": "S x S at native 10 m, one centred window per field box <= S, else 50% "
                                    "overlapping windows; reflection padding; pixel -> nearest window centre",
                         "pixel_embedding": "bilinear sample of the last-layer token grid at the pixel centre",
                         "nan_fill": "band median of the scene"},
            "patch_dates_summary": dates["summary"], "windows": n_windows,
            "seconds": round(time.time() - t0, 1), "device": str(device)}
    (out_dir / f"{subset}.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "windows", "seconds")}), flush=True)


def weights_sha(path):
    return {path.name: ys.sha256(path)}


def load_state(path):
    """Local checkpoint, tensors only."""
    import torch
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def capture_output(module):
    """Forward hook storage: the full token sequence that `module` (final norm or last block) outputs."""
    store = {}
    module.register_forward_hook(lambda m, i, o: store.__setitem__("x", o))
    return store


def tokens_to_grid(x, g):
    """(B, g*g, D) -> (B, D, g, g)."""
    b, n, d = x.shape
    if n != g * g:
        raise ValueError(f"{n} tokens is not a {g}x{g} grid")
    return x.transpose(1, 2).reshape(b, d, g, g)


def cli(make_model):
    """Standard command line of an embed_yieldsat_<model>.py script."""
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=ys.SUBSETS, required=True)
    p.add_argument("--batch-windows", type=int, default=32)
    a = p.parse_args()
    run(make_model(), a.subset, a.batch_windows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=ys.SUBSETS, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    build_dates(a.subset, a.workers)
