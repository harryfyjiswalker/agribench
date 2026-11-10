"""Frozen OlmoEarth-v1.2-Base embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

In-season pixel input: each pixel's own Sentinel-2 acquisitions from the 24 slots of the preprocessed file (one
field's season; 1, note 5), each at its real date (file time reference + slot offset, as for `galileo`; OlmoEarth's
temporal RoPE uses the actual day). The pixel is a 1 x 1 image with patch size 1, which the upstream inference guide
supports (inputs 1x1 to 128x128, patch size 1-8). All 12 L2A bands, upstream "computed" normalization. Only
Sentinel-2 is given; no location or other modality.

OlmoEarth takes at most 12 timesteps (max_sequence_length). Pixels with more valid observations (0.8% of ARG-C,
none elsewhere) have their two closest-in-time observations averaged until 12 remain. Missing slots are not passed
as masked tokens: the encoder's inference path attends to every token it is given, so only valid observations go in,
and pixels are batched by their number of observations. Embedding = mean of the pixel's output tokens over time.

    python -m tools.embed_yieldsat_olmoearth --subset GER-W     (GPU job; benchmark-v2 env, the pinned source is
                                                                 put on sys.path rather than installed)

Writes features/yieldsat/olmoearth/<subset>.npy float32 (pixels, 768), rows aligned with v1/<subset>.npy, plus
<subset>.json. Pixels with no valid S2 observation get NaN (the runner imputes them).
"""
import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import types

import numpy as np
import torch

from tools import yieldsat as ys
from tools.embed_yieldsat_galileo import check_dates, slot_months

SOURCE = ys.PROJECT / "upstream" / "olmoearth" / "olmoearth_pretrain"
SOURCE_REVISION = "13ca06ffaf0247bd49b8a4fccf705a3a85fe9d6d"
HF_REPO, HF_REVISION = "allenai/OlmoEarth-v1_2-Base", "2e99a734a30e9aeb993aaa39946c6dcf554739e0"
WEIGHTS = ys.PROJECT / "upstream" / "olmoearth" / f"OlmoEarth-v1_2-Base@{HF_REVISION}"
MAX_T = 12


def load_model():
    # model_loader imports universal-pathlib only for remote paths; a local checkpoint needs plain pathlib.
    sys.modules.setdefault("upath", types.SimpleNamespace(UPath=pathlib.Path))
    sys.path.insert(0, str(SOURCE))
    from olmoearth_pretrain.config import Config
    from olmoearth_pretrain.model_loader import patch_legacy_encoder_config

    config = patch_legacy_encoder_config(json.loads((WEIGHTS / "config.json").read_text()))
    model = Config.from_dict(config["model"]).build()
    model.load_state_dict(torch.load(WEIGHTS / "weights.pth", map_location="cpu", weights_only=True))
    if config["model"]["encoder_config"]["max_sequence_length"] != MAX_T:
        raise ValueError("Unexpected OlmoEarth max_sequence_length")
    return model.eval()


def merge_to(values, days, k):
    """values (n, bands), days (n,) sorted: average the closest adjacent pair until k observations remain."""
    values, days = [v for v in values], list(days)
    while len(days) > k:
        i = int(np.argmin(np.diff(days)))
        values[i:i + 2] = [(values[i] + values[i + 1]) / 2]
        days[i:i + 2] = [(days[i] + days[i + 1]) / 2]
    return np.array(values), np.array(days)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    out_dir = ys.FEATURES / "olmoearth"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    if not torch.cuda.is_available():
        raise SystemExit("Run this in a GPU job")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    model = load_model()
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    encoder = model.encoder.to(device)
    bands = list(Modality.SENTINEL2_L2A.band_order)
    n_sets = len(Modality.SENTINEL2_L2A.band_sets)
    normalizer = Normalizer(Strategy.COMPUTED)

    info = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())
    names = info["feature_names"]
    s2_cols = [[names.index(f"{b}_t{s:02d}") for s in range(ys.N_STEPS)] for b in bands]
    cols = np.array(s2_cols).T  # (24, 12) slot-major
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    start = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    _, exists, dates, country = slot_months(args.subset, fields)
    date_check = check_dates(args.subset, fields, dates, country, cache, start, s2_cols)
    ordinal = np.array([[d.toordinal() if d else 0 for d in dates[f]] for f in fields], np.float64)
    print(json.dumps({"subset": args.subset, "date_check": date_check, "bands": bands}), flush=True)

    def stamp(day_ordinal):
        """(..., ) float ordinal days -> (..., 3) long (day 1-31, month 0-11, year)."""
        flat = [dt.date.fromordinal(int(round(o))) for o in np.ravel(day_ordinal)]
        return np.array([[d.day, d.month - 1, d.year] for d in flat], np.int64).reshape(*np.shape(day_ordinal), 3)

    n, T = len(fidx), ys.N_STEPS
    width = None
    out = None
    t0, empty, merged, counts_all = time.time(), 0, 0, []
    chunk = args.batch_size * 16
    with torch.inference_mode():
        for a in range(0, n, chunk):
            z = min(a + chunk, n)
            f = fidx[a:z]
            raw = np.asarray(cache[a:z])[:, cols.ravel()].reshape(z - a, T, len(bands))
            ok = np.isfinite(raw).all(-1) & exists[f]
            k = ok.sum(1)
            empty += int((k == 0).sum())
            # Pack each pixel's valid observations to the front, in slot (= date) order.
            order = np.argsort(~ok, axis=1, kind="stable")
            vals = np.take_along_axis(raw, order[..., None], 1)[:, :MAX_T].astype(np.float32)
            days = np.take_along_axis(ordinal[f], order, 1)[:, :MAX_T]
            for i in np.flatnonzero(k > MAX_T):
                s = np.flatnonzero(ok[i])
                v, d = merge_to(raw[i, s], ordinal[f[i], s], MAX_T)
                vals[i], days[i] = v, d
                merged += 1
            k = np.minimum(k, MAX_T)
            counts_all.append(k)
            emb = None
            for kk in np.unique(k[k > 0]):
                rows = np.flatnonzero(k == kk)
                for b0 in range(0, len(rows), args.batch_size):
                    idx = rows[b0:b0 + args.batch_size]
                    x = normalizer.normalize(Modality.SENTINEL2_L2A, vals[idx, :kk])  # (b, kk, bands)
                    x = torch.as_tensor(np.asarray(x, np.float32), device=device)[:, None, None]  # b,1,1,t,c
                    sample = MaskedOlmoEarthSample(
                        sentinel2_l2a=x,
                        sentinel2_l2a_mask=torch.full((len(idx), 1, 1, int(kk), n_sets),
                                                      float(MaskValue.ONLINE_ENCODER.value), device=device),
                        timestamps=torch.as_tensor(stamp(days[idx, :kk]), device=device),
                    )
                    tokens = encoder(sample, fast_pass=True, patch_size=1)["tokens_and_masks"].sentinel2_l2a
                    e = tokens.mean(dim=(1, 2, 3, 4)).float().cpu().numpy()  # (b,1,1,t,s,d) -> (b,d)
                    if not np.isfinite(e).all():
                        raise RuntimeError(f"Non-finite OlmoEarth output in rows {a}:{z}")
                    if out is None:
                        width = e.shape[1]
                        out = np.lib.format.open_memmap(out_dir / f".{args.subset}.npy.tmp", mode="w+",
                                                        dtype=np.float32, shape=(n, width))
                    if emb is None:
                        emb = np.full((z - a, width), np.nan, np.float32)
                    emb[idx] = e
            out[a:z] = emb if emb is not None else np.nan
            rate = z / max(time.time() - t0, 1e-6)
            print(f"{z}/{n} pixels, {rate:.0f} px/s, ETA {(n - z) / rate / 60:.1f} min", flush=True)
    out.flush()
    del out
    (out_dir / f".{args.subset}.npy.tmp").rename(target)
    counts_all = np.concatenate(counts_all)
    meta_info = {"subset": args.subset, "kind": "dense", "rows": int(n),
                 "feature_names": [f"olmoearth_{i:03d}" for i in range(width)],
                 "aligned_with": f"v1/{args.subset}-meta.npz",
                 "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
                 "model": "OlmoEarth-v1.2-Base encoder", "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
                 "source": "https://github.com/allenai/olmoearth_pretrain", "source_revision": SOURCE_REVISION,
                 "weights_sha256": ys.sha256(WEIGHTS / "weights.pth"),
                 "design": {"temporal": "in-season", "spatial": "pixel"},
                 "input": "S2 L2A 12 bands; the pixel's own valid acquisitions from the field's season at their real "
                          "dates (file time reference + slot offset); 1x1 image, patch size 1; at most 12 timesteps "
                          "(closest pairs averaged beyond that); S2 only",
                 "normalization": "olmoearth_pretrain Normalizer(Strategy.COMPUTED)",
                 "pooling": "mean of the encoder's S2 output tokens over time (fast_pass, only valid timesteps given)",
                 "pixels_without_s2": empty, "pixels_merged_to_12": merged,
                 "timesteps": {"median": float(np.median(counts_all)), "max": int(counts_all.max())},
                 "date_check": date_check, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    print(json.dumps({k: meta_info[k] for k in ("subset", "rows", "pixels_without_s2", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
