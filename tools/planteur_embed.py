"""PLANTEUR inputs from the monthly Sentinel-2 mosaics, with the encoders and builders AgriBench uses for YieldSAT
(protocols/planteur.md 7). Runs as Slurm jobs, one array task per part of the 2,750 patches, then one combine step:

    python -m tools.planteur_embed --input dofa --submit             # array (20 parts) + combine job, CPU
    python -m tools.planteur_embed --input dofa --part 3 --parts 20  # one part (what each array task runs)
    python -m tools.planteur_embed --input dofa --combine --parts 20
    python -m tools.planteur_embed --input dofa --smoke              # 4 patches, prints shapes, writes nothing

Source: DATA_S2_MONTHLY (12 months x 12 bands, uint16, 0 = no data), month m dated at its mid-point
(2019-01-01 + 30 m + 15 days, MONTHLY/months.json). Nothing here reads a label, the void mask or a NORM_* file.

Two storage kinds (tools/planteur.py, run_planteur.Features):
  grid   encoder outputs: one token grid per patch, grid.npy (patches, g, g, D) float16 in meta.npz patch order, plus
         patch_valid.npy. info.json gives the pixel window the grid covers (origin o, span S: rows/cols [o, o + S)).
         A pixel's feature is the bilinear sample of its patch's grid at the pixel centre (border-clamped), the same
         rule as tools.yieldsat_patches.sample_tokens. This keeps 768-d inputs at < 1 GB instead of ~70 GB.
  dense  per-pixel tables (s2-harmonic, biophys) in the X.npy / valid.npy layout of tools.build_planteur_features.

Encoders:
  single-date patch models (copernicusfm, dofa, croma, softcon, clay, satmae, thor, terramind): each patch's
      peak-greenness month (highest median NDVI over the patch among months with >= 50% of pixels observed, else
      the best-covered month; label-free) as the 12 L2A bands in RAW_S2_BANDS order, NaN filled with the band
      median, through the YieldSAT wrapper's own `prepare` and `encode` (tools/embed_yieldsat_<model>.py) at its
      native window size. A window larger than 128 px is centred and reflect-padded (as YieldSAT pads small fields);
      a smaller one (CROMA 120, SatMAE 96) is tiled 2 x 2 at token-aligned offsets and overlapping tokens averaged.
      TerraMind takes S2 only (YieldSAT adds a DEM; PLANTEUR has none).
  galileo   Galileo base, the 12 months of the 10 DATA_S2 bands + NDVI, 16 x 16 px windows at patch size 4,
      average_tokens per window (protocol 7.1: the paper's patch 4 / window 16) -> 8 x 8 grid.
  presto    Presto, per pixel (patch size 1), 12 months, the patch centre's lat/lon, pooled over 16 x 16 px windows
      (protocol 7.1) -> 8 x 8 grid.
  olmoearth OlmoEarth-v1.2-Base on 16 x 16 px windows of the monthly series, patch size 4 -> 8 x 8 grid (differs from
      the shipped v1.3 128-d read-out).
  prithvi   Prithvi-EO-2.0-300M-TL, 4 label-free months (January, NDVI rise, peak, decline), 224 px window -> 14 x 14.
  anysat    AnySat-Base on the first 96 DATA_S2 observations of 2019 (the paper's input), patch 160 m -> 8 x 8.
  latlon / satclip / mind   lat/lon of the 8 x 8 grid cell centres (from ANNOTATIONS_GEOTIFF's CRS and transform),
      raw (float32) or through the SatCLIP / MIND location encoders of tools/embed_yieldsat_location.py.
  s2-harmonic   per pixel and band, c0 + 2 harmonics fitted to the 12 monthly values (t = day of year of the month
      mid-point, period 365 d), tools/build_yieldsat_s2_harmonic.fit -> 10 bands x 5 = 50 features.
  biophys   SNAP S2 biophysical processor (tools/build_yieldsat_biophys) per pixel and month: LAI, FAPAR, FCOVER,
      CCC, CWC x 12 months = 60 features; sun zenith at the patch centre's latitude, the month mid-point, 10:30.
"""
import argparse
import datetime as dt
import importlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tools import planteur as pl

MONTHLY = pl.DATA / "DATA_S2_MONTHLY"
MONTH_MID = [dt.date(2019, 1, 1) + dt.timedelta(days=30 * m + 15) for m in range(pl.N_MONTHS)]
MONTHLY_BANDS = pl.S2_BANDS + ["B01", "B09"]
RAW_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RAW_ORDER = [MONTHLY_BANDS.index(b) for b in RAW_S2_BANDS]
B04, B08 = MONTHLY_BANDS.index("B04"), MONTHLY_BANDS.index("B08")
MIN_COVER = 0.5
POOL = 16                 # protocol 7.1: 16 x 16 px windows for the time-series encoders
LOC_GRID = 8              # location inputs: 8 x 8 cells of 16 px (160 m)
PEAK_RULE = (f"peak greenness: the month with the highest median NDVI over all 128 x 128 pixels, among months with "
             f">= {MIN_COVER:.0%} of pixels observed (else the best-covered month); label-free")
PATCH_MODELS = ["copernicusfm", "dofa", "croma", "softcon", "clay", "satmae", "thor", "terramind"]
TIME_SERIES = ["galileo", "presto", "olmoearth", "prithvi", "anysat"]
GRID_INPUTS = PATCH_MODELS + TIME_SERIES + ["latlon", "satclip", "mind"]
DENSE_INPUTS = ["s2-harmonic", "biophys"]
PATCHFM_ENV = pl.PROJECT / "envs" / "patchfm" / "bin" / "python"
NEEDS_PATCHFM = {"clay", "satmae", "thor", "terramind"}
SLURM = "/cm/shared/apps/slurm/current/bin"


# ---- shared helpers ----------------------------------------------------------------------------------------------

def monthly(p):
    """(12 months, 12 bands in MONTHLY_BANDS order, 128, 128) float32 DN, NaN = no data."""
    a = np.load(MONTHLY / f"S2_{p}.npy")
    if a.shape != (pl.N_MONTHS, 12, pl.SIZE, pl.SIZE) or a.dtype != np.uint16:
        raise ValueError(f"S2_{p}.npy: unexpected {a.shape} {a.dtype}")
    a = a.astype(np.float32)
    a[a == 0] = np.nan
    return a


def ndvi(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a[:, B08] - a[:, B04]) / (a[:, B08] + a[:, B04])


def peak_month(a):
    """(month, its observed share) by PEAK_RULE."""
    v = ndvi(a)
    cover = np.isfinite(v).mean((1, 2))
    ok = cover >= MIN_COVER
    if not ok.any():
        ok = cover == cover.max()
    med = np.array([np.nanmedian(v[m]) if cover[m] > 0 else -np.inf for m in range(len(v))])
    med[~ok] = -np.inf
    m = int(np.argmax(med))  # ties -> earliest
    return m, float(cover[m])


def fill_nan(img):
    """Band-wise median fill (0 if a band is all NaN), as tools.yieldsat_patches.fill_nan."""
    img = img.copy()
    for b in range(img.shape[0]):
        bad = ~np.isfinite(img[b])
        if bad.any():
            img[b][bad] = np.nanmedian(img[b]) if (~bad).any() else 0.0
    return img


def latlon_grid(p, g):
    """(g, g, 2) lat, lon (degrees) of the centres of a g x g grid over the patch."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(pl.DATA / "ANNOTATIONS_GEOTIFF" / f"TARGET_{p}.tif") as src:
        crs, t = src.crs, src.transform
        if (src.height, src.width) != (pl.SIZE, pl.SIZE):
            raise ValueError(f"TARGET_{p}.tif is {src.height} x {src.width}")
    centre = (np.arange(g) + 0.5) * pl.SIZE / g
    cc, rr = np.meshgrid(centre, centre)
    x, y = t * (cc.ravel(), rr.ravel())
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.stack([lat, lon], -1).reshape(g, g, 2)


def part_ids(meta, part, parts):
    b = np.linspace(0, len(meta.patch_ids), parts + 1).astype(int)
    return [int(p) for p in meta.patch_ids[b[part]:b[part + 1]]], int(b[part])


def code_commit():
    """The frozen copy's COMMIT, else git HEAD (marked -dirty if the checkout has changes)."""
    root = Path(__file__).resolve().parents[1]
    if (root / "COMMIT").exists():
        return (root / "COMMIT").read_text().strip()
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- encoders: each returns (grid (n, g, g, D) float32, valid (n,), per-patch notes, description) ----------------

class Grid:
    """Accumulates per-patch grids for one part."""

    def __init__(self):
        self.grids, self.valid, self.notes = [], [], []

    def add(self, grid, valid=True, **notes):
        self.grids.append(np.asarray(grid, np.float32))
        self.valid.append(bool(valid))
        self.notes.append(notes)


def patch_model(name):
    if name == "terramind":
        return terramind_s2()
    return importlib.import_module(f"tools.embed_yieldsat_{name}").make_model()


def terramind_s2():
    """TerraMind-1.0-base on S2 L2A only (the YieldSAT wrapper adds the field DEM, which PLANTEUR doesn't have)."""
    import torch
    from terratorch.models.backbones.terramind.model import terramind_register as tr

    from tools import embed_yieldsat_terramind as tm
    from tools import patch_embed as pe
    mean = np.array(tr.v1_pretraining_mean["untok_sen2l2a@224"], np.float32)
    std = np.array(tr.v1_pretraining_std["untok_sen2l2a@224"], np.float32)
    net = tr.terramind_v1_base(modalities=["S2L2A"], merge_method="mean", pretrained=False)
    net.load_state_dict(tr.checkpoint_filter_fn(pe.load_state(tm.WEIGHTS), net), strict=True)
    net = net.to(device()).eval()

    def encode(x, dates):
        return pe.tokens_to_grid(net({"S2L2A": x})[-1], tm.SIZE // tm.TOKEN)

    return pe.Model("terramind", 768, tm.SIZE, tm.TOKEN,
                    lambda img: ((img - mean[:, None, None]) / std[:, None, None]).astype(np.float32), encode, {
                        "model": "TerraMind-1.0-base encoder (terratorch terramind_v1_base), S2L2A only",
                        "revision": {"weights": tm.HF_REVISION, "terratorch": tm.TERRATORCH_COMMIT,
                                     "weights_sha256": pe.weights_sha(tm.WEIGHTS)},
                        "settings": {"modalities": "S2 L2A 12 bands; no DEM, S1 or coordinates",
                                     "normalization": "TerraMind v1 pretraining mean/std (untok_sen2l2a@224)",
                                     "layer": "last encoder layer after encoder_norm"}})


def patch_geometry(size, token):
    """(window offsets per axis, grid origin o, span S, grid size g) for a model with S x S input, t px tokens."""
    if size >= pl.SIZE:
        o = -((size - pl.SIZE) // 2)
        return [o], o, size, size // token
    if (pl.SIZE - size) % token:
        raise ValueError(f"{size} px windows with {token} px tokens can't tile 128 px on the token grid")
    return [0, pl.SIZE - size], 0, pl.SIZE, pl.SIZE // token


def run_patch_model(name, ids, batch=8):
    import torch
    from tools import yieldsat_patches as yp
    model = patch_model(name)
    offs, origin, span, g = patch_geometry(model.size, model.token)
    gw = model.size // model.token
    out = Grid()
    todo = []

    def load(p):
        a = monthly(p)
        m, cover = peak_month(a)
        x = model.prepare(fill_nan(a[m][RAW_ORDER]))
        return p, m, cover, [yp.crop(x, r, c, model.size) for r in offs for c in offs]

    def flush():
        if not todo:
            return
        x = torch.as_tensor(np.stack([w for _, _, _, ws in todo for w in ws]).astype(np.float32), device=device())
        dates = [MONTH_MID[m] for _, m, _, ws in todo for _ in ws]
        with torch.inference_mode():
            tok = model.encode(x, dates).float()
        if tok.shape[1:] != (model.dim, gw, gw) or not torch.isfinite(tok).all():
            raise RuntimeError(f"{name}: encode gave {tuple(tok.shape)} (finite: {bool(torch.isfinite(tok).all())})")
        tok = tok.cpu().numpy()
        k = 0
        for p, m, cover, ws in todo:
            acc, cnt = np.zeros((model.dim, g, g), np.float32), np.zeros((g, g), np.float32)
            for r in offs:
                for c in offs:
                    i, j = (r - origin) // model.token, (c - origin) // model.token
                    acc[:, i:i + gw, j:j + gw] += tok[k]
                    cnt[i:i + gw, j:j + gw] += 1
                    k += 1
            out.add((acc / cnt).transpose(1, 2, 0), month=m, cover=round(cover, 4))
        todo.clear()

    with ThreadPoolExecutor(4) as ex:
        for item in ex.map(load, ids):
            todo.append(item)
            if len(todo) * len(offs) ** 2 >= batch:
                flush()
    flush()
    desc = {**model.description, "date_rule": PEAK_RULE, "input_size": model.size, "token_size_px": model.token,
            "windows": ("one centred window, reflect-padded" if len(offs) == 1 else
                        f"2 x 2 windows at offsets {offs}, overlapping tokens averaged"),
            "pixel_embedding": "bilinear sample of the token grid at the pixel centre (border-clamped)"}
    return out, {"origin": origin, "span": span, "grid": g, "dim": model.dim, "names": f"{name}_", **desc}


def galileo_source():
    import sys
    from tools import embed_yieldsat_galileo as yg
    sys.path.insert(0, str(yg.SOURCE))
    import single_file_galileo as galileo
    return yg, galileo


def run_galileo(ids, batch=64):
    import torch
    yg, galileo = galileo_source()
    bands = list(yg.OURS)                      # B2..B12 = the 10 DATA_S2 bands, same order as MONTHLY_BANDS[:10]
    if galileo.S2_BANDS != bands or [yg.OURS[b] for b in bands] != pl.S2_BANDS:
        raise ValueError("Galileo's S2 band list changed")
    st = galileo.SPACE_TIME_BANDS
    idx = [st.index(b) for b in bands] + [st.index("NDVI")]
    groups = [i for i, gname in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX) if gname.startswith("S2_") or
              gname == "NDVI"]
    stats = json.loads((yg.SOURCE / "config" / "normalization.json").read_text())[str(len(st))]
    shift = (np.asarray(stats["mean"], np.float32)[idx] - 2 * np.asarray(stats["std"], np.float32)[idx])
    scale = 4 * np.asarray(stats["std"], np.float32)[idx]
    dev = device()
    enc = galileo.Encoder.load_from_folder(yg.WEIGHTS, torch.device("cpu")).to(dev).eval()
    T, w, nw = pl.N_MONTHS, POOL, pl.SIZE // POOL
    out = Grid()
    masked_months = []

    def windows(p):
        a = monthly(p)[:, :10]                                         # (T, 10, H, W)
        x = np.concatenate([a, ndvi(a)[:, None]], 1)                   # + NDVI (B04, B08 are among the 10)
        x = x.reshape(T, 11, nw, w, nw, w).transpose(2, 4, 3, 5, 0, 1).reshape(nw * nw, w, w, T, 11)
        obs = np.isfinite(x).all(-1).mean((1, 2)) >= MIN_COVER          # (windows, T): month usable in the window
        for k in range(len(x)):                                         # fill gaps with the window-month median
            for t in range(T):
                xt = x[k, :, :, t]
                bad = ~np.isfinite(xt)
                if bad.any():
                    med = np.nanmedian(xt.reshape(-1, 11), 0) if np.isfinite(xt).any() else np.zeros(11)
                    xt[bad] = np.broadcast_to(np.nan_to_num(med), xt.shape)[bad]
        x = np.where(obs[:, None, None, :, None], (x - shift) / scale, 0).astype(np.float32)
        return p, x, obs

    with ThreadPoolExecutor(4) as ex, torch.inference_mode():
        for p, x, obs in ex.map(windows, ids):
            emb = np.full((len(x), 768), np.nan, np.float32)
            has = obs.any(1)
            masked_months.append(int((~obs).sum()))
            for a in range(0, int(has.sum()), batch):
                sel = np.flatnonzero(has)[a:a + batch]
                b = len(sel)
                s_t = torch.zeros((b, w, w, T, len(st)), device=dev)
                s_t[..., idx] = torch.as_tensor(x[sel], device=dev)
                s_t_m = torch.ones((b, w, w, T, len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)), device=dev)
                miss = torch.as_tensor(~obs[sel], dtype=s_t_m.dtype, device=dev)[:, None, None, :]
                for gi in groups:
                    s_t_m[..., gi] = miss
                encoded = enc(
                    s_t, torch.zeros((b, w, w, len(galileo.SPACE_BANDS)), device=dev),
                    torch.zeros((b, T, len(galileo.TIME_BANDS)), device=dev),
                    torch.zeros((b, len(galileo.STATIC_BANDS)), device=dev), s_t_m,
                    torch.ones((b, w, w, len(galileo.SPACE_BAND_GROUPS_IDX)), device=dev),
                    torch.ones((b, T, len(galileo.TIME_BAND_GROUPS_IDX)), device=dev),
                    torch.ones((b, len(galileo.STATIC_BAND_GROUPS_IDX)), device=dev),
                    torch.arange(T, device=dev).repeat(b, 1), patch_size=4)
                e = enc.average_tokens(*encoded[:8]).float().cpu().numpy()
                if not np.isfinite(e).all():
                    raise RuntimeError(f"Non-finite Galileo output, patch {p}")
                emb[sel] = e
            if not has.all():  # a window with no usable month: the mean of the patch's other windows
                emb[~has] = np.nanmean(emb[has], 0) if has.any() else 0
            out.add(emb.reshape(nw, nw, 768), valid=bool(has.any()), windows_without_data=int((~has).sum()))
    return out, {"origin": 0, "span": pl.SIZE, "grid": nw, "dim": 768, "names": "galileo_",
                 "model": "Galileo base (single_file_galileo)", "source_revision": yg.SOURCE_REVISION,
                 "weights_sha256": {f.name: pl.sha256(f) for f in sorted(yg.WEIGHTS.iterdir())},
                 "input": "12 monthly mosaics, 10 DATA_S2 bands + NDVI, month tokens 0-11 (Jan-Dec 2019); S1, ERA5, "
                          "topography, location masked",
                 "window": f"{POOL} x {POOL} px, patch size 4 (4 x 4 tokens per month and group)",
                 "missing": f"a month is masked in a window unless >= {MIN_COVER:.0%} of its pixels are observed; "
                            "other gaps filled with the window-month band median",
                 "normalization": "(x - (mean - 2 std)) / (4 std), Galileo upstream statistics",
                 "pooling": "average_tokens per window -> one vector per 16 x 16 px",
                 "masked_window_months": int(sum(masked_months))}


def run_presto(ids, batch=8192):
    import sys

    import torch
    from tools import embed_yieldsat_presto as yp_
    if pl.sha256(yp_.WEIGHTS) != yp_.WEIGHTS_SHA256:
        raise ValueError(f"{yp_.WEIGHTS} is not the pinned Presto default_model.pt")
    if yp_.BANDS != pl.S2_BANDS:
        raise ValueError("Presto band order differs from DATA_S2")
    sys.path.insert(0, str(yp_.SOURCE))
    import single_file_presto as presto
    model = presto.Presto.construct()
    model.load_state_dict(torch.load(yp_.WEIGHTS, map_location="cpu", weights_only=True))
    dev = device()
    enc = model.encoder.to(dev).eval()
    width = enc.embedding_size
    T, nw = pl.N_MONTHS, pl.SIZE // POOL
    out = Grid()

    def load(p):
        a = monthly(p)[:, :10].reshape(T, 10, -1).transpose(2, 0, 1)    # (pixels, T, 10)
        return p, a, latlon_grid(p, 1)[0, 0]

    with ThreadPoolExecutor(2) as ex, torch.inference_mode():
        for p, a, ll in ex.map(load, ids):
            ok = np.isfinite(a).all(-1)                                  # (pixels, T)
            v = np.where(ok[..., None], a, 0) / 1e4
            x = np.zeros((len(a), T, yp_.NUM_CHANNELS), np.float32)
            x[..., yp_.S2_SLICE] = v
            tot = v[..., yp_.B8] + v[..., yp_.B4]
            x[..., yp_.NDVI] = np.where(tot > 0, (v[..., yp_.B8] - v[..., yp_.B4]) / np.where(tot > 0, tot, 1), 0)
            mask = np.ones_like(x)
            mask[..., yp_.S2_SLICE] = 0
            mask[..., yp_.NDVI] = 0
            mask[~ok] = 1
            k_obs = ok.sum(1)
            emb = np.full((len(a), width), np.nan, np.float32)
            for k in np.unique(k_obs[k_obs > 0]):  # equal numbers of unmasked tokens per batch
                rows = np.flatnonzero(k_obs == k)
                for b0 in range(0, len(rows), batch):
                    r = rows[b0:b0 + batch]
                    emb[r] = enc(torch.from_numpy(x[r]).to(dev),
                                 torch.full((len(r), T), yp_.DW_MISSING, dtype=torch.long, device=dev),
                                 torch.as_tensor(np.repeat(ll[None], len(r), 0), dtype=torch.float32, device=dev),
                                 torch.from_numpy(mask[r]).to(dev),
                                 torch.arange(T, device=dev).repeat(len(r), 1), eval_task=True).float().cpu().numpy()
            if not np.isfinite(emb[k_obs > 0]).all():
                raise RuntimeError(f"Non-finite Presto output, patch {p}")
            e = emb.reshape(nw, POOL, nw, POOL, width)
            with np.errstate(invalid="ignore"):
                cell = np.nanmean(e, (1, 3))                             # pixels without any observation left out
            empty = ~np.isfinite(cell).all(-1)
            if empty.any():
                cell[empty] = np.nanmean(cell[~empty], 0) if (~empty).any() else 0
            out.add(cell, valid=bool((k_obs > 0).any()), pixels_without_s2=int((k_obs == 0).sum()))
    return out, {"origin": 0, "span": pl.SIZE, "grid": nw, "dim": width, "names": "presto_",
                 "model": "Presto (default_model.pt)", "source_revision": yp_.SOURCE_REVISION,
                 "weights_sha256": yp_.WEIGHTS_SHA256,
                 "input": "per pixel: 12 monthly mosaics, 10 S2 bands / 10000 + NDVI, month tokens 0-11; months "
                          "without data masked; S1, ERA5, SRTM, Dynamic World masked; the patch centre's lat/lon",
                 "pooling": f"eval_task=True per pixel, then the mean over {POOL} x {POOL} px windows"}


def run_location(name, ids):
    from tools import embed_yieldsat_location as yl
    out = Grid()
    ll = np.stack([latlon_grid(p, LOC_GRID) for p in ids])               # (n, g, g, 2)
    if name == "latlon":
        for g in ll:
            out.add(g)
        return out, {"origin": 0, "span": pl.SIZE, "grid": LOC_GRID, "dim": 2, "names": None,
                     "feature_names": ["lat", "lon"], "dtype": "float32",
                     "input": "lat/lon (EPSG:4326) of the grid cell centres, from ANNOTATIONS_GEOTIFF CRS and transform"}
    spec = yl.ENCODERS[name]
    encode = (yl.mind_encoder if name == "mind" else yl.satclip_encoder)(spec, str(device()))
    emb = np.asarray(encode(ll.reshape(-1, 2).astype(np.float64)), np.float32)
    if emb.shape != (ll.reshape(-1, 2).shape[0], spec["dim"]) or not np.isfinite(emb).all():
        raise RuntimeError(f"Unexpected {name} output {emb.shape}")
    for g in emb.reshape(len(ids), LOC_GRID, LOC_GRID, -1):
        out.add(g)
    return out, {"origin": 0, "span": pl.SIZE, "grid": LOC_GRID, "dim": spec["dim"], "names": f"{name}_",
                 "encoder": name, "source_revision": spec["source_revision"], "weights": spec["weights_repo"],
                 "weights_sha256": pl.sha256(spec["weights"]),
                 "input": f"lat/lon of {LOC_GRID} x {LOC_GRID} cell centres ({pl.SIZE // LOC_GRID} px) per patch"}


def window_series(a, w=POOL):
    """(T, C, H, W) -> (windows, w, w, T, C) gap-filled, and (windows, T) month usable (>= MIN_COVER observed)."""
    T, C, H, W = a.shape
    x = a.reshape(T, C, H // w, w, W // w, w).transpose(2, 4, 3, 5, 0, 1).reshape(-1, w, w, T, C).copy()
    obs = np.isfinite(x).all(-1).mean((1, 2)) >= MIN_COVER
    for k in range(len(x)):
        for t in range(T):
            xt = x[k, :, :, t]
            bad = ~np.isfinite(xt)
            if bad.any():
                med = np.nanmedian(xt.reshape(-1, C), 0) if np.isfinite(xt).any() else np.zeros(C)
                xt[bad] = np.broadcast_to(np.nan_to_num(med), xt.shape)[bad]
    return x, obs


def run_olmoearth(ids, batch=32):
    """OlmoEarth-v1.2-Base on 16 x 16 px windows of the monthly series at patch size 4, months at their mid-points;
    mean of the window's S2 tokens. (YieldSAT runs it per pixel at patch size 1; over 45 M pixels that is ~100x the
    compute, and protocol 7.1/7.2 pools to 16 px windows anyway.)"""
    import torch
    from tools import embed_yieldsat_olmoearth as yo
    model = yo.load_model()
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    dev = device()
    enc = model.encoder.to(dev)
    bands = list(Modality.SENTINEL2_L2A.band_order)
    bidx = [MONTHLY_BANDS.index(b) for b in bands]
    n_sets = len(Modality.SENTINEL2_L2A.band_sets)
    normalizer = Normalizer(Strategy.COMPUTED)
    stamps = np.array([[d.day, d.month - 1, d.year] for d in MONTH_MID], np.int64)
    nw, w = pl.SIZE // POOL, POOL
    out = Grid()
    with ThreadPoolExecutor(4) as ex, torch.inference_mode():
        for p, (x, obs) in zip(ids, ex.map(lambda p: window_series(monthly(p)[:, bidx]), ids)):
            emb = np.full((len(x), 0), np.nan, np.float32)
            k = obs.sum(1)
            for kk in np.unique(k[k > 0]):
                rows = np.flatnonzero(k == kk)
                for b0 in range(0, len(rows), batch):
                    r = rows[b0:b0 + batch]
                    order = np.argsort(~obs[r], axis=1, kind="stable")[:, :kk]           # kept months, in order
                    v = np.take_along_axis(x[r], order[:, None, None, :, None], 3)       # (b, w, w, kk, C)
                    v = np.asarray(normalizer.normalize(Modality.SENTINEL2_L2A, v), np.float32)
                    sample = MaskedOlmoEarthSample(
                        sentinel2_l2a=torch.as_tensor(v, device=dev),
                        sentinel2_l2a_mask=torch.full((len(r), w, w, int(kk), n_sets),
                                                      float(MaskValue.ONLINE_ENCODER.value), device=dev),
                        timestamps=torch.as_tensor(stamps[order], device=dev))
                    tok = enc(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a
                    e = tok.mean(dim=(1, 2, 3, 4)).float().cpu().numpy()
                    if not np.isfinite(e).all():
                        raise RuntimeError(f"Non-finite OlmoEarth output, patch {p}")
                    if emb.shape[1] == 0:
                        emb = np.full((len(x), e.shape[1]), np.nan, np.float32)
                    emb[r] = e
            has = k > 0
            if not has.all():
                emb[~has] = np.nanmean(emb[has], 0) if has.any() else 0
            out.add(emb.reshape(nw, nw, -1), valid=bool(has.any()), windows_without_data=int((~has).sum()))
    dim = out.grids[0].shape[-1]
    return out, {"origin": 0, "span": pl.SIZE, "grid": nw, "dim": dim, "names": "olmoearth_",
                 "model": "OlmoEarth-v1.2-Base encoder", "hf_repo": yo.HF_REPO, "hf_revision": yo.HF_REVISION,
                 "source_revision": yo.SOURCE_REVISION, "weights_sha256": pl.sha256(yo.WEIGHTS / "weights.pth"),
                 "input": "12 monthly mosaics, S2 L2A 12 bands, timestamps = month mid-points; months with "
                          f"< {MIN_COVER:.0%} of a window observed left out, other gaps window-month median",
                 "window": f"{POOL} x {POOL} px, patch size 4", "normalization": "Normalizer(Strategy.COMPUTED)",
                 "pooling": "mean of the window's S2 output tokens (space, time, band sets)",
                 "differs_from_shipped": "EMB_OLMOEARTH_V1_3 is v1.3, 128-d projected read-out, patch 2 / latent 1"}


def prithvi_months(a):
    """Four label-free months, sorted: January, NDVI rise (last month before the peak at or below the midpoint
    between the year's minimum and peak median NDVI), the peak (peak_month), decline (first month after it at or
    below that midpoint). Repeats are allowed."""
    v = ndvi(a)
    med = np.array([np.nanmedian(v[m]) if np.isfinite(v[m]).any() else np.nan for m in range(len(v))])
    peak, _ = peak_month(a)
    mid = (np.nanmin(med) + med[peak]) / 2
    up = max([m for m in range(peak) if med[m] <= mid] or [0])
    down = min([m for m in range(peak + 1, len(med)) if med[m] <= mid] or [len(med) - 1])
    return sorted([0, up, peak, down])


def run_prithvi(ids, batch=4):
    import torch
    from tools import embed_yieldsat_prithvi as ypr
    from tools import yieldsat_patches as yp
    dev = device()
    enc, cfg = ypr.load_model(dev)
    bidx = [MONTHLY_BANDS.index(b) for b in ypr.S2_FOR_HLS]
    mean = np.asarray(cfg["mean"], np.float32)[:, None, None]
    std = np.asarray(cfg["std"], np.float32)[:, None, None]
    size, frames = ypr.SIZE, ypr.FRAMES
    g = size // cfg["patch_size"][1]
    o = -((size - pl.SIZE) // 2)
    out = Grid()

    def load(p):
        a = monthly(p)
        months = prithvi_months(a)
        x = np.stack([yp.crop((fill_nan(a[m][bidx]) - mean) / std, o, o, size) for m in months], 1)  # C, T, S, S
        return p, months, x.astype(np.float32)

    todo = []

    def flush():
        if not todo:
            return
        x = torch.as_tensor(np.stack([t[2] for t in todo]), device=dev)
        coords = torch.tensor([[[2019, MONTH_MID[m].timetuple().tm_yday] for m in t[1]] for t in todo],
                              dtype=torch.float32, device=dev)
        with torch.inference_mode():
            tok = enc.forward_features(x, coords, None)[-1][:, 1:]
        tok = tok.reshape(len(todo), frames, g, g, -1).mean(1).float().cpu().numpy()
        if not np.isfinite(tok).all():
            raise RuntimeError("Non-finite Prithvi output")
        for (p, months, _), t in zip(todo, tok):
            out.add(t, months=months)
        todo.clear()

    with ThreadPoolExecutor(4) as ex:
        for item in ex.map(load, ids):
            todo.append(item)
            if len(todo) >= batch:
                flush()
    flush()
    return out, {"origin": o, "span": size, "grid": g, "dim": cfg["embed_dim"], "names": "prithvi_",
                 "model": "Prithvi-EO-2.0-300M-TL encoder", "hf_repo": ypr.HF_REPO, "hf_revision": ypr.HF_REVISION,
                 "weights_sha256": pl.sha256(ypr.WEIGHTS),
                 "input": "S2 B02,B03,B04,B8A,B11,B12 (as HLS B02-B07) DN of 4 monthly mosaics: January, NDVI rise, "
                          "peak, decline (prithvi_months, label-free); checkpoint mean/std; 224 px window centred on "
                          "the patch, reflection padding; temporal coords (2019, day of year of the month mid-point); "
                          "no location coords",
                 "pooling": "last-layer tokens averaged over the 4 frames; bilinear at the pixel centre"}


ANYSAT_MAX_T = 96          # card: the paper's AnySat input is the first 96 S2 observations of 2019
ANYSAT_PATCH_M = POOL * 10  # AnySat patch = 16 px (160 m): its "patch" output is one token per 16 x 16 px window


def s2_dates():
    feats = json.loads((pl.DATA / "metadata.geojson").read_text())["features"]
    return {int(f["properties"]["ID_PATCH"]): [dt.datetime.strptime(str(v), "%Y%m%d").date()
                                               for _, v in sorted(f["properties"]["dates-S2"].items(),
                                                                  key=lambda kv: int(kv[0]))] for f in feats}


def anysat_stats(meta, n_patches=64, n_obs=8, n_pix=2048):
    """Band mean/std of nonzero DATA_S2 values over a seeded sample of patches, 2019 observations and pixels
    (no labels; deterministic, so every part computes the same numbers)."""
    rng = np.random.default_rng(0)
    dates = s2_dates()
    vals = []
    for p in np.sort(rng.choice(meta.patch_ids, n_patches, replace=False)):
        a = np.load(pl.DATA / "DATA_S2" / f"S2_{int(p)}.npy", mmap_mode="r")
        t19 = [i for i, d in enumerate(dates[int(p)]) if d.year == 2019][:ANYSAT_MAX_T]
        ts = np.sort(rng.choice(t19, min(n_obs, len(t19)), replace=False))
        pix = rng.choice(pl.SIZE * pl.SIZE, n_pix, replace=False)
        v = np.asarray(a[ts]).reshape(len(ts), 10, -1)[:, :, pix].transpose(0, 2, 1).reshape(-1, 10)
        vals.append(v[(v > 0).all(1)])
    v = np.concatenate(vals).astype(np.float64)
    return v.mean(0).astype(np.float32), v.std(0).astype(np.float32), int(len(v))


def run_anysat(ids):
    import torch
    from tools import embed_yieldsat_anysat as ya
    if ya.BANDS != pl.S2_BANDS:
        raise ValueError("AnySat band order differs from DATA_S2")
    dev = device()
    model = ya.load_model(dev)
    meta = pl.Meta()
    mean, std, n_stat = anysat_stats(meta)
    dates = s2_dates()
    g = pl.SIZE // POOL
    out = Grid()

    def load(p):
        t19 = [i for i, d in enumerate(dates[p]) if d.year == 2019][:ANYSAT_MAX_T]
        a = np.load(pl.DATA / "DATA_S2" / f"S2_{p}.npy")
        if a.shape[1:] != (10, pl.SIZE, pl.SIZE) or len(dates[p]) != len(a):
            raise ValueError(f"S2_{p}.npy: shape {a.shape} vs {len(dates[p])} dates")
        x = ((a[t19].astype(np.float32) - mean[:, None, None]) / std[:, None, None])
        first = dates[p][t19[0]]
        days = [(dates[p][i] - dt.date(first.year, first.month, 1)).days for i in t19]
        return p, x, np.array(days, np.float32)

    with ThreadPoolExecutor(2) as ex, torch.inference_mode():
        for p, x, days in ex.map(load, ids):
            o = model({"s2": torch.as_tensor(x[None], device=dev),
                       "s2_dates": torch.as_tensor(days[None], device=dev)},
                      patch_size=ANYSAT_PATCH_M, output="patch").float().cpu().numpy()[0]
            if o.shape[:2] != (g, g) or not np.isfinite(o).all():
                raise RuntimeError(f"AnySat output {o.shape} for patch {p}")
            out.add(o, observations=len(days))
    return out, {"origin": 0, "span": pl.SIZE, "grid": g, "dim": out.grids[0].shape[-1], "names": "anysat_",
                 "model": "AnySat-Base", "hf_repo": ya.HF_REPO, "hf_revision": ya.HF_REVISION,
                 "source_revision": ya.SOURCE_REVISION,
                 "input": f"DATA_S2 10 bands, the first {ANYSAT_MAX_T} observations of 2019 (dates-S2), as the paper's "
                          "AnySat input (card); dates = days since the 1st of the first observation's month; the "
                          "whole 128 px patch as one tile",
                 "normalization": f"band mean/std of nonzero values, seeded sample ({n_stat} values, no labels)",
                 "pooling": f"AnySat output='patch' at patch size {ANYSAT_PATCH_M} m: one token per 16 x 16 px"}


# ---- dense per-pixel inputs --------------------------------------------------------------------------------------

def dense_harmonic():
    from tools import build_yieldsat_s2_harmonic as h
    t = np.array([(d - dt.date(2019, 1, 1)).days for d in MONTH_MID], np.float64)
    names = [f"{b}_{term}" for b in pl.S2_BANDS for term in h.TERMS]

    def per_patch(p, pix):
        a = monthly(p)[:, :10].reshape(pl.N_MONTHS, 10, -1)[:, :, pix].transpose(2, 0, 1)  # (n, T, 10)
        coef = h.fit(a.astype(np.float64), t)                             # (n, 10, 5)
        return coef.reshape(len(pix), -1), np.isfinite(coef).all((1, 2))

    return names, per_patch, {"input": "12 monthly values per band of the 10 DATA_S2 bands (DN)",
                              "model": f"c0 + {h.K} harmonics, period {h.PERIOD} d, t = day of year of the month "
                                       f"mid-point, ridge {h.RIDGE} on harmonic terms, >= {h.MIN_OBS} months",
                              "builder": "tools/build_yieldsat_s2_harmonic.fit"}


def dense_biophys():
    from tools import build_yieldsat_biophys as bp
    nets = {v: bp.load(v) for v in bp.VARIABLES}
    test_error = bp.check_networks(nets)
    cols = [MONTHLY_BANDS.index(b) for b in bp.BANDS]
    names = [f"{v}_m{m:02d}" for m in range(pl.N_MONTHS) for v in bp.VARIABLES.values()]
    cos_view = np.cos(np.radians(bp.VIEW_ZENITH_DEG))

    def per_patch(p, pix):
        a = monthly(p).reshape(pl.N_MONTHS, 12, -1)[:, cols][:, :, pix].transpose(2, 0, 1) / 1e4  # (n, T, 8)
        lat = float(latlon_grid(p, 1)[0, 0, 0])
        out = np.full((len(pix), pl.N_MONTHS, len(nets)), np.nan, np.float32)
        for m in range(pl.N_MONTHS):
            x = np.column_stack([a[:, m], np.full(len(pix), cos_view),
                                 np.full(len(pix), max(bp.sun_zenith_cos(lat, MONTH_MID[m]), 0.05)),
                                 np.full(len(pix), bp.COS_RELATIVE_AZIMUTH)])
            ok = np.isfinite(x).all(1)
            for v, net in enumerate(nets.values()):
                out[ok, m, v] = bp.network(net, x[ok])
        return out.reshape(len(pix), -1), np.isfinite(out).any((1, 2))

    return names, per_patch, {"variables": bp.VARIABLES, "snap_test_case_max_abs_error": test_error,
                              "geometry": {"view_zenith_deg": bp.VIEW_ZENITH_DEG,
                                           "cos_relative_azimuth": bp.COS_RELATIVE_AZIMUTH,
                                           "sun": f"patch centre latitude, month mid-point, {bp.LOCAL_SOLAR_HOUR} h"},
                              "source": f"senbox-org/s2tbx@{bp.COMMIT} auxdata 3_0/S2A (GPL-3.0)",
                              "missing": "months without data stay NaN (the heads impute or standardise to 0)"}


def encoder_output(name, ids):
    if name in PATCH_MODELS:
        return run_patch_model(name, ids)
    if name in TIME_SERIES:
        return globals()[f"run_{name}"](ids)
    return run_location(name, ids)


# ---- parts and combine -------------------------------------------------------------------------------------------

def run_part(name, part, parts):
    meta = pl.Meta()
    ids, start = part_ids(meta, part, parts)
    d = pl.FEATURES / name / "parts"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if name in DENSE_INPUTS:
        names, per_patch, desc = (dense_harmonic if name == "s2-harmonic" else dense_biophys)()
        xs, oks = [], []
        for i, p in enumerate(ids):
            k = meta.pos[p]
            pix = meta.pix[meta.offsets[k]:meta.offsets[k + 1]]
            x, ok = per_patch(p, pix)
            xs.append(x.astype(np.float32))
            oks.append(ok)
        x = np.concatenate(xs)
        np.save(d / f".{part:03d}.tmp.npy", x.astype(np.float16) if np.nanmax(np.abs(x)) < 6e4 else x)
        np.save(d / f".{part:03d}-valid.tmp.npy", np.concatenate(oks))
        info = {"names": names, "desc": desc, "rows": int(len(x)), "max_abs": float(np.nanmax(np.abs(x)))}
    else:
        g, desc = encoder_output(name, ids)
        grid = np.stack(g.grids)
        dtype = np.float32 if desc.get("dtype") == "float32" or np.abs(grid).max() >= 6e4 else np.float16
        np.save(d / f".{part:03d}.tmp.npy", grid.astype(dtype))
        np.save(d / f".{part:03d}-valid.tmp.npy", np.array(g.valid))
        info = {"desc": desc, "notes": g.notes, "max_abs": float(np.abs(grid).max()), "dtype": np.dtype(dtype).name}
    info.update(code_commit=code_commit(), part=part, parts=parts, first_patch_index=start, patches=len(ids), seconds=round(time.time() - t0, 1),
                device=str(device()) if name not in DENSE_INPUTS else "cpu")
    for suffix in ("", "-valid"):
        (d / f".{part:03d}{suffix}.tmp.npy").rename(d / f"{part:03d}{suffix}.npy")
    (d / f"{part:03d}.json").write_text(json.dumps(info, default=str) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("notes", "names", "desc")}), flush=True)


def combine(name, parts):
    meta = pl.Meta()
    out = pl.FEATURES / name
    d = out / "parts"
    infos = [json.loads((d / f"{i:03d}.json").read_text()) for i in range(parts)]
    if [i["first_patch_index"] for i in infos] != sorted(i["first_patch_index"] for i in infos) or \
            sum(i["patches"] for i in infos) != len(meta.patch_ids):
        raise ValueError(f"{name}: parts don't cover the {len(meta.patch_ids)} patches")
    desc = infos[0]["desc"]
    common = {"input": name, "rows": int(len(meta.y)), "meta_sha256": meta.sha256,
              "dataset_revision": "51b74a91424e7dc9ea1fe0042390353658af8185", "source": "DATA_S2_MONTHLY",
              "seconds": round(sum(i["seconds"] for i in infos), 1), "device": infos[0]["device"],
              "max_abs": max(i["max_abs"] for i in infos),
              "code_commit": sorted({i.get("code_commit") or "" for i in infos})}
    if name in DENSE_INPUTS:
        xs = [np.load(d / f"{i:03d}.npy", mmap_mode="r") for i in range(parts)]
        valid = np.concatenate([np.load(d / f"{i:03d}-valid.npy") for i in range(parts)])
        if sum(len(x) for x in xs) != len(meta.y):
            raise ValueError(f"{name}: {sum(len(x) for x in xs)} rows, meta.npz has {len(meta.y)}")
        dtype = np.result_type(*[x.dtype for x in xs])
        tmp = out / ".X.tmp.npy"
        with open(tmp, "wb") as fh:                      # sequential writes (no memmap on a nearly full Lustre)
            np.lib.format.write_array_header_1_0(fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                                      "fortran_order": False,
                                                      "shape": (len(meta.y), len(infos[0]["names"]))})
            for x in xs:
                for a in range(0, len(x), 1 << 20):
                    fh.write(np.ascontiguousarray(x[a:a + (1 << 20)], dtype=dtype).tobytes())
        np.save(out / ".valid.tmp.npy", valid)
        tmp.rename(out / "X.npy")
        (out / ".valid.tmp.npy").rename(out / "valid.npy")
        info = {**common, "kind": "dense", "dtype": np.dtype(dtype).name, "nodata": None,
                "feature_names": infos[0]["names"], "invalid_pixels": int((~valid).sum()), **desc}
    else:
        grid = np.concatenate([np.load(d / f"{i:03d}.npy") for i in range(parts)])
        valid = np.concatenate([np.load(d / f"{i:03d}-valid.npy") for i in range(parts)])
        np.save(out / ".grid.tmp.npy", grid)
        np.save(out / ".patch_valid.tmp.npy", valid)
        (out / ".grid.tmp.npy").rename(out / "grid.npy")
        (out / ".patch_valid.tmp.npy").rename(out / "patch_valid.npy")
        notes = [n for i in infos for n in i["notes"]]
        months = [n["month"] for n in notes if "month" in n]
        names = desc.pop("feature_names", None) or [f"{desc['names']}{k:04d}" for k in range(desc["dim"])]
        info = {**common, "kind": "grid", "dtype": grid.dtype.name, "grid_shape": list(grid.shape),
                "origin": desc.pop("origin"), "span": desc.pop("span"), "grid": desc.pop("grid"),
                "feature_names": names, "patches_invalid": int((~valid).sum()),
                **({"peak_month_counts": np.bincount(months, minlength=12).tolist()} if months else {}),
                **{k: v for k, v in desc.items() if k not in ("names", "dim", "dtype")}}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "feature_names"}, default=str)[:2000], flush=True)


def smoke(name):
    meta = pl.Meta()
    ids = [int(p) for p in meta.patch_ids[:2]] + [int(p) for p in meta.patch_ids[-2:]]
    t0 = time.time()
    if name in DENSE_INPUTS:
        names, per_patch, desc = (dense_harmonic if name == "s2-harmonic" else dense_biophys)()
        k = meta.pos[ids[0]]
        x, ok = per_patch(ids[0], meta.pix[meta.offsets[k]:meta.offsets[k + 1]])
        print(name, x.shape, len(names), "valid", ok.mean(), "nan share", np.isnan(x).mean(), "max", np.nanmax(x))
        return
    g, desc = encoder_output(name, ids)
    grid = np.stack(g.grids)
    print(name, grid.shape, "valid", g.valid, "notes", g.notes[:2], "max_abs", float(np.abs(grid).max()),
          f"{time.time() - t0:.0f}s for {len(ids)} patches", flush=True)
    print(json.dumps({k: v for k, v in desc.items() if k != "feature_names"}, default=str)[:1500])


# ---- submission --------------------------------------------------------------------------------------------------

RESOURCES = {"galileo": ("40", "16", "48G", "12:00:00"), "presto": ("20", "16", "48G", "06:00:00"),
             "olmoearth": ("40", "16", "48G", "12:00:00"), "prithvi": ("20", "16", "64G", "08:00:00"),
             "anysat": ("20", "16", "64G", "08:00:00"),
             "thor": ("20", "16", "64G", "08:00:00"), "clay": ("20", "16", "64G", "08:00:00"),
             "satmae": ("20", "16", "48G", "08:00:00")}
DEFAULT_RESOURCES = ("10", "16", "48G", "06:00:00")
SMALL = {"latlon": ("1", "2", "8G", "01:00:00"), "satclip": ("1", "4", "16G", "01:00:00"),
         "mind": ("1", "8", "32G", "02:00:00"), "s2-harmonic": ("20", "4", "16G", "04:00:00"),
         "biophys": ("20", "4", "16G", "04:00:00")}


def submit(name, repo):
    """Array of parts (main partition, CPU) and an afterok combine job. Runs from a frozen copy of the commit."""
    parts, cpus, mem, wall = SMALL.get(name) or RESOURCES.get(name) or DEFAULT_RESOURCES
    logs = pl.FEATURES / name / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    if name in NEEDS_PATCHFM:
        env["BENCHMARK_PYTHON"] = str(PATCHFM_ENV)
    base = [f"{SLURM}/sbatch", "--parsable", "--partition=main", f"--chdir={repo}"]
    array = subprocess.run(base + [f"--job-name=pl-emb-{name}", f"--array=0-{int(parts) - 1}%20",
                                   f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={wall}",
                                   f"--output={logs}/part-%A_%a.log", str(repo / "planteur-embed.sbatch"),
                                   "--input", name, "--parts", parts],
                           capture_output=True, text=True, check=True, env=env).stdout.strip()
    comb = subprocess.run(base + [f"--dependency=afterok:{array}", f"--job-name=pl-emb-{name}-combine",
                                  "--cpus-per-task=2", "--mem=32G", "--time=01:00:00",
                                  f"--output={logs}/combine-%j.log", str(repo / "planteur-embed.sbatch"),
                                  "--input", name, "--parts", parts, "--combine"],
                          capture_output=True, text=True, check=True, env=env).stdout.strip()
    print(f"{name}: parts {array} ({parts} x {cpus} CPU, {mem}, {wall}), combine {comb}", flush=True)
    return comb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, choices=GRID_INPUTS + DENSE_INPUTS)
    p.add_argument("--parts", type=int, default=None)
    p.add_argument("--part", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--submit", action="store_true", help="submit parts + combine for each input (frozen code)")
    a = p.parse_args()
    if a.submit:
        from tools.sweep_yieldsat import frozen_code
        repo = frozen_code(Path(__file__).resolve().parents[1], pl.RUNS / "embed", False)
        for name in a.input:
            if (pl.FEATURES / name / "info.json").exists():
                print(f"{name}: already built, skipped")
                continue
            submit(name, repo)
        return
    for name in a.input:
        if a.smoke:
            smoke(name)
        elif a.combine:
            combine(name, a.parts)
        else:
            part = a.part if a.part is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
            run_part(name, part, a.parts)


if __name__ == "__main__":
    main()
