"""Image patches around YieldSAT pixels, for patch-based foundation models (protocols/yieldsat.md, 2.5).

What YieldSAT has spatially. The v1 cache is a per-pixel table, but every row keeps its pixel's (row, col) inside
its field's raster (v1/<subset>-meta.npz), and the dataset ships those rasters: per field, Raw/<country>/<country>/
<field>/ holds one 12-band Sentinel-2 L2A GeoTIFF per acquisition (s2_images/S2_L2A_<yyyymmdd>.tif, uint16 DN,
10 m, all bands resampled to one grid), the matching scene-classification layer (scl_masks/), DEM derivatives (dem/),
soil and weather layers, all on one grid per field. The grid is the field's bounding box in its UTM zone (median
about 55x60 px in Germany, 120x130 px elsewhere, at most 320 px a side), and pixels outside the field keep their real
reflectance; only about a third of the box carries a yield label. tools/build_yieldsat_latlon.py already relies on
(row, col) indexing this grid. There is no Sentinel-1 anywhere in YieldSAT.

Approach. So no raster has to be rebuilt from pixel coordinates: a patch model sees the field's own Raw raster,
which also gives it the real surroundings inside the bounding box. Per field and model input size S (e.g. 224):
  * the raster is cut into S x S windows (`windows`). A box no larger than S gets one window centred on it; a larger
    box gets evenly spaced, overlapping windows. Areas outside the raster are filled by reflection (`crop`), so the
    model sees plausible texture rather than an out-of-distribution constant. Native 10 m GSD is kept (no resizing).
  * each labelled pixel is assigned to the window whose centre is nearest (`assign`), so it has the most context.
  * its embedding is the model's patch-token grid sampled at the pixel centre (`sample_tokens`): bilinear between
    the 4 surrounding token centres, clamped at the border (`mode="nearest"` gives the token that covers the pixel).
    With 16-px tokens a token covers 160 m, so within-field variation comes from the interpolation and from tokens
    differing across the field; this is the price of patch models on 10 m pixels and is recorded per input.

Date choice for single-date models (`peak_date`, label-free): among the field's acquisitions from sowing to harvest
(metadata dates; the harvest date only bounds the window), keep those whose SCL is clear (vegetation, bare soil,
water, unclassified) on at least MIN_CLEAR of the field's labelled pixels; take the one with the highest median EVI2
over those pixels. If no acquisition is clear enough, the threshold drops to FALLBACK_CLEAR, then the clearest
acquisition is used. `phenology_dates` adds the clear acquisitions nearest to half-amplitude green-up (before the
peak) and half-amplitude senescence (after it), for models fed a few dates. Nothing here looks at yield.

    from tools import yieldsat_patches as yp
    for f in yp.fields("GER-W"):          # v1 order; f.rows is the field's slice of v1 rows
        date = yp.peak_date(f)
        img = yp.fill_nan(f.s2(date))     # (12, H, W) float32 DN, RAW_S2_BANDS order
        for k, (r0, c0) in enumerate(yp.windows(*img.shape[1:], 224)):
            x = yp.crop(img, r0, c0, 224)                 # (12, 224, 224)
            ...                                           # model -> tokens (D, 14, 14)
            sel = yp.assign(f.row, f.col, wins, 224) == k
            emb[f.rows][sel] = yp.sample_tokens(tokens, f.row[sel], f.col[sel], r0, c0, 224)

    python -m tools.yieldsat_patches --subset GER-W [--fields 20]    # sanity check: grids, rows/cols, date rule
"""
import argparse
import datetime as dt
import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import yieldsat as ys

# Band order inside every Raw S2 GeoTIFF (checked against the file's band descriptions on read).
RAW_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CLEAR_SCL = (4, 5, 6, 7)   # vegetation, not vegetated, water, unclassified (cloud, shadow, cirrus, snow excluded)
MIN_CLEAR = 0.9            # share of a field's labelled pixels that must be clear for a date to count
FALLBACK_CLEAR = 0.6
DN_SCALE = 10000.0         # L2A DN -> reflectance (no BOA offset in these files: values sit around 1500-4000 DN)


def _date(stem):
    s = stem[-8:]
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


@dataclass
class Field:
    subset: str
    name: str
    country: str
    index: int          # field index in v1/<subset>-meta.npz
    rows: slice         # this field's rows in the v1 cache
    row: np.ndarray     # labelled pixels' raster row, aligned with `rows`
    col: np.ndarray
    sowing: dt.date
    harvest: dt.date

    @property
    def folder(self):
        return ys.DATA / "Raw" / self.country / self.country / self.name

    @functools.cached_property
    def acquisitions(self):
        """Sorted list of (date, path) of the field's S2 GeoTIFFs."""
        return sorted((_date(p.stem), p) for p in (self.folder / "s2_images").glob("S2_L2A_*.tif"))

    @functools.cached_property
    def shape(self):
        import rasterio
        with rasterio.open(self.acquisitions[0][1]) as src:
            return src.height, src.width

    def s2(self, date):
        """(12, H, W) float32 DN in RAW_S2_BANDS order; 0 and 65535 (nodata / saturated) become NaN."""
        import rasterio
        path = dict(self.acquisitions)[date]
        with rasterio.open(path) as src:
            names = [d.split(" ")[0] for d in src.descriptions]
            if names != RAW_S2_BANDS:
                raise ValueError(f"{path}: unexpected band order {names}")
            x = src.read().astype(np.float32)
        if x.shape[1:] != self.shape:
            raise ValueError(f"{path}: grid {x.shape[1:]} differs from the field's {self.shape}")
        x[(x == ys.S2_NODATA) | (x == 0)] = np.nan
        return x

    def scl(self, date):
        import rasterio
        path = self.folder / "scl_masks" / f"S2_L2A_SCL_{date:%Y%m%d}.tif"
        with rasterio.open(path) as src:
            return src.read(1)

    def dem(self):
        """(H, W) float32 elevation in metres from the field's Raw DEM (same grid as S2)."""
        import rasterio
        with rasterio.open(self.folder / "dem" / f"dem-{self.name}.tif") as src:
            x = src.read(1).astype(np.float32)
        if x.shape != self.shape:
            raise ValueError(f"{self.name}: DEM grid {x.shape} differs from S2 {self.shape}")
        return x

    def stack(self, dates):
        """(T, 12, H, W) for several dates (e.g. for multi-temporal patch models)."""
        return np.stack([self.s2(d) for d in dates])


@functools.lru_cache(maxsize=None)
def _season_dates():
    out = {}
    for path in (ys.DATA / "Raw").glob("*/*/*/metadata-*.json"):
        m = json.loads(path.read_text())
        out[m["field_shared_name"]] = (m["adm_units"]["country"],
                                       dt.datetime.strptime(m["seeding_date"], "%d.%m.%Y").date(),
                                       dt.datetime.strptime(m["harvesting_date"], "%d.%m.%Y").date())
    return out


def fields(subset):
    """Every field of a subset in v1 order, with its v1 row slice and its labelled pixels' raster positions."""
    meta = np.load(ys.FEATURES / "v1" / f"{subset}-meta.npz")
    names, fidx, row, col = meta["fields"], meta["field_idx"], meta["row"], meta["col"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    season = _season_dates()
    for a, b in zip(bounds[:-1], bounds[1:]):
        name = str(names[fidx[a]])
        country, sowing, harvest = season[name]
        yield Field(subset, name, country, int(fidx[a]), slice(int(a), int(b)), row[a:b].astype(int),
                    col[a:b].astype(int), sowing, harvest)


# ---- date choice -------------------------------------------------------------------------------------------------

def evi2(x):
    """Two-band EVI from a (12, ...) DN array."""
    nir, red = x[RAW_S2_BANDS.index("B08")] / DN_SCALE, x[RAW_S2_BANDS.index("B04")] / DN_SCALE
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1)


def greenness_curve(field):
    """Per in-season acquisition: (date, clear share of labelled pixels, median EVI2 over clear labelled pixels).

    In season = sowing to harvest; if the window holds no acquisition (should not happen) all acquisitions count.
    """
    acq = [d for d, _ in field.acquisitions if field.sowing <= d <= field.harvest] or \
          [d for d, _ in field.acquisitions]
    out = []
    for d in acq:
        clear = np.isin(field.scl(d)[field.row, field.col], CLEAR_SCL)
        x = field.s2(d)[:, field.row, field.col]
        v = evi2(x)
        ok = clear & np.isfinite(v)
        out.append((d, float(ok.mean()), float(np.median(v[ok])) if ok.any() else np.nan))
    return out


def _usable(curve):
    for threshold in (MIN_CLEAR, FALLBACK_CLEAR):
        ok = [c for c in curve if c[1] >= threshold and np.isfinite(c[2])]
        if ok:
            return ok, threshold
    best = max(curve, key=lambda c: c[1])
    return [best], best[1]


def peak_date(field, curve=None):
    """The clear in-season acquisition with the highest median EVI2 (see module docstring)."""
    ok, _ = _usable(curve or greenness_curve(field))
    return max(ok, key=lambda c: (np.nan_to_num(c[2], nan=-9), -c[0].toordinal()))[0]


def phenology_dates(field, curve=None):
    """[green-up, peak, senescence]: clear acquisitions nearest half amplitude before / after the EVI2 peak.

    Half amplitude = (min + peak) / 2 with min over the clear season. When no clear date exists on one side of the
    peak, that slot repeats the peak date, so the output always has three dates.
    """
    curve = curve or greenness_curve(field)
    ok, _ = _usable(curve)
    peak = peak_date(field, curve)
    v = {c[0]: c[2] for c in ok if np.isfinite(c[2])}
    if not v:
        return [peak, peak, peak]
    half = (min(v.values()) + v[peak]) / 2
    before = [d for d in v if d < peak]
    after = [d for d in v if d > peak]
    up = min(before, key=lambda d: abs(v[d] - half)) if before else peak
    down = min(after, key=lambda d: abs(v[d] - half)) if after else peak
    return [up, peak, down]


# ---- windows and token sampling ----------------------------------------------------------------------------------

def windows(height, width, size, overlap=0.5):
    """Top-left (r0, c0) of S x S windows covering an H x W raster (r0/c0 may be negative: padded)."""
    def axis(n):
        if n <= size:
            return [-((size - n) // 2)]
        k = int(np.ceil((n - size) / (size * (1 - overlap)))) + 1
        return [int(round(v)) for v in np.linspace(0, n - size, k)]
    return [(r, c) for r in axis(height) for c in axis(width)]


def assign(row, col, wins, size):
    """Index of the window whose centre is nearest (Chebyshev) to each pixel."""
    centres = np.array([(r + size / 2, c + size / 2) for r, c in wins])
    d = np.maximum(np.abs(row[:, None] + 0.5 - centres[None, :, 0]), np.abs(col[:, None] + 0.5 - centres[None, :, 1]))
    return d.argmin(1)


def crop(img, r0, c0, size, mode="reflect"):
    """(C, S, S) window of a (C, H, W) image starting at (r0, c0); outside the raster is filled by `mode`."""
    c, h, w = img.shape
    top, left = max(0, -r0), max(0, -c0)
    bottom, right = max(0, r0 + size - h), max(0, c0 + size - w)
    if top or left or bottom or right:
        img = np.pad(img, ((0, 0), (top, bottom), (left, right)), mode=mode)
    r, cc = r0 + top, c0 + left
    return img[:, r:r + size, cc:cc + size]


def fill_nan(img):
    """Replace NaN with the band's median over the image (0 if the band is all NaN)."""
    img = img.copy()
    for b in range(img.shape[0]):
        bad = ~np.isfinite(img[b])
        if bad.any():
            fill = np.nanmedian(img[b]) if (~bad).any() else 0.0
            img[b][bad] = fill
    return img


def sample_tokens(tokens, row, col, r0, c0, size, mode="bilinear"):
    """Embeddings at pixel centres from a (D, gh, gw) token grid that covers the window [r0, r0+S) x [c0, c0+S).

    Bilinear between token centres (border-clamped), or the covering token with mode="nearest". Accepts numpy or
    torch tokens; returns float32 numpy (n, D).
    """
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(tokens).float()
    y = (torch.as_tensor(row - r0, dtype=torch.float32) + 0.5) / size * 2 - 1
    x = (torch.as_tensor(col - c0, dtype=torch.float32) + 0.5) / size * 2 - 1
    grid = torch.stack([x, y], -1).to(t.device)[None, None]          # (1, 1, n, 2), x first
    out = F.grid_sample(t[None], grid, mode=mode, padding_mode="border", align_corners=False)
    return out[0, :, 0].T.cpu().numpy()


def check(subset, n_fields=20):
    """Sanity checks on a sample of fields: grids, labelled pixels inside them, and the date rule."""
    fs = list(fields(subset))
    pick = np.linspace(0, len(fs) - 1, min(n_fields, len(fs))).astype(int)
    report = []
    for i in pick:
        f = fs[i]
        h, w = f.shape
        if f.row.max() >= h or f.col.max() >= w:
            raise ValueError(f"{f.name}: labelled pixel outside its {h}x{w} raster")
        curve = greenness_curve(f)
        ph = phenology_dates(f, curve)
        clear = dict((c[0], c[1]) for c in curve)
        report.append({"shape": [h, w], "labelled_share": round(len(f.row) / (h * w), 3),
                       "acq_in_season": len(curve), "peak_doy_after_sowing": (ph[1] - f.sowing).days,
                       "peak_clear": round(clear[ph[1]], 3),
                       "greenup_peak_senescence_days": [(d - f.sowing).days for d in ph],
                       "windows_224": len(windows(h, w, 224))})
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=ys.SUBSETS, required=True)
    p.add_argument("--fields", type=int, default=20)
    a = p.parse_args()
    for r in check(a.subset, a.fields):
        print(json.dumps(r))
