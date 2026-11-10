"""Shared PLANTEUR definitions: paths, classes, inputs, splits, the pixel cache and the metric.

Everything here follows protocols/planteur.md; section numbers refer to that page.

The pixel cache (built by tools/build_planteur_features.py) lives in features/planteur/:

    meta.npz            one row per non-void pixel of every patch, patches in ascending ID order and pixels in
                        row-major order within a patch: y (uint8 class), pix (uint16 row*128+col); and per patch:
                        patch_ids, offsets (start row of each patch; len = patches + 1)
    <input>/X.npy       (rows, d) features aligned with meta.npz (float16 embeddings, uint16 S2 reflectance)
    <input>/valid.npy   (rows,) bool: False where the input has no value for the pixel (D7: embedding missing,
                        zero or NaN); such pixels are left out of training and predicted as Background
    <input>/info.json   feature names, dtype, no-data value, source hashes, meta.npz hash

Void (255) pixels are not in the cache at all: they are never trained on or scored (3.1).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("AGRIBENCH_PROJECT", "/lustre/shared/agriscience-benchmark"))
DATA = Path(os.environ.get("AGRIBENCH_PLANTEUR", PROJECT / "datasets" / "planteur"))
FEATURES = Path(os.environ.get("AGRIBENCH_PLANTEUR_FEATURES", PROJECT / "features" / "planteur"))
RUNS = PROJECT / "runs" / "planteur"

SIZE = 128
VOID = 255
CLASSES = ["Background", "Meadow", "Fruits, vegetables, flowers", "Orchard", "Sugarcane", "Banana", "Pineapple",
           "Tropical tuber"]
N_CLASSES = len(CLASSES)
# LOIO folds in the order of paper Table 6 (splits/loio.json lists them in a different order).
ISLANDS = ["Réunion", "Guadeloupe", "Martinique", "Guyane", "Mayotte"]
SPLITS = ("default", "loio")

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # first 10 of DATA_S2_MONTHLY
N_MONTHS = 12
# Inputs read straight from the dataset (3.6). kind "s2": DATA_S2_MONTHLY, 12 months x the 10 bands above;
# kind "emb": a shipped EMB_* product, (C, 128, 128) float32.
INPUTS = {
    "s2-monthly": {"kind": "s2", "folder": "DATA_S2_MONTHLY", "prefix": "S2"},
    "emb-aef": {"kind": "emb", "folder": "EMB_AEF", "prefix": "AEF"},
    "emb-tessera-v1": {"kind": "emb", "folder": "EMB_TESSERA_V1", "prefix": "TESSERA"},
    "emb-tessera-v1.1": {"kind": "emb", "folder": "EMB_TESSERA_V1_1", "prefix": "TESSERA"},
    "emb-tessera-v2": {"kind": "emb", "folder": "EMB_TESSERA_V2", "prefix": "TESSERA"},
    "emb-olmoearth-v1.3": {"kind": "emb", "folder": "EMB_OLMOEARTH_V1_3", "prefix": "OE13"},
}
# Inputs AgriBench builds from DATA_S2_MONTHLY (tools/planteur_embed.py, protocol 7): "grid" = per-patch token grids
# sampled per pixel by the runner, "dense" = per-pixel tables like the ones above.
for _name in ["galileo", "presto", "copernicusfm", "dofa", "croma", "softcon", "clay", "satmae", "thor", "terramind",
              "latlon", "satclip", "mind"]:
    INPUTS[_name] = {"kind": "grid", "builder": "tools.planteur_embed"}
for _name in ["s2-harmonic", "biophys"]:
    INPUTS[_name] = {"kind": "dense", "builder": "tools.planteur_embed"}
SHIPPED = [k for k, v in INPUTS.items() if v["kind"] in ("s2", "emb")]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def manifest_sha(relpath):
    """SHA-256 of a dataset file from MANIFEST.sha256 (written by the verified download), else hashed now."""
    manifest = DATA / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, _, name = line.partition("  ")
            if name.strip() == relpath:
                return digest
    return sha256(DATA / relpath)


def fold_lists(split, fold):
    """{"train": [...], "val": [...], "test": [...]} patch IDs for one fold: split "default" (fold "default",
    splits/443.json, 3.2) or "loio" (fold = held-out island, splits/loio.json, 3.3)."""
    if split == "default":
        if fold not in (None, "default"):
            raise SystemExit("split=default has one fold, 'default'")
        s = json.loads((DATA / "splits" / "443.json").read_text())["splits"]
        rel = "splits/443.json"
    elif split == "loio":
        folds = json.loads((DATA / "splits" / "loio.json").read_text())["folds"]
        if fold not in folds:
            raise SystemExit(f"split=loio needs fold=<one of {ISLANDS}>, got {fold!r}")
        s = folds[fold]
        rel = "splits/loio.json"
    else:
        raise SystemExit(f"split must be one of {SPLITS}")
    out = {k: sorted(int(i) for i in s[k]) for k in ("train", "val", "test")}
    if set(out["train"]) & set(out["val"]) or set(out["test"]) & (set(out["train"]) | set(out["val"])):
        raise ValueError(f"{rel} fold {fold}: train, val and test overlap")
    return out, rel


def folds_of(split):
    return ["default"] if split == "default" else list(ISLANDS)


def fewshot_pixels(x):
    """Official few-shot pixels for X per class (splits/fewshot_pixels.json, seed 0): {patch: [(row, col, class)]}."""
    data = json.loads((DATA / "splits" / "fewshot_pixels.json").read_text())
    if str(x) not in data["X"]:
        raise SystemExit(f"Few-shot X must be one of {sorted(data['X'], key=int)}")
    return {int(p): [tuple(v) for v in pts] for p, pts in data["X"][str(x)].items()}


class Meta:
    """The pixel table's row layout (meta.npz): which rows belong to which patch, and their labels."""

    def __init__(self, root=FEATURES):
        self.path = Path(root) / "meta.npz"
        m = np.load(self.path)
        self.y, self.pix = m["y"], m["pix"]
        self.patch_ids, self.offsets = m["patch_ids"], m["offsets"]
        self.pos = {int(p): i for i, p in enumerate(self.patch_ids)}
        self.sha256 = sha256(self.path)

    def rows(self, patches):
        """Row indices of all pixels of the given patches, in cache order."""
        idx = sorted(self.pos[int(p)] for p in patches)
        return np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]) if idx else \
            np.empty(0, np.int64)

    def patch_of_rows(self, rows):
        return self.patch_ids[np.searchsorted(self.offsets, rows, side="right") - 1]

    def rows_of_pixels(self, pixels):
        """Rows of {patch: [(row, col, class)]} pixels; checks the class against the cache labels."""
        out = []
        for p, pts in sorted(pixels.items()):
            i = self.pos[int(p)]
            a, b = self.offsets[i], self.offsets[i + 1]
            pix = self.pix[a:b]
            for r, c, k in pts:
                j = np.searchsorted(pix, r * SIZE + c)
                if j >= len(pix) or pix[j] != r * SIZE + c:
                    raise ValueError(f"Few-shot pixel {p}:{r},{c} is void or missing in the cache")
                if self.y[a + j] != k:
                    raise ValueError(f"Few-shot pixel {p}:{r},{c} has class {k} but the label is {self.y[a + j]}")
                out.append(a + j)
        return np.array(sorted(out), dtype=np.int64)


def confusion(y, p, n=N_CLASSES):
    """n x n confusion matrix (rows = truth, columns = prediction), int64."""
    return np.bincount(y.astype(np.int64) * n + p.astype(np.int64), minlength=n * n).reshape(n, n)


def scores(cm):
    """mIoU-8, OA and per-class IoU (in %) from one confusion matrix over all evaluated pixels (3.4). A class with
    no ground-truth pixels has IoU 0 in the mean, as the paper does (README.md:177-179)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt, pred = cm.sum(1), cm.sum(0)
    denom = gt + pred - tp
    iou = np.where((gt > 0) & (denom > 0), tp / np.maximum(denom, 1), 0.0)
    return {"miou8": float(100 * iou.mean()), "oa": float(100 * tp.sum() / max(cm.sum(), 1)),
            "iou": [float(100 * v) for v in iou], "absent_classes": [int(c) for c in np.flatnonzero(gt == 0)]}
