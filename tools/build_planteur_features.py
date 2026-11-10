"""Build the PLANTEUR per-pixel cache (protocols/planteur.md 3.6) from the downloaded patches. Runs as a Slurm job:

    sbatch --cpus-per-task=8 --mem=16G prepare.sbatch tools.build_planteur_features --input meta
    sbatch --cpus-per-task=8 --mem=16G prepare.sbatch tools.build_planteur_features --input emb-aef
    ... --input all     # meta, then every input in tools/planteur.INPUTS

`meta` (the row layout and labels, from ANNOTATIONS/) must exist before any input. Every input is one table over
all 2,750 patches, aligned with meta.npz, so the default split and the five LOIO folds select rows from the same
table instead of each keeping a copy. Rows are the non-void pixels only (void is never trained on or scored).

  s2-monthly          12 months x 10 bands of DATA_S2_MONTHLY (B01 and B09 dropped), uint16, 0 = no data. Columns
                      are month-major (m00_B02 ... m11_B12), so the first k months are the first 10k columns.
  emb-*               the shipped EMB_* vector, stored as float16 (the embeddings are small numbers; the builder
                      checks the range). valid = finite and not all zero (D7); a patch without a file is invalid.

The shipped NORM_* files are never read (D3): statistics are fitted per fold by the runner.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tools import planteur as pl

CHUNK = 64  # patches read in parallel at a time (bounded memory)


def patch_ids():
    s = json.loads((pl.DATA / "splits" / "443.json").read_text())["splits"]
    ids = sorted(int(i) for k in ("train", "val", "test") for i in s[k])
    if len(ids) != len(set(ids)):
        raise ValueError("Patch listed twice in splits/443.json")
    return ids


def chunks(seq, n):
    for a in range(0, len(seq), n):
        yield seq[a:a + n]


def build_meta(workers):
    ids = patch_ids()

    def read(p):
        t = np.load(pl.DATA / "ANNOTATIONS" / f"TARGET_{p}.npy")
        if t.shape != (1, pl.SIZE, pl.SIZE) or t.dtype != np.uint8:
            raise ValueError(f"TARGET_{p}.npy: unexpected {t.shape} {t.dtype}")
        t = t[0].ravel()
        keep = np.flatnonzero(t != pl.VOID)
        if (t[keep] >= pl.N_CLASSES).any():
            raise ValueError(f"TARGET_{p}.npy: class id outside 0-7")
        return t[keep], keep.astype(np.uint16)

    ys, pixs = [], []
    with ThreadPoolExecutor(workers) as ex:
        for part in chunks(ids, CHUNK):
            for y, pix in ex.map(read, part):
                ys.append(y)
                pixs.append(pix)
    offsets = np.r_[0, np.cumsum([len(y) for y in ys])].astype(np.int64)
    y, pix = np.concatenate(ys), np.concatenate(pixs)
    pl.FEATURES.mkdir(parents=True, exist_ok=True)
    tmp = pl.FEATURES / ".meta.tmp.npz"
    np.savez(tmp, y=y, pix=pix, patch_ids=np.array(ids, dtype=np.int32), offsets=offsets)
    tmp.rename(pl.FEATURES / "meta.npz")
    void = len(ids) * pl.SIZE ** 2 - len(y)
    counts = np.bincount(y, minlength=pl.N_CLASSES)
    print(json.dumps({"patches": len(ids), "rows": int(len(y)), "void_pixels": int(void),
                      "class_pixels": {pl.CLASSES[c]: int(n) for c, n in enumerate(counts)}}, indent=2))


def build_input(name, workers):
    spec = pl.INPUTS[name]
    meta = pl.Meta()
    folder = pl.DATA / spec["folder"]
    out = pl.FEATURES / name
    out.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "s2":
        names = [f"m{m:02d}_{b}" for m in range(pl.N_MONTHS) for b in pl.S2_BANDS]
        dtype, nodata = np.uint16, 0
    else:
        first = np.load(folder / f"{spec['prefix']}_{int(meta.patch_ids[0])}.npy", mmap_mode="r")
        names = [f"e{c:03d}" for c in range(first.shape[0])]
        dtype, nodata = np.float16, None
    d = len(names)
    rows = len(meta.y)
    x = np.lib.format.open_memmap(out / ".X.tmp.npy", mode="w+", dtype=dtype, shape=(rows, d))
    valid = np.ones(rows, dtype=bool)
    stats = {"missing_files": [], "invalid_pixels": 0, "patches_with_invalid": 0, "max_abs": 0.0}

    def read(i):
        p = int(meta.patch_ids[i])
        pix = meta.pix[meta.offsets[i]:meta.offsets[i + 1]]
        f = folder / f"{spec['prefix']}_{p}.npy"
        if not f.exists():
            return i, None, None
        a = np.load(f)
        if spec["kind"] == "s2":
            if a.shape != (pl.N_MONTHS, 12, pl.SIZE, pl.SIZE) or a.dtype != np.uint16:
                raise ValueError(f"{f.name}: unexpected {a.shape} {a.dtype}")
            v = a[:, :10].reshape(pl.N_MONTHS * 10, -1)[:, pix].T
            return i, np.ascontiguousarray(v), np.ones(len(pix), bool)
        if a.shape != (d, pl.SIZE, pl.SIZE):
            raise ValueError(f"{f.name}: unexpected shape {a.shape}")
        v = a.reshape(d, -1)[:, pix].T.astype(np.float32)
        ok = np.isfinite(v).all(1) & (v != 0).any(1)
        v[~np.isfinite(v)] = 0
        return i, v, ok

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for part in chunks(range(len(meta.patch_ids)), CHUNK):
            for i, v, ok in ex.map(read, part):
                a, b = meta.offsets[i], meta.offsets[i + 1]
                if v is None:
                    stats["missing_files"].append(int(meta.patch_ids[i]))
                    x[a:b] = 0
                    valid[a:b] = False
                    continue
                if spec["kind"] == "emb":
                    m = float(np.abs(v).max()) if len(v) else 0.0
                    stats["max_abs"] = max(stats["max_abs"], m)
                    if m > 6e4:
                        raise ValueError(f"{name} patch {meta.patch_ids[i]}: |value| {m} overflows float16")
                    stats["patches_with_invalid"] += int((~ok).any())
                x[a:b] = v.astype(dtype)
                valid[a:b] = ok
    x.flush()
    del x
    stats["invalid_pixels"] = int((~valid).sum())
    np.save(out / ".valid.tmp.npy", valid)
    (out / ".X.tmp.npy").rename(out / "X.npy")
    (out / ".valid.tmp.npy").rename(out / "valid.npy")
    info = {"input": name, "kind": spec["kind"], "source": spec["folder"], "rows": rows, "dtype": np.dtype(dtype).name,
            "nodata": nodata, "feature_names": names, "meta_sha256": meta.sha256,
            "dataset_revision": "51b74a91424e7dc9ea1fe0042390353658af8185",
            "manifest_sha256": pl.sha256(pl.DATA / "MANIFEST.sha256") if (pl.DATA / "MANIFEST.sha256").exists() else None,
            "seconds": round(time.time() - t0, 1), **stats}
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "feature_names"}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, choices=["meta", "all", *pl.INPUTS])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.input in ("meta", "all"):
        build_meta(args.workers)
    for name in (pl.INPUTS if args.input == "all" else [args.input] if args.input != "meta" else []):
        build_input(name, args.workers)


if __name__ == "__main__":
    main()
