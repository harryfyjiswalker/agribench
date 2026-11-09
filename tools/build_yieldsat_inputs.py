"""Build the YieldSAT pixel feature caches that every head reads (protocols/yieldsat.md, 2.5).

    python -m tools.build_yieldsat_inputs --country Germany     # one country (all its subsets)
    python -m tools.build_yieldsat_inputs                       # all four countries

For each country-crop subset, writes under $AGRIBENCH_PROJECT/features/yieldsat/v1/:
    <subset>.npy        float32 (pixels, 488) memory-mappable feature matrix, rows in file order (grouped by field)
    <subset>-meta.npz   per row: field index, target (t/ha), row, col, pixel UUID; plus the field names
    <subset>.json       feature names, column groups per input, source file and code provenance

Features per pixel: the 16 time-varying bands (12 Sentinel-2 + 4 weather) at all 24 steps, named <band>_t<step>,
then the 104 static bands (topography, soil, coordinates) once. S2 values of 65535 become NaN; missing steps stay
NaN. Input `s2` = the 288 S2 columns; input `s2-adm` = all 488 columns.
Run it in a Slurm job (it streams through files of up to 63 GB). Existing caches are never overwritten.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import h5py
import numpy as np

from tools import yieldsat as ys

VERSION = "v1"


def code_labels(h, name):
    """Integer code -> label table that xarray stored in the variable's attributes."""
    return {int(k): (v.decode() if isinstance(v, bytes) else str(v))
            for k, v in h[name].attrs.items() if k.isdigit()}


def build_country(country, out):
    path = ys.preprocessed_file(country)
    with h5py.File(path, "r") as h:
        bands = [b.decode() for b in h["band"][:]]
        if len(bands) != 120 or h["sample"].shape[1:] != (ys.N_STEPS, 120):
            raise ValueError(f"Unexpected layout in {path}: {h['sample'].shape}")
        t_idx = [bands.index(b) for b in ys.TEMPORAL_BANDS]
        s_names = [b for b in bands if b not in ys.TEMPORAL_BANDS]
        s_idx = [bands.index(b) for b in s_names]
        s2_pos = [i for i, b in enumerate(ys.TEMPORAL_BANDS) if b in ys.S2_BANDS]
        names = [f"{b}_t{s:02d}" for b in ys.TEMPORAL_BANDS for s in range(ys.N_STEPS)] + s_names
        s2_cols = [i for i, n in enumerate(names) if n.split("_t")[0] in ys.S2_BANDS and "_t" in n]

        code = h["field_shared_name"][:]
        if np.any(np.diff(code) < 0):
            raise ValueError("Rows are not grouped by field; this builder relies on contiguous fields")
        labels = code_labels(h, "field_shared_name")
        starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
        stops = np.r_[starts[1:], len(code)]
        field_rows = {labels[int(code[a])]: (int(a), int(b)) for a, b in zip(starts, stops)}

        by_subset = {}
        for field in field_rows:
            parts = field.split("_")
            if parts[0] != country or parts[-2] not in ys.CROPS:
                raise ValueError(f"Cannot read country/crop from field name {field}")
            by_subset.setdefault(ys.subset_name(country, parts[-2]), []).append(field)

        for subset, fields in sorted(by_subset.items()):
            dest = out / f"{subset}.npy"
            if dest.exists():
                print(f"{subset}: {dest} exists, skipping (caches are never overwritten)", flush=True)
                continue
            n = sum(field_rows[f][1] - field_rows[f][0] for f in fields)
            t0 = time.time()
            print(f"{subset}: {len(fields)} fields, {n:,} pixels x {len(names)} features", flush=True)
            fd, tmp = tempfile.mkstemp(dir=out, prefix=f".{subset}-", suffix=".npy")
            os.close(fd)
            x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, len(names)))
            field_idx = np.empty(n, np.int32)
            target = np.empty(n, np.float32)
            rowcol = np.empty((n, 2), np.int32)
            uuid = np.empty(n, "S36")
            pos = 0
            for i, f in enumerate(fields):
                a, b = field_rows[f]
                block = h["sample"][a:b]                       # (pixels, 24, 120)
                temporal = block[:, :, t_idx]                  # (pixels, 24, 16)
                s2 = temporal[:, :, s2_pos]
                s2[s2 == ys.S2_NODATA] = np.nan
                temporal[:, :, s2_pos] = s2
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices stay NaN
                    static = np.nanmax(block[:, :, s_idx], axis=1)   # constant over time
                k = b - a
                x[pos:pos + k, :len(ys.TEMPORAL_BANDS) * ys.N_STEPS] = temporal.transpose(0, 2, 1).reshape(k, -1)
                x[pos:pos + k, len(ys.TEMPORAL_BANDS) * ys.N_STEPS:] = static
                field_idx[pos:pos + k] = i
                target[pos:pos + k] = h["target"][a:b]
                rowcol[pos:pos + k, 0] = h["row"][a:b]
                rowcol[pos:pos + k, 1] = h["col"][a:b]
                uuid[pos:pos + k] = h["index"][a:b]
                pos += k
            x.flush()
            del x
            if pos != n or not np.isfinite(target).all():
                raise RuntimeError(f"{subset}: wrote {pos} of {n} rows, or the target has missing values")
            meta_tmp = tmp[:-4] + "-meta.npz"
            np.savez(meta_tmp, fields=np.array(fields), field_idx=field_idx, target=target,
                     row=rowcol[:, 0], col=rowcol[:, 1], pixel_uuid=uuid)
            info = {
                "version": VERSION, "subset": subset, "pixels": n, "fields": len(fields),
                "feature_names": names, "inputs": {"s2": s2_cols, "s2-adm": list(range(len(names)))},
                "source": {"file": str(path), "bytes": path.stat().st_size, "mtime": path.stat().st_mtime},
                "code": {f: ys.sha256(Path(__file__).parent / f) for f in ("yieldsat.py", "build_yieldsat_inputs.py")},
                "s2_nodata_to_nan": ys.S2_NODATA, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            for staged in (tmp, meta_tmp):
                os.chmod(staged, 0o664)
            os.replace(meta_tmp, out / f"{subset}-meta.npz")
            (out / f"{subset}.json").write_text(json.dumps(info, indent=1) + "\n")
            os.replace(tmp, dest)  # the .npy appears last, so its presence means the subset is complete
            print(f"{subset}: done in {time.time() - t0:.0f}s -> {dest}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", choices=sorted(ys.COUNTRIES), action="append",
                   help="country to build (repeatable); default all")
    p.add_argument("--out", type=Path, default=ys.FEATURES / VERSION)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for country in args.country or sorted(ys.COUNTRIES):
        build_country(country, args.out)


if __name__ == "__main__":
    sys.exit(main())
