"""Location-encoder features for YieldSAT pixels (protocols/yieldsat.md, 2.5): MIND or SatCLIP.

A location encoder only sees (lat, lon), and at 10 m neighbouring pixels get practically the same embedding. So
pixels are snapped to a grid of GRID_DEG degrees (about 110 m x 70 m), each grid cell is embedded once, and every
pixel points to its cell. That keeps MIND's 3072-d output at ~1 GB instead of ~76 GB.

    python -m tools.embed_yieldsat_location --encoder mind      (GPU job; CPU works but is slow)

Needs features/yieldsat/latlon/ (tools.build_yieldsat_latlon). Writes features/yieldsat/<encoder>/:
    table.npy               float16 (cells, dim)   one embedding per occupied grid cell, all subsets together
    cells.npy               float64 (cells, 2)     the cell centres (lat, lon) that were embedded
    <subset>-index.npy      int32 (pixels,)        row of table.npy for each cache row of v1/<subset>.npy
    <subset>.json           kind "indexed", feature names, provenance
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from tools import yieldsat as ys

GRID_DEG = 0.001
ENCODERS = {
    "mind": {
        "source": ys.PROJECT / "upstream" / "mind",
        "source_revision": "eb25937f368679af8efafa3317f8b5a230d54c10",
        "weights": ys.PROJECT / "checkpoints" / "mind" / "mind.safetensors",
        "weights_repo": "taylor-geospatial/MIND@1437c4d85309e27b3aeca66f93bd5677e1a36ddf",
        "dim": 3072,
    },
    "satclip": {
        "source": ys.PROJECT / "upstream" / "satclip",
        "source_revision": "68c73233426d465bcdf5914abf13ebc88ba60f5e",
        "weights": ys.PROJECT / "checkpoints" / "satclip" / "satclip-vit16-l40.ckpt",
        "weights_repo": "microsoft/SatCLIP-ViT16-L40@0ef2acc0b91d3c8cfdf4a0cc03207095989287ab",
        "dim": 256,
    },
}


def mind_encoder(spec, device):
    sys.path.insert(0, str(spec["source"]))
    import mind_standalone as mind

    model = mind.load_mind(str(spec["weights"]), device=device)
    if model.use_year:
        raise ValueError("Expected the location-only MIND checkpoint")

    def encode(latlon):  # MIND takes (lat, lon) in degrees; full 3072-d trunk features
        return mind.embed(model, latlon[:, 0], latlon[:, 1], dim=None, device=device, batch_size=16384)

    return encode


def satclip_encoder(spec, device):
    # Import only the location encoder: the package __init__ pulls in training-only dependencies.
    pkg = types.ModuleType("satclip")
    pkg.__path__ = [str(spec["source"] / "satclip")]
    sys.modules["satclip"] = pkg
    from satclip.load_lightweight import get_satclip_loc_encoder

    model = get_satclip_loc_encoder(str(spec["weights"]), device).to(device)

    def encode(latlon):  # SatCLIP takes (lon, lat) in degrees, float64
        out = []
        with torch.inference_mode():
            for a in range(0, len(latlon), 65536):
                x = torch.as_tensor(latlon[a:a + 65536, ::-1].copy(), dtype=torch.float64, device=device)
                out.append(model(x).float().cpu().numpy())
        return np.concatenate(out)

    return encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    args = parser.parse_args()
    spec = ENCODERS[args.encoder]
    out_dir = ys.FEATURES / args.encoder
    if (out_dir / "table.npy").exists():
        raise SystemExit(f"{out_dir} already has a table; delete the folder to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snapped = {}
    for s in ys.SUBSETS:
        latlon = np.load(ys.FEATURES / "latlon" / f"{s}.npy")
        snapped[s] = np.round(latlon / GRID_DEG).astype(np.int64)
    cells, inverse = np.unique(np.concatenate(list(snapped.values())), axis=0, return_inverse=True)
    centres = cells * GRID_DEG
    print(f"{args.encoder}: {sum(len(v) for v in snapped.values())} pixels -> {len(cells)} grid cells", flush=True)

    encode = (mind_encoder if args.encoder == "mind" else satclip_encoder)(spec, device)
    table = encode(centres)
    if table.shape != (len(cells), spec["dim"]) or not np.isfinite(table).all():
        raise RuntimeError(f"Unexpected {args.encoder} output {table.shape}")
    np.save(out_dir / "table.npy", table.astype(np.float16))
    np.save(out_dir / "cells.npy", centres)

    pos = 0
    for s, v in snapped.items():
        idx = inverse[pos:pos + len(v)].astype(np.int32).ravel()
        pos += len(v)
        np.save(out_dir / f"{s}-index.npy", idx)
        info = {"subset": s, "kind": "indexed", "encoder": args.encoder, "rows": int(len(idx)),
                "feature_names": [f"{args.encoder}_{i:04d}" for i in range(spec["dim"])],
                "grid_deg": GRID_DEG, "cells_in_subset": int(len(np.unique(idx))),
                "aligned_with": f"v1/{s}-meta.npz",
                "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{s}-meta.npz"),
                "source": str(spec["source"]), "source_revision": spec["source_revision"],
                "weights": spec["weights_repo"], "weights_sha256": ys.sha256(spec["weights"])}
        (out_dir / f"{s}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps({k: info[k] for k in ("subset", "rows", "cells_in_subset")}), flush=True)


if __name__ == "__main__":
    main()
