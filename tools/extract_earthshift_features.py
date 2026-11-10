"""Cache frozen-backbone features for EarthShift (protocols/earthshift.md, 4c).

    sbatch -c 16 --mem 32G -t 6:00:00 prepare.sbatch tools.extract_earthshift_features --model galileo-base

Runs the frozen backbone once over every sample in the packed input cache (all used chips and windows), at the
model's resized input size, in eval mode and fp32, and writes per country:
    <out>/<country>-features.npy   float32 (N, C, h, w), rows in the same order as <country>-index.csv
    <out>/manifest.json            model, shapes, hashes of code, inputs and weights, git commit
Only models without a trainable input adapter can be cached (decision 6.6); the decoder then trains on CPU with
`tools.earthshift_train --features`. Features stay on Lustre, outside the repo.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tools import earthshift as es
from tools.earthshift_train import CACHED_MODELS, FEATURES, Cache, git_commit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=CACHED_MODELS, required=True)
    p.add_argument("--cache", type=Path, default=es.INPUTS / "v1")
    p.add_argument("--out", type=Path, help="default: inputs/earthshift/features-v1/<model>")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--countries", nargs="+", default=list(es.COUNTRIES))
    p.add_argument("--smoke", action="store_true", help="first 16 samples per country only (rest left as zeros)")
    args = p.parse_args()
    out = args.out or FEATURES / args.model
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already has a manifest; features are written once")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    backbone = CACHED_MODELS[args.model]().to(device).eval()
    size = backbone.input_size
    cache = Cache(args.cache)
    shapes = {}
    for country in args.countries:
        cache.load(country)
        images = cache.images[country]
        n = 16 if args.smoke else len(images)
        idx = pd.read_csv(args.cache / f"{country}-index.csv")
        months = np.array([cache.month(country, w) for w in idx.window[:n]])
        t0, arr, tmp = time.time(), None, out / f"{country}-features.tmp.npy"
        with torch.no_grad():
            for s in range(0, n, args.batch):
                dn = torch.as_tensor(np.asarray(images[s:min(s + args.batch, n)]).astype(np.float32), device=device)
                x = F.interpolate(dn, size=(size, size), mode="bilinear", antialias=True, align_corners=False)  # = resize()
                mo = torch.as_tensor(months[s:s + len(dn)], device=device)
                f = backbone.features(backbone.prepare(x, mo), mo).float().cpu().numpy()
                if arr is None:  # smoke: full length so row indices line up, zeros after the first n rows
                    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                                    shape=(len(images), *f.shape[1:]))
                arr[s:s + len(f)] = f
                if s == 0 or (s // args.batch) % 25 == 0:
                    print(f"  {country} {s + len(f)}/{n}  {time.time() - t0:.0f} s", flush=True)
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite features for {country}")
        arr.flush()
        del arr
        tmp.rename(out / f"{country}-features.npy")
        shapes[country] = [len(images), *np.load(out / f"{country}-features.npy", mmap_mode="r").shape[1:]]
        print(f"{country}: {shapes[country]} in {time.time() - t0:.0f} s", flush=True)
    manifest = {
        "model": args.model, "input_size": size, "dtype": "float32", "backbone_mode": "eval", "smoke": args.smoke,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "torch": torch.__version__,
        "shapes": shapes, "git_commit": git_commit(),
        "code_sha256": {f: es.sha256(Path(__file__).parent / f) for f in ("extract_earthshift_features.py",
                                                                          "earthshift_train.py")},
        "inputs_sha256": {c: es.sha256(args.cache / f"{c}-index.csv") for c in args.countries},
        "weights_sha256": backbone.weights_sha256(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
