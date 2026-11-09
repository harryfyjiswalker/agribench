"""Train and score field-delineation models on EarthShift (protocols/earthshift.md, 2.2-2.8).

    python -m tools.earthshift_train --model resnet50-imagenet --setting frozen --task de-geo \
        --mode sweep --lr 1e-4 3e-4 5e-4 --seeds 0 1 2 3 4
    python -m tools.earthshift_train --model galileo-base --setting full --task za-season \
        --mode final --lr 3e-3 --seeds 0 1 2 3 4 --confirm-test

Training tasks (each trains once and scores every test set it owns):
    de-geo     Germany train/val of de-kh/de-dk; scores DE ID test, Cambodia and Denmark (location shift)
    de-year    Germany 2018 bands -> ID test bands, OOD = Germany 2019 grid g1 (temporal shift)
    za-season  South Africa window A -> window B (temporal shift)
Modes:
    sweep   learning-rate selection: scores the ID validation set only (2.3)
    final   scores ID and OOD test sets; needs --confirm-test and appends to the test ledger (2.7)
Every (lr, seed) appends one JSON line to <out>/results.jsonl. --smoke uses a few samples and 1 epoch.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")  # raised inside SequentialLR

from tools import earthshift as es

RUNS = es.PROJECT / "runs" / "earthshift"
LEDGER = RUNS / "test-ledger.jsonl"
CHECKPOINTS = es.PROJECT / "checkpoints"

# Training rules from the paper (2.2): epochs per setting, batch 32, 20-epoch warm-up then cosine, AdamW, no wd.
EPOCHS = {"frozen": 40, "full": 50}
BATCH = 32
WARMUP_EPOCHS = 20
# Learning-rate grids (2.3).
LR_GRID = {"full": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3],
           "frozen": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2]}
# FTW statistics (Clay's Sentinel-2 values) on reflectance = DN / 10000, bands R, G, B, NIR (2.2).
FTW_MEAN = [0.1552, 0.1355, 0.1105, 0.2743]
FTW_STD = [0.1888, 0.1757, 0.1809, 0.1742]
# Middle calendar month (0 = January) of each window, from data_config_<country>.json (Galileo input only).
WINDOW_MONTH = {("germany", "a"): 7, ("germany", "b"): 4, ("south_africa", "a"): 7, ("south_africa", "b"): 4,
                ("cambodia", "a"): 9, ("cambodia", "b"): 5, ("denmark", "a"): 8, ("denmark", "b"): 3}

TRAIN_TASKS = {
    # name: (split file for train/val, {test name: (split file, role)})
    "de-geo": ("de-kh", {"id": ("de-kh", "id_test"), "ood_cambodia": ("de-kh", "ood_test"),
                         "ood_denmark": ("de-dk", "ood_test")}),
    "de-year": ("de-year", {"id": ("de-year", "id_test"), "ood": ("de-year", "ood_test")}),
    "za-season": ("za-season", {"id": ("za-season", "id_test"), "ood": ("za-season", "ood_test")}),
}


# ----------------------------------------------------------------------------------------------- data

class Cache:
    """The packed per-country input cache (section 0), loaded into memory once."""

    def __init__(self, root):
        self.root, self.images, self.masks, self.rows = Path(root), {}, {}, {}

    def load(self, country):
        if country not in self.images:
            self.images[country] = np.load(self.root / f"{country}-images.npy", mmap_mode="r")
            self.masks[country] = np.load(self.root / f"{country}-masks.npy", mmap_mode="r")
            idx = pd.read_csv(self.root / f"{country}-index.csv")
            self.rows[country] = {(a, w): r for r, a, w in zip(idx.row, idx.aoi_id, idx.window)}
        return self

    def gather(self, samples):
        """uint16 images (N, 4, 256, 256), uint8 masks (N, 256, 256), months (N,) for a sample table."""
        imgs, masks, months = [], [], []
        for country, g in samples.groupby("country", sort=False):
            self.load(country)
            rows = np.array([self.rows[country][(a, w)] for a, w in zip(g.aoi_id, g.window)])
            order = np.argsort(rows)  # sorted reads from the memory map, then back to table order
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            imgs.append(np.asarray(self.images[country][rows[order]])[inv])
            masks.append(np.asarray(self.masks[country][rows[order]])[inv])
            months.append(np.array([WINDOW_MONTH[(country, w)] for w in g.window]))
        return np.concatenate(imgs), np.concatenate(masks), np.concatenate(months)


def load_split(name, role):
    d = pd.read_csv(es.SPLITS / f"{name}.csv")
    return d[d.role == role].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- models

class SimpleDecoder(nn.Module):
    """The paper's head: 3x3 conv (256) -> ReLU -> 1x1 conv (2.2; EarthShift utils.py:99-108)."""

    def __init__(self, in_channels, n_classes=3):
        super().__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, n_classes, 1))

    def forward(self, x):
        return self.decode(x)


class ResNet50(nn.Module):
    """torchvision ResNet-50, stride-32 features; learnable 1x1 adapter 4 -> 3 bands (2.2, 2.5)."""
    input_size = 224

    def __init__(self, pretrained):
        super().__init__()
        import torchvision
        os.environ.setdefault("TORCH_HOME", str(CHECKPOINTS / "torch"))
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None  # as EarthShift
        net = torchvision.models.resnet50(weights=weights)
        self.adapter = nn.Conv2d(4, 3, 1, bias=False)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.out_channels = 2048
        self.register_buffer("mean", torch.tensor(FTW_MEAN).view(1, 4, 1, 1))
        self.register_buffer("std", torch.tensor(FTW_STD).view(1, 4, 1, 1))

    def prepare(self, dn, months):
        return (dn / 10000.0 - self.mean) / self.std

    def features(self, x, months):
        return self.backbone(self.adapter(x))


class Galileo(nn.Module):
    """Galileo Base, full encoder, per-patch features (patch 8 -> 28 x 28 at 224 px).

    R, G, B, NIR go straight into Galileo's own S2 bands (B4, B3, B2, B8) and NDVI is computed from them, with
    Galileo's normalisation; the other band groups are masked. No adapter is needed (2.5).
    """
    input_size = 224
    patch = 8

    def __init__(self):
        super().__init__()
        src = es.PROJECT / "upstream" / "galileo"
        sys.path.insert(0, str(src))
        import single_file_galileo as g
        self.g = g
        self.encoder = g.Encoder.load_from_folder(CHECKPOINTS / "galileo" / "base", torch.device("cpu"))
        self.out_channels = self.encoder.embedding_size
        bands = g.SPACE_TIME_BANDS
        self.idx = [bands.index(b) for b in ("B4", "B3", "B2", "B8")]  # R, G, B, NIR
        self.ndvi = bands.index("NDVI")
        stats = json.loads((src / "config" / "normalization.json").read_text())[str(len(bands))]
        mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        self.register_buffer("shift", torch.tensor(mean - 2 * std))
        self.register_buffer("scale", torch.tensor(4 * std))
        groups = list(g.SPACE_TIME_BANDS_GROUPS_IDX)
        self.st_keep = [groups.index(k) for k in ("S2_RGB", "S2_NIR_10m", "NDVI")]
        self.n_st_groups = len(groups)

    def prepare(self, dn, months):
        b, _, h, w = dn.shape
        x = torch.zeros(b, h, w, 1, len(self.g.SPACE_TIME_BANDS), device=dn.device)
        x[..., 0, self.idx] = dn.permute(0, 2, 3, 1)
        red, nir = dn[:, 0], dn[:, 3]
        x[..., 0, self.ndvi] = (nir - red) / (nir + red).clamp(min=1e-6)
        return (x - self.shift) / self.scale

    def features(self, x, months):
        g, b, h, w = self.g, x.shape[0], x.shape[1], x.shape[2]
        dev = x.device
        st_m = torch.ones(b, h, w, 1, self.n_st_groups, device=dev)
        st_m[..., self.st_keep] = 0
        out = self.encoder(
            x, torch.zeros(b, h, w, len(g.SPACE_BANDS), device=dev),
            torch.zeros(b, 1, len(g.TIME_BANDS), device=dev), torch.zeros(b, len(g.STATIC_BANDS), device=dev),
            st_m, torch.ones(b, h, w, len(g.SPACE_BAND_GROUPS_IDX), device=dev),
            torch.ones(b, 1, len(g.TIME_BAND_GROUPS_IDX), device=dev),
            torch.ones(b, len(g.STATIC_BAND_GROUPS_IDX), device=dev),
            months.view(b, 1), patch_size=self.patch)
        tokens = g.Encoder.apply_mask_and_average_tokens_per_patch(*out[:8])  # (b, h/p * w/p, d)
        hp, wp = h // self.patch, w // self.patch
        return tokens.transpose(1, 2).reshape(b, -1, hp, wp)


MODELS = {
    "resnet50-imagenet": lambda: ResNet50(pretrained=True),
    "resnet50-random": lambda: ResNet50(pretrained=False),
    "galileo-base": Galileo,
}


class Segmenter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.decoder = SimpleDecoder(backbone.out_channels)

    def forward(self, x, months):
        logits = self.decoder(self.backbone.features(x, months))
        return F.interpolate(logits, size=x.shape[-2:] if x.dim() == 4 else x.shape[1:3], mode="bilinear",
                             align_corners=False)


def set_trainable(model, setting):
    for p in model.parameters():
        p.requires_grad = setting == "full"
    if setting == "frozen":  # head and input adapter train; backbone weights fixed (2.2, 2.5)
        for p in model.decoder.parameters():
            p.requires_grad = True
        adapter = getattr(model.backbone, "adapter", None)
        if adapter is not None:
            for p in adapter.parameters():
                p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


# ------------------------------------------------------------------------------------------- training

def resize(images_dn, masks, size):
    """Images: bilinear with antialias (torchvision Resize); masks: nearest (2.2)."""
    x = F.interpolate(images_dn.float(), size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    m = F.interpolate(masks[:, None].float(), size=(size, size), mode="nearest")[:, 0].to(torch.uint8)
    return x, m


class Confusion:
    """Pixel confusion matrix summed over a whole test set; IoU per class, macro mean (2.4)."""

    def __init__(self, device, k=3):
        self.k, self.m = k, torch.zeros(k, k, dtype=torch.long, device=device)

    def update(self, pred, target):
        ok = target < self.k
        self.m += torch.bincount(target[ok].long() * self.k + pred[ok].long(), minlength=self.k ** 2).view(self.k, self.k)

    def iou(self):
        m = self.m.double()
        tp = m.diag()
        union = m.sum(0) + m.sum(1) - tp
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.nan)
        return {"miou": float(torch.nanmean(iou)), "iou": [float(v) for v in iou]}


def evaluate(model, data, device, micro):
    """mIoU at the model's input size (headline) and at native 256 x 256 (secondary)."""
    model.eval()
    size = model.backbone.input_size
    resized, native = Confusion(device), Confusion(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, len(data["masks"]), micro):
            dn = torch.as_tensor(data["images"][s:s + micro].astype(np.float32), device=device)
            mk = torch.as_tensor(data["masks"][s:s + micro], device=device)
            months = torch.as_tensor(data["months"][s:s + micro], device=device)
            x, m = resize(dn, mk, size)
            logits = model(model.backbone.prepare(x, months), months).float()
            resized.update(logits.argmax(1), m)
            up = F.interpolate(logits, size=mk.shape[-2:], mode="bilinear", align_corners=False)
            native.update(up.argmax(1), mk)
    r, n = resized.iou(), native.iou()
    return {"miou": r["miou"], "iou": r["iou"], "miou_native": n["miou"], "iou_native": n["iou"],
            "boundary_iou_native": n["iou"][2], "samples": int(len(data["masks"]))}


def train_one(args, model_name, setting, lr, seed, train, val, tests, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Segmenter(MODELS[model_name]()).to(device)
    params = set_trainable(model, setting)
    size = model.backbone.input_size
    # Resize the training set once and keep it on the GPU (small: under 0.5 GB).
    x_all, m_all = [], []
    for s in range(0, len(train["masks"]), 256):
        x, m = resize(torch.as_tensor(train["images"][s:s + 256].astype(np.float32), device=device),
                      torch.as_tensor(train["masks"][s:s + 256], device=device), size)
        x_all.append(x)  # float32: reflectance DN would lose precision in bf16
        m_all.append(m)
    x_all, m_all = torch.cat(x_all), torch.cat(m_all)
    months_all = torch.as_tensor(train["months"], device=device)

    epochs = 1 if args.smoke else EPOCHS[setting]
    steps_per_epoch = math.ceil(len(m_all) / BATCH)
    warm = min(WARMUP_EPOCHS, epochs) * steps_per_epoch
    decay = max(epochs * steps_per_epoch - warm, 1)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay)], milestones=[warm])
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()  # whole model in train mode, as the paper's code does (2.2)
        perm = torch.randperm(len(m_all), generator=gen).to(device)
        total = 0.0
        for s in range(0, len(perm), BATCH):
            batch = perm[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            for c in range(0, len(batch), args.micro_batch):  # gradient accumulation; same effective batch
                mb = batch[c:c + args.micro_batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    x = model.backbone.prepare(x_all[mb], months_all[mb])
                    logits = model(x, months_all[mb])
                    loss = loss_fn(logits.float(), m_all[mb].long()) * len(mb) / len(batch)
                loss.backward()
                total += float(loss.detach()) * len(batch)
            opt.step()
            sched.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"    epoch {epoch + 1}/{epochs} loss {total / len(m_all):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{time.time() - t0:.0f} s", flush=True)
    del x_all, m_all
    result = {"val": evaluate(model, val, device, args.micro_batch * 2)}
    for name, data in tests.items():
        result[name] = evaluate(model, data, device, args.micro_batch * 2)
    result["train_seconds"] = round(time.time() - t0, 1)
    return result


# ------------------------------------------------------------------------------------------------ main

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--setting", choices=EPOCHS, required=True)
    p.add_argument("--task", choices=TRAIN_TASKS, required=True)
    p.add_argument("--mode", choices=["sweep", "final"], required=True)
    p.add_argument("--lr", type=float, nargs="+", help="default: the whole grid for this setting (2.3)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--micro-batch", type=int, default=32, help="split each batch of 32 to save GPU memory")
    p.add_argument("--confirm-test", action="store_true", help="required for --mode final")
    p.add_argument("--smoke", action="store_true", help="few samples, 1 epoch; scores are meaningless")
    p.add_argument("--cache", type=Path, default=es.INPUTS / "v1")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.mode == "final" and not (args.confirm_test or args.smoke):
        sys.exit("--mode final scores the test sets; add --confirm-test (it is logged in the ledger)")
    lrs = args.lr or LR_GRID[args.setting]
    if not args.smoke and any(lr not in LR_GRID[args.setting] for lr in lrs):
        sys.exit(f"learning rates must come from the {args.setting} grid {LR_GRID[args.setting]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    fit_split, test_defs = TRAIN_TASKS[args.task]
    tables = {"train": load_split(fit_split, "train"), "val": load_split(fit_split, "val")}
    if args.mode == "final":
        tables.update({n: load_split(f, r) for n, (f, r) in test_defs.items()})
    if args.smoke:
        tables = {k: v.groupby("country", group_keys=False).head(40 if k == "train" else 8) for k, v in tables.items()}
    data = {}
    for k, t in tables.items():
        imgs, masks, months = cache.gather(t)
        data[k] = {"images": imgs, "masks": masks, "months": months}
    print(f"{args.model} {args.setting} {args.task} {args.mode}: " +
          ", ".join(f"{k} {len(v['masks'])}" for k, v in data.items()), flush=True)

    job = os.environ.get("SLURM_JOB_ID", "local")
    out = args.out or RUNS / os.environ.get("USER", "user") / f"{time.strftime('%Y%m%d-%H%M%S')}-{job}"
    out.mkdir(parents=True, exist_ok=True)
    split_hashes = {f: es.sha256(es.SPLITS / f"{f}.csv") for f in sorted({fit_split, *(f for f, _ in test_defs.values())})}
    base = {"model": args.model, "setting": args.setting, "task": args.task, "mode": args.mode,
            "smoke": args.smoke, "epochs": 1 if args.smoke else EPOCHS[args.setting], "batch": BATCH,
            "split_sha256": split_hashes, "splits_manifest_sha256": es.sha256(es.SPLITS / "manifest.json"),
            "code_sha256": es.sha256(Path(__file__)), "git_commit": git_commit(), "job": job,
            "torch": torch.__version__, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}
    tests = {k: v for k, v in data.items() if k not in ("train", "val")}
    for lr in lrs:
        for seed in args.seeds:
            print(f"  lr {lr:g} seed {seed}", flush=True)
            r = train_one(args, args.model, args.setting, lr, seed, data["train"], data["val"], tests, device)
            row = {**base, "lr": lr, "seed": seed, **r, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            line = json.dumps(row) + "\n"
            with open(out / "results.jsonl", "a") as f:
                f.write(line)
            if args.mode == "final" and not args.smoke:
                LEDGER.parent.mkdir(parents=True, exist_ok=True)
                with open(LEDGER, "a") as f:
                    f.write(line)
            summary = "  ".join(f"{k} {v['miou'] * 100:.2f}" for k, v in r.items() if isinstance(v, dict))
            print(f"  -> mIoU {summary}  ({r['train_seconds']:.0f} s)", flush=True)
    print(f"Wrote {out / 'results.jsonl'}")


if __name__ == "__main__":
    main()
