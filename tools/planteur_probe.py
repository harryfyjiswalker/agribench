"""PLANTEUR linear probe, as the paper trains it (protocols/planteur.md 3.5-3.6, D4).

One linear layer per pixel, cross-entropy without class weights on every training pixel, 50 epochs, AdamW, batches of
4,096, final-epoch weights. The learning rate comes from np.geomspace(1e-4, 0.5, 8) and is chosen by mIoU-8 on the
fold's val pixels. Features are standardised with the training pixels' mean and std (missing values -> the mean,
z-scores clipped to +/-10, as tools/probe.py does).

All eight learning rates train at once, as eight independent heads that see the same batches: one forward pass gives
all eight losses, and each head has its own AdamW parameter group. That is the same as eight separate runs with one
data order (seed 0), at about the cost of one. The standardised training table is kept on the GPU in float16
(31 M x 128 is 8 GB) when it fits, else in pinned host memory.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

Z_CLIP = 10.0
CHUNK = 1 << 20


class Standardiser:
    """Mean/std of the training rows (NaN ignored), fitted in float64 chunks."""

    def fit(self, x):
        d = x.shape[1]
        s, sq, n = np.zeros(d), np.zeros(d), np.zeros(d)
        for a in range(0, len(x), CHUNK):
            c = np.asarray(x[a:a + CHUNK], dtype=np.float64)
            ok = np.isfinite(c)
            c = np.where(ok, c, 0)
            s += c.sum(0)
            sq += (c * c).sum(0)
            n += ok.sum(0)
        self.mean = s / np.maximum(n, 1)
        std = np.sqrt(np.maximum(sq / np.maximum(n, 1) - self.mean ** 2, 0))
        std[std < 1e-12] = 1.0
        self.std = std
        return self

    def transform(self, c):
        z = (np.asarray(c, dtype=np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        z[~np.isfinite(z)] = 0.0
        return np.clip(z, -Z_CLIP, Z_CLIP)


def to_device_table(x, scaler, device):
    """Standardised float16 copy of x on the GPU if it fits (with headroom), else pinned on the host."""
    n, d = x.shape
    need = n * d * 2
    on_gpu = device.type == "cuda" and torch.cuda.mem_get_info(device)[0] > need * 1.25 + (2 << 30)
    out = torch.empty((n, d), dtype=torch.float16, device=device if on_gpu else "cpu",
                      pin_memory=(not on_gpu and device.type == "cuda"))
    for a in range(0, n, CHUNK):
        out[a:a + CHUNK] = torch.from_numpy(scaler.transform(x[a:a + CHUNK]).astype(np.float16)).to(out.device)
    return out, on_gpu


@torch.no_grad()
def predict(heads, x, scaler, device, batch=1 << 18):
    """argmax class per row for each head: (n_heads, n) uint8. x is a host array (rows, d)."""
    w = torch.stack([h[0] for h in heads])  # (H, d, C)
    b = torch.stack([h[1] for h in heads])  # (H, C)
    out = np.empty((len(heads), len(x)), dtype=np.uint8)
    for a in range(0, len(x), batch):
        z = torch.from_numpy(scaler.transform(x[a:a + batch])).to(device)
        logits = torch.einsum("nd,hdc->hnc", z, w) + b[:, None]
        logits = torch.nan_to_num(logits, nan=-1e30)
        out[:, a:a + batch] = logits.argmax(-1).to(torch.uint8).cpu().numpy()
    return out


def train(x, y, n_classes, lrs, epochs=50, batch=4096, seed=0, weight_decay=0.01, log=print):
    """Train one linear head per learning rate on (x, y). Returns (heads, scaler, info)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    scaler = Standardiser().fit(x)
    table, on_gpu = to_device_table(x, scaler, device)
    labels = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(table.device)
    n, d = table.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    heads = []
    for _ in lrs:  # nn.Linear's default initialisation, one generator for all heads
        bound = 1 / np.sqrt(d)
        w = (torch.rand((d, n_classes), generator=g) * 2 - 1) * bound
        bb = (torch.rand(n_classes, generator=g) * 2 - 1) * bound
        heads.append((w.to(device).requires_grad_(), bb.to(device).requires_grad_()))
    opt = torch.optim.AdamW([{"params": list(h), "lr": float(lr)} for h, lr in zip(heads, lrs)],
                            weight_decay=weight_decay)
    log(f"probe: {n:,} rows x {d} on {'GPU' if on_gpu else 'host'} ({time.time() - t0:.0f}s to stage); "
        f"{len(lrs)} learning rates, {epochs} epochs, batch {batch}")
    gperm = torch.Generator(device=table.device).manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gperm, device=table.device)
        total = torch.zeros(len(lrs), device=device)
        for a in range(0, n, batch):
            idx = perm[a:a + batch]
            xb = table[idx].to(device, non_blocking=True).float()
            yb = labels[idx].to(device, non_blocking=True)
            w = torch.stack([h[0] for h in heads])
            b = torch.stack([h[1] for h in heads])
            logits = torch.einsum("nd,hdc->hnc", xb, w) + b[:, None]
            loss = F.cross_entropy(logits.reshape(-1, n_classes), yb.repeat(len(lrs)), reduction="none")
            loss = loss.view(len(lrs), -1).mean(1)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            total += loss.detach() * len(idx)
        losses.append((total / n).tolist())
        if epoch in (0, epochs - 1) or (epoch + 1) % 10 == 0:
            log(f"  epoch {epoch + 1}: loss " + " ".join(f"{v:.4f}" for v in losses[-1]) +
                f" ({time.time() - t0:.0f}s)")
    del table, labels
    heads = [(w.detach(), b.detach()) for w, b in heads]
    return heads, scaler, {"device": str(device), "table_on_gpu": on_gpu, "final_loss": losses[-1],
                           "train_seconds": round(time.time() - t0, 1)}
