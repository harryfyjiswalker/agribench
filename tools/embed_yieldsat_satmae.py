"""Frozen SatMAE (multispectral, ViT-L/8) patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

SatMAE (Cong et al. 2022, arXiv:2207.08051) is a masked autoencoder for multispectral imagery. Bands are split into
groups with their own patch embedding and a group (spectral) encoding, and groups are masked independently. The
checkpoint is the fMoW-Sentinel ViT-Large pretraining (200 epochs, zenodo 7338613, CC BY-NC 4.0). Input as in its
pretraining: 10 bands (B02-B08, B8A, B11, B12; B01, B09, B10 dropped), groups (B02 B03 B04 B08), (B05 B06 B07 B8A),
(B11 B12), scaled with SatMAE's SentinelNormalize, (x - (mean - 2 std)) / (4 std) clipped to [0, 1], fMoW-Sentinel
statistics; 96 px windows (its pretraining size) and 8 px tokens (12 x 12 grid, 80 m). Per position, the three
group tokens after the final norm are averaged into one 1024-d vector.

The encoder is re-implemented here from upstream models_mae_group_channels.forward_encoder (pinned at
upstream/satmae/src, commit 0b210ac) with no masking, on timm Blocks. The checkpoint stores its training args as an
argparse.Namespace, so that one class is allow-listed for the tensors-only load.

    python -m tools.embed_yieldsat_satmae --subset GER-W
"""
import argparse
from functools import partial

import numpy as np
import torch
from torch import nn

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "satmae" / "weights" / "pretrain-vit-large-e199.pth"
SOURCE_REVISION = "sustainlab-group/SatMAE@0b210aceb37a14bbbd897110db5b104b3271d818"
SIZE, TOKEN = 96, 8
BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]           # RAW_S2_BANDS indices: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))      # upstream --grouped_bands 0 1 2 6 / 3 4 5 7 / 8 9
# fMoW-Sentinel statistics (upstream util/datasets.py SentinelIndividualImageDataset), 13-band order without
# B01 (0), B09 (9), B10 (10)
_MEAN13 = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144, 1645.40315151, 1846.87040806,
           1762.59530783, 1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
_STD13 = [633.15169573, 650.2842772, 712.12507725, 965.23119807, 948.9819932, 1108.06650639, 1258.36394548,
          1233.1492281, 1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]
KEEP13 = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
MEAN = np.array([_MEAN13[i] for i in KEEP13], np.float32)
STD = np.array([_STD13[i] for i in KEEP13], np.float32)


class GroupChannelEncoder(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, channel_embed=256):
        from timm.layers import PatchEmbed
        from timm.models.vision_transformer import Block
        super().__init__()
        self.patch_embed = nn.ModuleList([PatchEmbed(SIZE, TOKEN, len(g), dim) for g in GROUPS])
        n = (SIZE // TOKEN) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim - channel_embed))
        self.channel_embed = nn.Parameter(torch.zeros(1, len(GROUPS), channel_embed))
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, qkv_bias=True, norm_layer=norm) for _ in range(depth)])
        self.norm = norm(dim)

    def forward(self, x):
        b = x.shape[0]
        x = torch.stack([pe_(x[:, list(g)]) for pe_, g in zip(self.patch_embed, GROUPS)], 1)   # (B, G, L, D)
        _, G, L, D = x.shape
        pos = self.pos_embed[:, 1:].unsqueeze(1).expand(-1, G, -1, -1)
        chan = self.channel_embed.unsqueeze(2).expand(-1, -1, L, -1)
        x = (x + torch.cat([pos, chan], -1)).reshape(b, G * L, D)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 1:].reshape(b, G, L, D)
        return x.mean(1)                                                                       # (B, L, D)


def make_model():
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["model"]
    enc = {k: v for k, v in state.items() if not k.startswith("decoder") and k != "mask_token"}
    net = GroupChannelEncoder()
    net.load_state_dict(enc, strict=True)
    net = net.to(device).eval()

    def prepare(img):
        x = img[BANDS]
        lo = (MEAN - 2 * STD)[:, None, None]
        return np.clip((x - lo) / (4 * STD)[:, None, None], 0, 1).astype(np.float32)

    def encode(x, dates):
        return pe.tokens_to_grid(net(x), SIZE // TOKEN)

    return pe.Model("satmae", 1024, SIZE, TOKEN, prepare, encode, {
        "model": "SatMAE multispectral ViT-L/8 (fMoW-Sentinel pretraining, group channels)",
        "revision": {"source": SOURCE_REVISION, "timm": timm.__version__,
                     "weights": "zenodo 7338613 pretrain-vit-large-e199.pth (CC BY-NC 4.0)",
                     "weights_sha256": pe.weights_sha(WEIGHTS)},
        "settings": {"modalities": "S2 L2A 10 bands in 3 groups", "groups": [list(g) for g in GROUPS],
                     "normalization": "(x - (mean - 2 std)) / (4 std) clipped to [0, 1], fMoW-Sentinel statistics",
                     "layer": "encoder output after the final norm; the 3 group tokens per position averaged"}})


if __name__ == "__main__":
    pe.cli(make_model)
