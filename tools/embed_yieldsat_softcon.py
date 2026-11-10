"""Frozen SoftCon (ViT-B/14, DINOv2 architecture) patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

SoftCon (Wang et al. 2024, arXiv:2405.20462) is a multi-label-guided soft contrastive model (continual pretraining
from DINOv2 weights) on SSL4EO-S12 13-band L1C. Input: the peak-greenness scene; the 12 L2A bands plus B10
(cirrus, absent from L2A) set to its SSL4EO mean, scaled with Copernicus-Bench's SoftCon recipe, (x - (mean -
2 std)) / (4 std) clipped to [0, 1], SSL4EO-S12 statistics; 224 px windows, 14 px tokens (16 x 16 grid).

Model code: timm vit_base_patch14_dinov2 via torchgeo (benchmark-v2); weights pinned under upstream/softcon/weights.

    python -m tools.embed_yieldsat_softcon --subset GER-W
"""
import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "softcon" / "weights" / "B13_vitb14_softcon_enc.pth"
SIZE, TOKEN = 224, 14
B10_AT = 10  # L1C order B01..B8A, B09, B10, B11, B12


def prepare(img):
    x = pe.minmax2std(img)
    return np.insert(x, B10_AT, 0.5, axis=0).astype(np.float32)   # B10 at its mean -> 0.5 after scaling


def make_model():
    import timm
    import torchgeo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model("vit_base_patch14_dinov2", in_chans=13, img_size=SIZE, num_classes=0)
    missing, unexpected = net.load_state_dict(pe.load_state(WEIGHTS), strict=False)
    if missing or unexpected:
        raise ValueError(f"SoftCon weights: missing {missing}, unexpected {unexpected}")
    net = net.to(device).eval()

    def encode(x, dates):
        tok = net.forward_features(x)[:, net.num_prefix_tokens:]
        return pe.tokens_to_grid(tok, SIZE // TOKEN)

    return pe.Model("softcon", 768, SIZE, TOKEN, prepare, encode, {
        "model": "SoftCon ViT-B/14 (S2, 13 bands), timm vit_base_patch14_dinov2",
        "revision": {"torchgeo": torchgeo.__version__, "timm": timm.__version__,
                     "weights_sha256": pe.weights_sha(WEIGHTS),
                     "weights_url": "https://huggingface.co/wangyi111/softcon/resolve/"
                                    "bae909781911f8ec034b4b959992fae17b973c0c/B13_vitb14_softcon_enc.pth"},
        "settings": {"modalities": "S2 L2A 12 bands + B10 filled with its SSL4EO mean",
                     "normalization": "(x - (mean - 2 std)) / (4 std) clipped to [0, 1], SSL4EO-S12 statistics",
                     "layer": "last block, final norm (timm forward_features)"}})


if __name__ == "__main__":
    pe.cli(make_model)
