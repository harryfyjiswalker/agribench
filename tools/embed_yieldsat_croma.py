"""Frozen CROMA (ViT-B/8, optical encoder) patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

CROMA (Fuller et al. 2023, arXiv:2311.00566) pretrains separate S1 and S2 encoders plus a joint encoder with
contrastive radar-optical alignment and masked reconstruction on SSL4EO-S12. YieldSAT has no S1, so only the
optical encoder is used (its features were still shaped by the cross-modal objective). Input: the peak-greenness
scene, 12 L2A bands (CROMA's 12-band S2, cirrus removed), scaled as in the CROMA README, (x - (mean - 2 std)) /
(4 std) clipped to [0, 1], but with fixed SSL4EO-S12 statistics instead of batch statistics so a pixel's input does
not depend on its batch; 120 px windows (CROMA's native size) and 8 px tokens (15 x 15 grid, 80 m per token).

Model code: torchgeo (benchmark-v2) CROMA; weights pinned under upstream/croma/weights.

    python -m tools.embed_yieldsat_croma --subset GER-W
"""
import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "croma" / "weights" / "CROMA_base-0238d814.pt"
SIZE, TOKEN = 120, 8


def make_model():
    import torchgeo
    from torchgeo.models import CROMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CROMA(modalities=["optical"], encoder_dim=768, encoder_depth=12, num_heads=16, patch_size=TOKEN,
                image_size=SIZE)
    state = pe.load_state(WEIGHTS)
    for part in ("s2_encoder", "s2_GAP_FFN"):
        getattr(net, part).load_state_dict(state[part], strict=True)
    net = net.to(device).eval()

    def encode(x, dates):
        return pe.tokens_to_grid(net(x_optical=x)["optical_encodings"], SIZE // TOKEN)

    return pe.Model("croma", 768, SIZE, TOKEN, lambda img: pe.minmax2std(img).astype(np.float32), encode, {
        "model": "CROMA base, optical (S2) encoder, torchgeo CROMA",
        "revision": {"torchgeo": torchgeo.__version__, "weights_sha256": pe.weights_sha(WEIGHTS),
                     "weights_url": "https://hf.co/torchgeo/croma/resolve/387883f08af79d777167519c57cd826eda89a16f/"
                                    "CROMA_base-0238d814.pt"},
        "settings": {"modalities": "S2 L2A 12 bands (optical encoder only; no S1 in YieldSAT)",
                     "normalization": "(x - (mean - 2 std)) / (4 std) clipped to [0, 1], SSL4EO-S12 statistics",
                     "layer": "optical encoder output tokens"}})


if __name__ == "__main__":
    pe.cli(make_model)
