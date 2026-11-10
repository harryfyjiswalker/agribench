"""Frozen DOFA (ViT-B/16) patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

DOFA (Xiong et al. 2024, arXiv:2403.15356) is a masked-autoencoder ViT whose patch embedding is generated from each
band's wavelength, so one model takes any sensor; pretrained on SatlasPretrain, Five-Billion-Pixels and
HySpecNet-11k. Here: the peak-greenness scene (12 L2A bands, centre wavelengths in um), z-scored with the SSL4EO-S12
statistics Copernicus-Bench uses for DOFA, 224 px windows, 16 px tokens.

Model code: torchgeo (benchmark-v2) DOFA; weights pinned under upstream/dofa/weights.

    python -m tools.embed_yieldsat_dofa --subset GER-W
"""
import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "dofa" / "weights" / "dofa_base_patch16_224-a0275954.pth"
SIZE, TOKEN = 224, 16
WAVELENGTH_UM = [w / 1000 for w in pe.WAVELENGTH_NM]


def make_model():
    import torchgeo
    from torchgeo.models import DOFA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DOFA(img_size=SIZE, patch_size=TOKEN, embed_dim=768, depth=12, num_heads=12, global_pool=True)
    missing, unexpected = net.load_state_dict(pe.load_state(WEIGHTS), strict=False)
    if set(missing) - {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"} or unexpected:
        raise ValueError(f"DOFA weights: missing {missing}, unexpected {unexpected}")
    net = net.to(device).eval()
    store = pe.capture_output(net.blocks[-1])  # checkpoint has no final norm

    def encode(x, dates):
        net.forward_features(x, WAVELENGTH_UM)
        return pe.tokens_to_grid(store["x"][:, 1:], SIZE // TOKEN)

    return pe.Model("dofa", 768, SIZE, TOKEN, lambda img: pe.zscore(img).astype(np.float32), encode, {
        "model": "DOFA ViT-B/16 (MAE), torchgeo DOFA",
        "revision": {"torchgeo": torchgeo.__version__, "weights_sha256": pe.weights_sha(WEIGHTS),
                     "weights_url": "https://hf.co/torchgeo/dofa/resolve/b8db318b64a90b9e085ec04ba8851233c5893666/"
                                    "dofa_base_patch16_224-a0275954.pth"},
        "settings": {"modalities": "S2 L2A 12 bands", "normalization": "SSL4EO-S12 z-score",
                     "wavelengths_um": WAVELENGTH_UM, "layer": "last transformer block output (the checkpoint has no final norm)"}})


if __name__ == "__main__":
    pe.cli(make_model)
