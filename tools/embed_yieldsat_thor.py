"""Frozen THOR-1.0-base patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

THOR (Forgaard et al. 2026, arXiv:2601.16011; Norwegian Computing Center, UiT, ESA Phi-lab; Apache-2.0) is a
compute-adaptive FlexiViT with ALiBi attention, pretrained on paired Sentinel-1, -2 and -3 data (THOR-Pretrain).
Each band group gets its own patch embedding at its native resolution. Here: the peak-greenness scene, the 12 S2
L2A bands as reflectance (DN / 10000), z-scored with THOR's pretraining statistics; a 2880 m ground cover (288 px
windows at 10 m, THOR's default), which the wrapper resamples to each group's native grid (10 m RGB+NIR, 20 m red
edge/SWIR, 60 m aerosol/water vapour); 8 px patches at 10 m (36 x 36 grid, 80 m per token; 20 m and 60 m groups
give 18 x 18 and 6 x 6). The groups' last-layer tokens are bilinearly upsampled to 36 x 36 and averaged
(terratorch merge_method "mean"). No S1/S3 (not in YieldSAT).

Model code: thor_terratorch_ext 00223bb and thor (FM4CS/THOR 7ded3ce) in the layered env envs/patchfm. Weights are
pinned under upstream/thor/weights and loaded by THOR's registry with torch.load, which is tensors-only in this
torch (>= 2.6); key coverage is checked here because THOR loads with strict=False.

    envs/patchfm/bin/python -m tools.embed_yieldsat_thor --subset GER-W
"""
import re

import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "thor" / "weights" / "thor_v1_vit_base.pt"
HF_REVISION = "FM4CS/THOR-1.0-base@9c13d7b8f7bd996c6109e1c61620eeffb543ab9a"
CODE = {"thor_terratorch_ext": "FM4CS/thor_terratorch_ext@00223bb70dac06d60aac38347afb54e32a3aabf7",
        "thor": "FM4CS/THOR@7ded3cea673ac21fe2bcadf9f3d9d4506eb6ab5f"}
GROUND_COVER_M, PATCH = 2880, 8
SIZE = GROUND_COVER_M // 10
BAND_NAMES = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
              "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]   # = RAW_S2_BANDS order


def make_model():
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 6):
        raise RuntimeError("THOR's loader relies on torch.load defaulting to weights_only=True (torch >= 2.6)")
    import thor_terratorch_ext  # noqa: F401  (registers the backbones)
    from terratorch import BACKBONE_REGISTRY
    from thor_terratorch_ext.models.backbones import thor_vit as tv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BACKBONE_REGISTRY.build("thor_v1_base", pretrained=True, ckpt=str(WEIGHTS), model_bands=BAND_NAMES,
                                  ground_cover=GROUND_COVER_M, patch_sizes=[PATCH], merge_method="mean")
    # strict=False upstream: check every encoder weight came from the checkpoint
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    from thor.core.model_registry import extract_model_state_dict_from_ckpt
    sd = extract_model_state_dict_from_ckpt(ckpt)["encoder"]
    model_sd = net.model.state_dict()
    ignore = [re.compile(r) for r in ("decoder_.*", "decode_.*", "ref_pos_embed.*", "pos_embed.*")]
    missing = [k for k in model_sd if k not in sd and not any(r.match(k) for r in ignore)]
    used = [k for k in model_sd if k in sd]
    if missing or not used or any(not torch.equal(model_sd[k].cpu(), sd[k]) for k in used[:20] + used[-20:]):
        raise ValueError(f"THOR weights not fully loaded: missing {missing[:10]}")
    net = net.to(device).eval()
    keys = [tv.lookup_band[b] for b in BAND_NAMES]
    mean = np.array([tv.THOR_NORMALIZATION_PARAMS[k]["mean"] for k in keys], np.float32)
    std = np.array([tv.THOR_NORMALIZATION_PARAMS[k]["std"] for k in keys], np.float32)

    def prepare(img):
        return ((img / 10000.0 - mean[:, None, None]) / std[:, None, None]).astype(np.float32)

    def encode(x, dates):
        return net(x)[-1]

    return pe.Model("thor", 768, SIZE, PATCH, prepare, encode, {
        "model": "THOR-1.0-base (FlexiViT, ALiBi), thor_terratorch_ext thor_v1_base",
        "revision": {"weights": HF_REVISION, **CODE, "weights_sha256": pe.weights_sha(WEIGHTS),
                     "loaded_tensors": len(used)},
        "settings": {"modalities": "S2 L2A 12 bands (no S1/S3 in YieldSAT)",
                     "normalization": "reflectance (DN/10000) z-scored with THOR pretraining statistics",
                     "ground_cover_m": GROUND_COVER_M, "patch_size_px_10m": PATCH,
                     "groups": "10 m RGB+NIR, 20 m red edge + SWIR, 60 m aerosol + water vapour (native grids)",
                     "layer": "last block; group token grids upsampled to 36 x 36 and averaged (merge 'mean')"}})


if __name__ == "__main__":
    pe.cli(make_model)
