"""Frozen TerraMind-1.0-base patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

TerraMind (Jakubik et al. 2025, arXiv:2504.11171; IBM and ESA) is an any-to-any generative multimodal model:
pretrained with a dual-scale masked objective over pixel and token inputs of S1, S2, DEM, land cover, NDVI,
coordinates and captions (TerraMesh). Here the encoder takes two raw ("untok") image modalities of the
peak-greenness date: S2 L2A (12 bands, z-scored with TerraMind's v1 pretraining statistics) and the field's Raw DEM
(z-scored likewise). The 224 px windows give 16 px tokens; S2 and DEM tokens at the same position are averaged
(terratorch merge_method "mean") after the final encoder norm. No S1 (not in YieldSAT), no coordinates.

Model code: terratorch (pinned, layered env envs/patchfm, tools/setup_patchfm_env.sh); weights pinned under
upstream/terramind/weights.

    envs/patchfm/bin/python -m tools.embed_yieldsat_terramind --subset GER-W
"""
import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "terramind" / "weights" / "TerraMind_v1_base.pt"
HF_REVISION = "ibm-esa-geospatial/TerraMind-1.0-base@fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a"
TERRATORCH_COMMIT = "83c3a928416fcf97deed924cfb99a59a817730bf"
SIZE, TOKEN = 224, 16


def make_model():
    from terratorch.models.backbones.terramind.model import terramind_register as tr
    mean = np.array(tr.v1_pretraining_mean["untok_sen2l2a@224"], np.float32)
    std = np.array(tr.v1_pretraining_std["untok_sen2l2a@224"], np.float32)
    dem_mean, dem_std = tr.v1_pretraining_mean["untok_dem@224"][0], tr.v1_pretraining_std["untok_dem@224"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = tr.terramind_v1_base(modalities=["S2L2A", "DEM"], merge_method="mean", pretrained=False)
    state = tr.checkpoint_filter_fn(pe.load_state(WEIGHTS), net)
    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()

    def prepare(img):
        s2 = (img[:12] - mean[:, None, None]) / std[:, None, None]
        dem = (img[12:] - dem_mean) / dem_std
        return np.concatenate([s2, dem]).astype(np.float32)

    def encode(x, dates):
        out = net({"S2L2A": x[:, :12], "DEM": x[:, 12:]})
        return pe.tokens_to_grid(out[-1], SIZE // TOKEN)

    return pe.Model("terramind", 768, SIZE, TOKEN, prepare, encode, {
        "model": "TerraMind-1.0-base encoder (terratorch terramind_v1_base)",
        "revision": {"weights": HF_REVISION, "terratorch": TERRATORCH_COMMIT,
                     "weights_sha256": pe.weights_sha(WEIGHTS)},
        "settings": {"modalities": "S2 L2A 12 bands + DEM (Raw dem-<field>.tif); no S1, no coordinates",
                     "normalization": "TerraMind v1 pretraining mean/std (untok_sen2l2a@224, untok_dem@224)",
                     "layer": "last encoder layer after encoder_norm; S2 and DEM tokens averaged per position"}},
        with_dem=True)


if __name__ == "__main__":
    pe.cli(make_model)
