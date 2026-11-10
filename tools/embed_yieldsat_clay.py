"""Frozen Clay v1.5 patch embeddings for every YieldSAT pixel (protocols/yieldsat.md, 2.5).

Clay v1.5 (Clay Foundation, 2025) is a ViT-L/8 masked autoencoder with a DOFA-style wavelength-conditioned patch
embedding and a DINOv2 teacher (representation loss), trained on many sensors. Its position encoding adds the
image's GSD, week of year / hour and lat/lon. Here: the peak-greenness scene, the 10 S2 bands Clay was trained on
(B02-B08, B8A, B11, B12; no B01/B09), z-scored with Clay's sentinel-2-l2a statistics, wavelengths in um, gsd 10 m,
256 px windows (Clay's training size) and 8 px tokens (32 x 32 grid, 80 m per token), 1024-d. Time is the scene's
week of year with hour 10 (Sentinel-2 overpass is about 10:30 local); lat/lon are zeros (withheld, Clay's convention
for unknown location), so location enters only through the location-encoder inputs. No masking, no shuffling.

Model code: terratorch clay_v15 Encoder (pinned, layered env envs/patchfm); weights pinned under upstream/clay/weights.

    envs/patchfm/bin/python -m tools.embed_yieldsat_clay --subset GER-W
"""
import math

import numpy as np
import torch

from tools import patch_embed as pe
from tools import yieldsat as ys

WEIGHTS = ys.PROJECT / "upstream" / "clay" / "weights" / "clay-v1.5.ckpt"
HF_REVISION = "made-with-clay/Clay@70200ebcccdf67bf2a0cb9984c77ddee26c10ed2 v1.5/clay-v1.5.ckpt"
TERRATORCH_COMMIT = "83c3a928416fcf97deed924cfb99a59a817730bf"
SIZE, TOKEN = 256, 8
BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]   # RAW_S2_BANDS indices of B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
# upstream/clay/src/configs/metadata.yaml, sentinel-2-l2a
MEAN = np.array([1105., 1355., 1552., 1887., 2422., 2630., 2743., 2785., 2388., 1835.], np.float32)
STD = np.array([1809., 1757., 1888., 1870., 1732., 1697., 1742., 1648., 1470., 1379.], np.float32)
WAVES_UM = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
HOUR = 10


def make_model():
    from terratorch.models.backbones.clay_v15.model import Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Encoder(mask_ratio=0.0, patch_size=TOKEN, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64,
                  mlp_ratio=4)
    ckpt = pe.load_state(WEIGHTS)
    state = ckpt.get("state_dict", ckpt)
    enc = {k.removeprefix("model.encoder."): v for k, v in state.items() if k.startswith("model.encoder.")}
    net.load_state_dict(enc, strict=True)
    net = net.to(device).eval()
    waves = torch.tensor(WAVES_UM, device=device)

    def prepare(img):
        return ((img[BANDS] - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32)

    def encode(x, dates):
        b = len(dates)
        week = torch.tensor([d.isocalendar()[1] * 2 * math.pi / 52 for d in dates], device=x.device)
        hour = torch.full((b,), HOUR * 2 * math.pi / 24, device=x.device)
        time = torch.stack([torch.sin(week), torch.cos(week), torch.sin(hour), torch.cos(hour)], 1)
        cube = {"pixels": x, "time": time, "latlon": torch.zeros((b, 4), device=x.device),
                "gsd": torch.tensor(10.0, device=x.device), "waves": waves}
        tok = net(cube)[0][:, 1:]
        return pe.tokens_to_grid(tok, SIZE // TOKEN)

    return pe.Model("clay", 1024, SIZE, TOKEN, prepare, encode, {
        "model": "Clay v1.5 encoder (ViT-L/8), terratorch clay_v15",
        "revision": {"weights": HF_REVISION, "terratorch": TERRATORCH_COMMIT,
                     "weights_sha256": pe.weights_sha(WEIGHTS)},
        "settings": {"modalities": "S2 L2A 10 bands (B02-B08, B8A, B11, B12)",
                     "normalization": "Clay sentinel-2-l2a mean/std", "waves_um": WAVES_UM, "gsd_m": 10,
                     "metadata": "week of year of the scene, hour 10; lat/lon zeros (withheld)",
                     "layer": "encoder output (final LayerNorm), class token dropped"}})


if __name__ == "__main__":
    pe.cli(make_model)
