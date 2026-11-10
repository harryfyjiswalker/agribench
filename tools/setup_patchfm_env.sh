#!/bin/bash
# Layered environment for the terratorch-based patch foundation models on YieldSAT (TerraMind, Clay v1.5, THOR): a venv on top of benchmark-v2, so the shared env is never modified.
# Run once on the Anunna login node (needs internet):
#     bash tools/setup_patchfm_env.sh
set -euo pipefail
project=/lustre/shared/agriscience-benchmark
base=$project/envs/benchmark-v2/bin/python
venv=$project/envs/patchfm
TERRATORCH_COMMIT=83c3a928416fcf97deed924cfb99a59a817730bf
THOR_EXT_COMMIT=00223bb70dac06d60aac38347afb54e32a3aabf7   # FM4CS/thor_terratorch_ext: registers the THOR backbones
THOR_COMMIT=7ded3cea673ac21fe2bcadf9f3d9d4506eb6ab5f       # FM4CS/THOR: model code the extension imports
[ -d "$venv" ] || "$base" -m venv "$venv"
# benchmark-v2 is itself a venv, so --system-site-packages would not see its packages: link them with a .pth file
# (packages installed here take precedence).
"$base" -c "import site; print(site.getsitepackages()[0])" > "$("$venv/bin/python" -c "import site; print(site.getsitepackages()[0])")/benchmark-v2.pth"
"$venv/bin/pip" install -q "terratorch @ https://github.com/IBM/terratorch/archive/$TERRATORCH_COMMIT.zip"
# THOR without dependencies: its extra ones (geojson, h5netcdf) are only needed for its datasets, not the backbone
"$venv/bin/pip" install -q --no-deps "thor_terratorch_ext @ https://github.com/FM4CS/thor_terratorch_ext/archive/$THOR_EXT_COMMIT.zip"
"$venv/bin/pip" install -q --no-deps "thor @ https://github.com/FM4CS/THOR/archive/$THOR_COMMIT.zip"
"$venv/bin/python" -c "import terratorch, thor_terratorch_ext; from terratorch.registry import BACKBONE_REGISTRY; print('patchfm env ok')"
