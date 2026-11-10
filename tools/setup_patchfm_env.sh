#!/bin/bash
# Layered environment for the terratorch-based patch foundation models on YieldSAT (TerraMind, THOR) and the
# pinned upstream sources of Clay and SatMAE: a venv on top of benchmark-v2, so the shared env is never modified.
# Run once on the Anunna login node (needs internet):
#     bash tools/setup_patchfm_env.sh
set -euo pipefail
project=/lustre/shared/agriscience-benchmark
base=$project/envs/benchmark-v2/bin/python
venv=$project/envs/patchfm
TERRATORCH_COMMIT=83c3a928416fcf97deed924cfb99a59a817730bf
[ -d "$venv" ] || "$base" -m venv "$venv"
# benchmark-v2 is itself a venv, so --system-site-packages would not see its packages: link them with a .pth file
# (packages installed here take precedence).
"$base" -c "import site; print(site.getsitepackages()[0])" > "$("$venv/bin/python" -c "import site; print(site.getsitepackages()[0])")/benchmark-v2.pth"
"$venv/bin/pip" install -q "terratorch @ https://github.com/IBM/terratorch/archive/$TERRATORCH_COMMIT.zip"
"$venv/bin/python" -c "import terratorch; from terratorch.registry import BACKBONE_REGISTRY; print('patchfm env ok, terratorch', terratorch.__version__)"
