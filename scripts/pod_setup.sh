#!/usr/bin/env bash
# One-time setup on a fresh pod: install the Python dependencies the
# training and evaluation pipelines need.  Idempotent — re-running is a
# no-op once everything is installed.
#
# Run on the pod:
#     bash /workspace/pixel-vs-prior/scripts/pod_setup.sh
set -euo pipefail

pip install --break-system-packages -q --root-user-action=ignore \
    "numpy<2" \
    pandas \
    pillow \
    diffusers \
    transformers \
    timm

# Verify
python3 - <<'PY'
import torch, numpy, PIL, pandas, diffusers
print(f"torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"numpy    {numpy.__version__}")
print(f"pandas   {pandas.__version__}")
print(f"pillow   {PIL.__version__}")
print(f"diffusers {diffusers.__version__}")
PY

echo "pod ready."
