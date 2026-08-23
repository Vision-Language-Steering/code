#!/bin/bash
# One-shot environment build on the NAS (everything lives under BASE; local disk untouched).
# Run:  bash scripts/setup_env_nas.sh 2>&1 | tee /mnt/nas/ishneet/vls-rebuttal/setup.log
set -euo pipefail

BASE=/mnt/nas/ishneet/vls-rebuttal
CODE=$BASE/code
export UV_CACHE_DIR=$BASE/uv-cache
export HF_HOME=$BASE/hf_home
export PIP_CACHE_DIR=$BASE/pip-cache
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR"

echo "=== [1/6] Clone submodules at pinned commits (from requirements.txt) ==="
mkdir -p "$CODE/third_party"
if [ ! -d "$CODE/third_party/lerobot/.git" ]; then
  git clone https://github.com/Treeeplanter/lerobot.git "$CODE/third_party/lerobot"
fi
git -C "$CODE/third_party/lerobot" checkout fbf0fb1e6a39a9223b61fcd09aead75c95363194

if [ ! -d "$CODE/third_party/libero_pro/.git" ]; then
  git clone https://github.com/Treeeplanter/LIBERO-PRO.git "$CODE/third_party/libero_pro"
fi
git -C "$CODE/third_party/libero_pro" checkout 6adba85c03663a7dac0df22ae54dbcf069f55623

echo "=== [2/6] Create venv (python 3.12, on NAS) ==="
if [ ! -d "$BASE/venv" ]; then
  uv venv --python 3.12 "$BASE/venv"
fi
source "$BASE/venv/bin/activate"

echo "=== [3/6] Install torch cu118 (pinned) ==="
uv pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "=== [4/6] Install requirements (frozen env: --no-deps, no resolution) ==="
# requirements.txt is a pip freeze of the authors' working env — reproduce it verbatim.
# Filter: editable/git-ssh lines, torch family (installed above with cu118 index),
# and the transformers git pin (installed separately below).
grep -vE '^-e |^torch==|^torchvision==|^torchaudio==|^transformers @' "$CODE/requirements.txt" \
  > /tmp/reqs_filtered.txt
uv pip install --no-deps -r /tmp/reqs_filtered.txt
uv pip install --no-deps "git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"

echo "=== [5/6] Install lerobot fork + LIBERO-PRO (editable, no-deps to keep pins) ==="
uv pip install -e "$CODE/third_party/lerobot" --no-deps
uv pip install -e "$CODE/third_party/libero_pro" --no-deps

echo "=== [6/6] Smoke imports + weight prefetch ==="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
import lerobot
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print("lerobot + pi05 import OK")
import libero
print("libero import OK")
from huggingface_hub import snapshot_download
p = snapshot_download("lerobot/pi05_libero_finetuned_v044")
print("pi05 checkpoint at", p)
q = snapshot_download("facebook/dinov3-vitb16-pretrain-lvd1689m")
print("dinov3 weights at", q)
EOF

echo ""
echo "=== SETUP COMPLETE ==="
echo "Activate with: source $BASE/venv/bin/activate && export HF_HOME=$BASE/hf_home"
