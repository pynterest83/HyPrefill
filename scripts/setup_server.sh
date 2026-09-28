#!/usr/bin/env bash
# HyPrefill — one-time setup on the 8×H200 server. Run from the repo root:
#   bash scripts/setup_server.sh            # env + checks + week-0 models
#   bash scripts/setup_server.sh --no-models
#
# Safe to re-run: every step skips work that is already done.
# Needs no root. Clock locking (week 1+) is checked but not required here.
set -euo pipefail

ENV_NAME="${ENV_NAME:-hyprefill}"
# Put the HF cache on the biggest fast NVMe, NOT on the home directory.
export HF_HOME="${HF_HOME:-/data/hf_cache}"
DOWNLOAD_MODELS=1
[[ "${1:-}" == "--no-models" ]] && DOWNLOAD_MODELS=0

say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. GPU and driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NGPU"
echo "Other processes on the GPUs right now (should be empty for clean measurements):"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true

say "2. Disk for HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME"
df -h "$HF_HOME"
AVAIL_GB=$(df --output=avail -BG "$HF_HOME" | tail -1 | tr -dc '0-9')
echo "Available: ${AVAIL_GB} GB (need ~800 GB for the 5 main models, ~2 TB with the optional ones)"
[[ "$AVAIL_GB" -lt 800 ]] && echo "WARNING: not enough space; set HF_HOME to a larger volume."

say "3. Can we lock GPU clocks? (needed for clean timing from week 1)"
if sudo -n true 2>/dev/null; then
  echo "sudo without password: OK — lock with: sudo nvidia-smi -lgc 1980,1980"
else
  echo "No passwordless sudo. Ask the admin to lock clocks, or record clocks during every run"
  echo "(nvidia-smi --query-gpu=clocks.sm --format=csv -lms 100) and use more repetitions."
fi

say "4. Conda environment '$ENV_NAME'"
if ! command -v conda >/dev/null; then echo "conda not found — install Miniforge first"; exit 1; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
pip install -q --upgrade pip
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers accelerate safetensors pandas matplotlib seaborn "huggingface_hub[hf_transfer]"
python -c "import flash_attn" 2>/dev/null || pip install -q flash-attn --no-build-isolation
python -c "import fla" 2>/dev/null || pip install -q flash-linear-attention

say "5. Third-party code"
mkdir -p third_party
[[ -d third_party/vllm ]] || git clone --depth 1 https://github.com/vllm-project/vllm third_party/vllm
[[ -d third_party/layered-prefill ]] || git clone https://github.com/scale-snu/layered-prefill third_party/layered-prefill
# vLLM from source is slow to build; do it once when week 5+ needs it:
#   (cd third_party/vllm && pip install -e .)

say "6. Environment check"
python scripts/check_env.py

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
  say "7. Week-0 models (resumable; re-run if interrupted)"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  for m in Qwen/Qwen3-Next-80B-A3B-Instruct Qwen/Qwen3.8-27B Qwen/Qwen3-30B-A3B; do
    echo "--- $m"; huggingface-cli download "$m" --quiet || echo "FAILED: $m (check the repo id / access)"
  done
  echo
  echo "Later (weeks 1–2), when disk allows:"
  echo "  huggingface-cli download Qwen/Qwen3.8-Flash-Next-FP8"
  echo "  huggingface-cli download moonshotai/Kimi-Linear-48B-A3B-Instruct"
fi

say "Done. Add to ~/.bashrc so every shell uses the same cache:"
echo "  export HF_HOME=$HF_HOME"
