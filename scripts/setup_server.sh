#!/usr/bin/env bash
# HyPrefill — one-time setup on the 8×H200 server. Run from the repo root:
#   bash scripts/setup_server.sh            # env + checks + step-0 models
#   bash scripts/setup_server.sh --no-models
#
# Safe to re-run: every step skips work that is already done.
# Needs no root. Clock locking (step 1+) is checked but not required here.
set -euo pipefail

ENV_NAME="${ENV_NAME:-hyprefill}"
# On this server $HOME is the large writable NVMe (RAID0, ~5.6 TB free); /mnt/models
# is a shared read-only store. Override HF_HOME if you move to another machine.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
# Raw dumps (nsys/ncu, expert routes, traces) live outside the repo.
DATA_DIR="${DATA_DIR:-$HOME/hyprefill_data}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
# Benchmarks call the kernels vLLM serves with (FA3 via vllm_flash_attn, FlashInfer GDN),
# so the env is built around a pinned vLLM wheel; it pins torch (2.13.0+cu130 for 0.30.0).
# The standalone flash-attn 2.8.3 has no wheel for recent torch and fails to build (C++20).
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
DOWNLOAD_MODELS=1
[[ "${1:-}" == "--no-models" ]] && DOWNLOAD_MODELS=0

say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. GPU and driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NGPU"
echo "Other processes on the GPUs right now (should be empty for clean measurements):"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true
# Processes in other containers do not show up above, so check memory and load directly.
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
while IFS=', ' read -r i used total util; do
  if (( used * 10 > total )) || (( util > 5 )); then
    echo "WARNING: GPU $i busy: ${used}/${total} MiB used, ${util}% util (maybe another container)"
  fi
done

say "2. Disk for HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME" "$DATA_DIR"
df -h "$HF_HOME"
AVAIL_GB=$(df --output=avail -BG "$HF_HOME" | tail -1 | tr -dc '0-9')
echo "Available: ${AVAIL_GB} GB (need ~800 GB for the 5 main models, ~2 TB with the optional ones)"
[[ "$AVAIL_GB" -lt 800 ]] && echo "WARNING: not enough space; set HF_HOME to a larger volume."

say "3. Can we lock GPU clocks? (needed for clean timing from step 1)"
if sudo -n true 2>/dev/null; then
  echo "sudo without password: OK. In a container that may still not allow clock changes"
  echo "(on quangch1-dev-0 it does not: the host admin has to lock). Test on a free GPU with"
  echo "  sudo nvidia-smi -i <gpu> -lgc 1980,1980 && sudo nvidia-smi -i <gpu> -rgc"
else
  echo "No passwordless sudo. Ask the admin to lock clocks, or record clocks during every run"
  echo "(nvidia-smi --query-gpu=clocks.sm --format=csv -lms 100) and use more repetitions."
fi

say "4. Conda environment '$ENV_NAME'"
if ! command -v conda >/dev/null && [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  echo "conda not found — installing Miniforge into $CONDA_DIR"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && rm -f /tmp/miniforge.sh
fi
command -v conda >/dev/null || export PATH="$CONDA_DIR/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
# pypi.org times out often from this server; a timed-out index page makes pip fall back
# to building sdists (pandas failed that way), so use long timeouts and wheels only.
PIP=(pip install -q --timeout 120 --retries 10)
"${PIP[@]}" --upgrade pip
# pypi.org/simple/python-dateutil/ hangs from this server while the JSON API works,
# so install that one wheel by URL before anything depends on it.
python -c "import dateutil" 2>/dev/null || "${PIP[@]}" six "$(curl -s https://pypi.org/pypi/python-dateutil/json |
  python -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('.whl')][0])")"
python -c "import vllm" 2>/dev/null || "${PIP[@]}" "vllm==$VLLM_VERSION"
"${PIP[@]}" --only-binary=:all: transformers accelerate safetensors pandas matplotlib seaborn huggingface_hub

say "5. Third-party code"
mkdir -p third_party
# Source checkout at the same tag as the installed wheel, for reading and later forking.
[[ -d third_party/vllm ]] || git clone --depth 1 --branch "v$VLLM_VERSION" https://github.com/vllm-project/vllm third_party/vllm
[[ -d third_party/layered-prefill ]] || git clone https://github.com/scale-snu/layered-prefill third_party/layered-prefill
# vLLM from source is slow to build; do it once when step 5+ needs it:
#   (cd third_party/vllm && pip install -e .)

say "6. Environment check"
python scripts/check_env.py

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
  say "7. Step-0 models (resumable; re-run if interrupted)"
  for m in Qwen/Qwen3-Next-80B-A3B-Instruct Qwen/Qwen3.8-27B Qwen/Qwen3-30B-A3B; do
    echo "--- $m"; hf download "$m" --quiet || echo "FAILED: $m (check the repo id / access)"
  done
  echo
  echo "Later (steps 1–2), when disk allows:"
  echo "  hf download Qwen/Qwen3.8-Flash-Next-FP8"
  echo "  hf download moonshotai/Kimi-Linear-48B-A3B-Instruct"
fi

say "Done. Add to ~/.bashrc so every shell uses the same cache and toolchain:"
echo "  export HF_HOME=$HF_HOME"
echo "  export CUDA_HOME=$CUDA_HOME PATH=$CUDA_HOME/bin:\$PATH"
echo "  source $(conda info --base)/etc/profile.d/conda.sh"
echo
echo "Models already on the shared store (read-only, pass the path directly, do not copy):"
echo "  ls /mnt/models/hf"
