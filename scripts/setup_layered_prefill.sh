#!/usr/bin/env bash
# Separate env for the scale-snu/layered-prefill fork (step 0 demo, step 5 prototype).
# The fork pins torch 2.8.0 + CUDA 12.8 and builds its own vllm-flash-attn, so it cannot
# share the `hyprefill` env (vLLM 0.30, torch 2.13). Follows the fork's README, plus:
#   - Hopper only (TORCH_CUDA_ARCH_LIST=9.0) to halve the flash-attention build
#   - pip timeouts: pypi.org times out from this server (see AGENTS.md, Server)
#   - build pinned to NUMA-0 CPUs so it does not disturb benchmarks on GPU 4-7
# Run detached:  tmux new -d -s lp-setup 'bash scripts/setup_layered_prefill.sh'
set -euo pipefail
cd "$(dirname "$0")/../third_party/layered-prefill"

ENV_NAME="${LP_ENV:-layered-prefill}"
CPUS="${BUILD_CPUS:-0-23,96-119}"
export MAX_JOBS="${MAX_JOBS:-32}"
LOG="$HOME/hyprefill_data/setup_layered_prefill_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
# conda-forge only: mixing the nvidia channel in gave cuda-toolkit 12.6 + nvcc 12.4 + cudart 12.8
conda install -y -n "$ENV_NAME" --override-channels -c conda-forge \
  "cuda-toolkit=12.8" "cuda-nvcc=12.8" "cuda-version=12.8" cmake ninja ccache c-compiler cxx-compiler
conda run -n "$ENV_NAME" nvcc --version | grep -q "release 12.8" || { echo "nvcc is not 12.8"; exit 1; }
set +u  # conda activate.d scripts (cuda-nvcc) read unset variables
conda activate "$ENV_NAME"
set -u
PIP=(pip install --timeout 120 --retries 10)
python -c "import torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null || "${PIP[@]}" torch==2.8.0 uv httpie psutil
python -c "import dateutil" 2>/dev/null || "${PIP[@]}" six "$(curl -s https://pypi.org/pypi/python-dateutil/json |
  python -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('.whl')][0])")"

# nvcc lives in $CONDA_PREFIX/bin, headers/libs in targets/x86_64-linux (the fork's README
# sets CUDA_HOME to the latter, which only works at run time, not for the build)
T="$CONDA_PREFIX/targets/x86_64-linux"
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
export CPATH="$T/include${CPATH:+:$CPATH}" LIBRARY_PATH="$T/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0" CCACHE_NOHASHDIR=true UV_HTTP_TIMEOUT=120
if [[ ! -d flash-attention ]]; then
  git clone https://github.com/vllm-project/flash-attention.git flash-attention
  (cd flash-attention && git checkout d9e577e && patch -p0 < ../flash-attention.patch)
fi
python -c "import vllm_flash_attn" 2>/dev/null ||
  taskset -c "$CPUS" uv pip install -e flash-attention --verbose --no-build-isolation
python -c "import nanovllm.ops" 2>/dev/null ||
  taskset -c "$CPUS" uv pip install -e . --verbose --no-build-isolation
python -c "import torch, vllm_flash_attn, nanovllm; print('OK torch', torch.__version__, torch.version.cuda)"
echo "=== layered-prefill env ready"
