#!/usr/bin/env bash
# Step 2 kernel sweep: MoE and decode-side costs for the oracle, on one locked GPU.
#   MoE (prefill c and decode batch sizes), decode attention (batch x context), GDN decode,
#   dense ops at decode batch sizes, plus a card check: a few step 1 FA/GDN rows re-measured on
#   this GPU, to compare with the GPU 4 tables before mixing cost tables across cards.
#   tmux new -d -s s2g7 'GPU=7 CPUS=72-95 bash bench/run_step02.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-4}"; CPUS="${CPUS:-48-71}"; LOCK_MHZ="${LOCK_MHZ:-1980}"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hyprefill}"
LOG="$HOME/hyprefill_data/step02_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
idle=$(nvidia-smi -i "$GPU" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
[[ "$idle" == "$LOCK_MHZ" ]] || { echo "GPU $GPU not locked at $LOCK_MHZ (idle $idle)" | tee -a "$LOG"; exit 1; }
run(){ echo "=== $*" | tee -a "$LOG"
       CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CPUS" python -u bench/op_cost.py --clock-locked "$@" 2>&1 |
         grep -vE "Warning|warn\(" | tee -a "$LOG"; }

PREFILL_C=64,128,256,512,1024,2048,4096,8192
DECODE_B=1,8,16,32,64
T=4096,16384,32768,65536,131072,262144
for mt in Qwen_Qwen3-Next-80B-A3B-Instruct:2 Qwen_Qwen3.8-27B:1; do
  m="${mt%%:*}"; tp="${mt##*:}"; cfg="results/step00/configs/$m.json"; tag="${m#Qwen_}_tp${tp}_gpu$GPU"
  run --config "$cfg" --tp "$tp" --op moe --c "1,8,16,32,$PREFILL_C" --tag "s02_${tag}_moe"
  run --config "$cfg" --tp "$tp" --op fa_decode --c "$DECODE_B" --t "$T" --tag "s02_${tag}_fa_decode"
  run --config "$cfg" --tp "$tp" --op gdn_decode,dense_attn,dense_gdn,dense_mlp --c "$DECODE_B" --tag "s02_${tag}_decode_dense"
  # card check against the GPU 4 step 1 tables
  run --config "$cfg" --tp "$tp" --op fa,gdn --c 256,2048,8192 --t 0,65536,262144 --tag "s02_${tag}_cardcheck"
done
echo "=== all finished" | tee -a "$LOG"
