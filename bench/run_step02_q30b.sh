#!/usr/bin/env bash
# Cost tables for Qwen3-30B-A3B (TP2, full attention + MoE), the model of the step 0 Layered
# demo, so the token-flow simulator can be checked against the real chunked-vs-layered result.
#   tmux new -d -s q30g7 'GPU=7 CPUS=72-95 bash bench/run_step02_q30b.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-7}"; CPUS="${CPUS:-72-95}"; LOCK_MHZ="${LOCK_MHZ:-1980}"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
LOG="$HOME/hyprefill_data/step02_q30b_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
idle=$(nvidia-smi -i "$GPU" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
[[ "$idle" == "$LOCK_MHZ" ]] || { echo "GPU $GPU not locked" | tee -a "$LOG"; exit 1; }
run(){ echo "=== $*" | tee -a "$LOG"
       CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CPUS" python -u bench/op_cost.py --clock-locked "$@" 2>&1 |
         grep -vE "Warning|warn\(" | tee -a "$LOG"; }
cfg=results/step00/configs/Qwen_Qwen3-30B-A3B.json; tag="Qwen3-30B-A3B_tp2_gpu$GPU"
run --config "$cfg" --tp 2 --op fa,dense_attn --t 0,4096,16384,32768 --tag "s02_${tag}_prefill"
run --config "$cfg" --tp 2 --op moe --c 1,8,16,32,64,128,256,512,1024,2048,4096,8192 --tag "s02_${tag}_moe"
run --config "$cfg" --tp 2 --op fa_decode --c 1,8,16,32,64 --t 4096,16384,32768 --tag "s02_${tag}_fa_decode"
run --config "$cfg" --tp 2 --op dense_attn --c 1,8,16,32,64 --tag "s02_${tag}_decode_dense"
echo "=== all finished" | tee -a "$LOG"
