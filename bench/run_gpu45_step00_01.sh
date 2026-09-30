#!/usr/bin/env bash
# GPUs 4-5, in plan order: step 1 all-reduce (TP2 needs both GPUs, a few minutes), then the
# remaining step 0 item: Layered vs chunked on H200 (bench/run_campaign_1980.sh QUEUE=DEMO).
#   tmux new -d -s g45 'bash bench/run_gpu45_step00_01.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export LOCK_MHZ="${LOCK_MHZ:-1980}"
LOG="$HOME/hyprefill_data/gpu45_$(date +%Y%m%d_%H%M%S).log"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
for g in 4 5; do
  [[ "$(nvidia-smi -i $g --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')" == "$LOCK_MHZ" ]] ||
    { say "GPU $g not locked at $LOCK_MHZ"; exit 1; }
done
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
say "step 1: TP2 all-reduce, hidden 2048 (Qwen3-Next)"
CUDA_VISIBLE_DEVICES=4,5 taskset -c 48-71 torchrun --nproc-per-node 2 bench/allreduce_cost.py --hidden 2048 \
  --tag Qwen3-Next-80B-A3B-Instruct_tp2 2>&1 | grep -E "^n=|wrote|Error|Traceback" | tee -a "$LOG"
say "step 0: Layered vs chunked on H200"
QUEUE=DEMO bash bench/run_campaign_1980.sh; say "exit $?"
say "=== GPUs 4-5 finished"
