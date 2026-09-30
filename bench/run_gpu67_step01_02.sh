#!/usr/bin/env bash
# GPUs 6-7 after the first step 2 pass (2026-09-30):
#   in parallel: GPU 6 = §2.5 with concurrent decode (Qwen3.8-27B TP1, provisional GPU 6 tables;
#                measured times are what matter, predictions are recomputed from final tables)
#                GPU 7 = step 2 tables, second card (for the 2-card median)
#   then:        GPUs 6+7 = real MoE routing dump, Qwen3-Next TP2 (step 2, Việc 4); final merge
#   tmux new -d -s g67 'bash bench/run_gpu67_step01_02.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export LOCK_MHZ="${LOCK_MHZ:-1980}"
LOG="$HOME/hyprefill_data/gpu67_$(date +%Y%m%d_%H%M%S).log"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
for g in 6 7; do
  [[ "$(nvidia-smi -i $g --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')" == "$LOCK_MHZ" ]] ||
    { say "GPU $g not locked at $LOCK_MHZ"; exit 1; }
done
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
say "GPU 6: §2.5 with concurrent decode | GPU 7: step 2 tables, second card"
( CUDA_VISIBLE_DEVICES=6 taskset -c 72-83 python -u bench/validate_forward.py --c 512,2048,8192 --t 16384,65536 \
    --decode-batch 8,32,64 --repeats 3 2>&1 | grep -vE "Warning|warn\(|INFO" > "$HOME/hyprefill_data/validate_decode_$(date +%H%M).log" ) & A=$!
( GPU=7 CPUS=84-95 bash bench/run_step02.sh > /dev/null; GPU=7 CPUS=84-95 bash bench/run_step02_q30b.sh > /dev/null ) & B=$!
wait $A; say "§2.5 with decode: exit $?"
wait $B; say "step 2 second card: done"
say "final step 2 merge (2 cards)"; bash bench/merge_step02.sh 2>&1 | grep -v Warn | tee -a "$LOG"
say "GPUs 6+7: MoE routing dump, Qwen3-Next TP2, arXiv, 64 requests"
M=$(ls -d "${HF_HOME:-$HOME/hf_cache}"/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/*/ | head -1)
CUDA_VISIBLE_DEVICES=6,7 taskset -c 72-95 python -u bench/moe_routing_dump.py --model "$M" --tp 2 --dataset arxiv \
  --n 64 --max-prompt 16384 --gen 128 2>&1 | grep -E "wrote|Error|Traceback|assert" | tee -a "$LOG"
say "=== GPUs 6-7 finished"
