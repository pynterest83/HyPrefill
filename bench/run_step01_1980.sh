#!/usr/bin/env bash
# Step 1 (plan/01) at the standard 1980 MHz lock; also closes step 0's repeatability item.
# Sweep once per card (GPU 6, 7) in parallel -> two passes; then §2.4 on GPU 6 and, after the
# tables are merged, §2.5 (vLLM prefill vs predicted) on GPU 7.
#   tmux new -d -s step01 'bash bench/run_step01_1980.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export LOCK_MHZ="${LOCK_MHZ:-1980}"
DATE=$(date +%F)
LOG="$HOME/hyprefill_data/step01_1980_$(date +%Y%m%d_%H%M%S).log"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
for g in 6 7; do
  [[ "$(nvidia-smi -i $g --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')" == "$LOCK_MHZ" ]] ||
    { say "GPU $g not locked at $LOCK_MHZ"; exit 1; }
done
say "environment snapshot"; bash scripts/record_env.sh "results/env/$DATE" | tee -a "$LOG"

say "sweep: FA (paged KV), GDN, dense, GDN triton — GPU 6 and GPU 7 in parallel"
GPU=6 CPUS=72-95 bash bench/run_step01.sh > /dev/null & A=$!
GPU=7 CPUS=72-95 bash bench/run_step01.sh > /dev/null & B=$!
wait $A $B; say "sweeps done"

say "merge tables and check repeatability (GPU 6 vs GPU 7)"
for m in Qwen3-Next-80B-A3B-Instruct_tp2 Qwen3-Next-80B-A3B-Instruct_tp2_gdn-triton Qwen3.8-27B_tp1 Qwen3.8-27B_tp1_gdn-triton; do
  runs=(results/step01/${DATE}_${m}_gpu6 results/step01/${DATE}_${m}_gpu7)
  python bench/analyze_step01.py --merge "${runs[@]}" --out "results/step01/cost_tables/$m.csv" | tee -a "$LOG"
  python bench/analyze_step01.py --repeat "${runs[@]}" > "results/step01/cost_tables/${m}_repeatability.txt"
  grep -E "^[a-z_]+:|^->" "results/step01/cost_tables/${m}_repeatability.txt" | tee -a "$LOG"
done
python bench/analyze_step01.py results/step01/${DATE}_Qwen3-Next-80B-A3B-Instruct_tp2_gpu6 \
  results/step01/${DATE}_Qwen3.8-27B_tp1_gpu6 --fig figures/step01_fig1.png > results/step01/cost_tables/analysis.txt 2>&1
grep -E "^####|->" results/step01/cost_tables/analysis.txt | tee -a "$LOG"

say "§2.4 intra/inter (GPU 6) and §2.5 validation (GPU 7) in parallel"
GPU=6 CPUS=72-95 bash bench/run_step01_intra_inter.sh > /dev/null & A=$!
nvidia-smi -i 7 --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active \
  --format=csv,noheader -lms 200 > "$HOME/hyprefill_data/validate_gpu7_$DATE.csv" & MON=$!
CUDA_VISIBLE_DEVICES=7 taskset -c 72-95 python -u bench/validate_forward.py --c 512,1024,2048,4096,8192 \
  --t 0,16384,32768,65536 --repeats 5 2>&1 | grep -E "^c=" | tee -a "$LOG"
kill $MON; wait $A
say "§2.5 per-step wall vs GPU busy (CPU cost per iteration), Qwen3.8-27B TP1, GPU 6"
CUDA_VISIBLE_DEVICES=6 taskset -c 72-95 python -u bench/profile_vllm_step.py --sweep 512,1024,2048,4096,8192 \
  --t-list 0,16384,65536 --tp 1 --tag Qwen3.8-27B_tp1 2>&1 | grep -E "^c=|wrote|Error|Traceback" | tee -a "$LOG"
say "=== step 1 finished"
