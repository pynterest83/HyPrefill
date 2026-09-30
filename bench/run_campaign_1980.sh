#!/usr/bin/env bash
# Full measurement campaign at the standard 1980 MHz lock, GPUs 4-7 dedicated (2026-09-29).
# Replaces the 1830 MHz data in ~/hyprefill_data/clk1830/. Four queues, each in its own tmux
# session; POST waits for A and B, then merges tables and runs the analyses.
#
#   tmux new -d -s cA    'QUEUE=A    bash bench/run_campaign_1980.sh'   # GPU 6: step 1, step 2, 30B, §2.4
#   tmux new -d -s cB    'QUEUE=B    bash bench/run_campaign_1980.sh'   # GPU 7: step 1, step 2, 30B, G1c, paged check
#   tmux new -d -s cDEMO 'QUEUE=DEMO bash bench/run_campaign_1980.sh'   # GPU 4-5: Layered vs chunked rate sweep
#   tmux new -d -s cPOST 'QUEUE=POST bash bench/run_campaign_1980.sh'   # merge, repeatability, §2.5 (GPU 6), oracle, sim
#
# Timing: A and B ~1 h, POST ~1 h after them, DEMO ~4.5 h.
set -uo pipefail
cd "$(dirname "$0")/.."
export LOCK_MHZ="${LOCK_MHZ:-1980}"
Q="${QUEUE:?QUEUE=A|B|DEMO|POST}"
DATE="${CAMPAIGN_DATE:-$(date +%F)}"
MARK="$HOME/hyprefill_data/campaign_$DATE"; mkdir -p "$MARK"
LOG="$MARK/queue_${Q}_$(date +%H%M%S).log"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
idle(){ nvidia-smi -i "$1" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' '; }
need_lock(){ for g in "$@"; do [[ "$(idle "$g")" == "$LOCK_MHZ" ]] || { say "GPU $g idle clock $(idle "$g") != $LOCK_MHZ; stop"; exit 1; }; done; }
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
py(){ local gpu=$1 cpus=$2; shift 2
      CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" python -u "$@" 2>&1 | grep -vE "Warning|warn\(" | tee -a "$LOG"; }

case "$Q" in
A|B)
  GPU=$([[ $Q == A ]] && echo 6 || echo 7); CPUS=72-95
  need_lock "$GPU"
  [[ $Q == A ]] && { say "environment snapshot"; bash scripts/record_env.sh "results/env/$DATE" | tee -a "$LOG"; }
  say "step 1: FA (paged KV), GDN, dense, GDN triton on GPU $GPU"
  GPU=$GPU CPUS=$CPUS bash bench/run_step01.sh; say "exit $?"
  say "step 2: MoE, decode on GPU $GPU";      GPU=$GPU CPUS=$CPUS bash bench/run_step02.sh; say "exit $?"
  say "step 2: Qwen3-30B-A3B on GPU $GPU";    GPU=$GPU CPUS=$CPUS bash bench/run_step02_q30b.sh; say "exit $?"
  if [[ $Q == A ]]; then
    say "step 1 §2.4 intra/inter";  GPU=$GPU CPUS=$CPUS bash bench/run_step01_intra_inter.sh; say "exit $?"
  else
    say "G1c indexer memory";                   py "$GPU" "$CPUS" bench/g1c_indexer_mem.py
    say "FA paged vs contiguous (Qwen3.8-27B)"; py "$GPU" "$CPUS" bench/fa_paged_check.py
    say "FA paged vs contiguous (Qwen3-Next TP2)"
    py "$GPU" "$CPUS" bench/fa_paged_check.py --config results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json --tp 2 --block 544
  fi
  touch "$MARK/$Q.done"; say "=== queue $Q finished" ;;

DEMO)
  need_lock 4 5
  say "Layered vs chunked, Qwen3-30B-A3B, arXiv, default torch.compile settings (as the authors run it)"
  RATES="${RATES:-1.3 2.0 2.5 2.6 2.7 2.8 2.9 3.0 3.5}" bash bench/run_step00_layered_check.sh; say "exit $?"
  D=$(ls -dt results/step00/layered_demo/*/ | head -1)
  python bench/summarize_demo.py "$D" | tee -a "$LOG"
  touch "$MARK/DEMO.done"; say "=== queue DEMO finished" ;;

POST)
  say "waiting for queues A and B"
  until [[ -e "$MARK/A.done" && -e "$MARK/B.done" ]]; do sleep 60; done
  mkdir -p results/step02
  for d in results/step01/*_s02_*; do [[ -e $d ]] && mv "$d" results/step02/; done
  say "merge step 1 tables (one run per card)"
  for m in Qwen3-Next-80B-A3B-Instruct_tp2 Qwen3-Next-80B-A3B-Instruct_tp2_gdn-triton Qwen3.8-27B_tp1 Qwen3.8-27B_tp1_gdn-triton; do
    runs=(results/step01/${DATE}_${m}_gpu6 results/step01/${DATE}_${m}_gpu7)
    python bench/analyze_step01.py --merge "${runs[@]}" --out "results/step01/cost_tables/$m.csv" | tee -a "$LOG"
    python bench/analyze_step01.py --repeat "${runs[@]}" > "results/step01/cost_tables/${m}_repeatability.txt"
    grep -E "^[a-z_]+:|^->" "results/step01/cost_tables/${m}_repeatability.txt" | tee -a "$LOG"
  done
  python bench/analyze_step01.py "results/step01/${DATE}_Qwen3-Next-80B-A3B-Instruct_tp2_gpu6" "results/step01/${DATE}_Qwen3.8-27B_tp1_gpu6" \
    --fig figures/step01_fig1.png > results/step01/cost_tables/analysis.txt 2>&1
  grep -E "^####|->" results/step01/cost_tables/analysis.txt | tee -a "$LOG"
  say "merge step 2 tables (GPU 6 + GPU 7)"
  mkdir -p results/step02/cost_tables/parts
  for mt in Qwen3-Next-80B-A3B-Instruct_tp2:"moe fa_decode decode_dense" Qwen3.8-27B_tp1:"fa_decode decode_dense" \
            Qwen3-30B-A3B_tp2:"prefill moe fa_decode decode_dense"; do
    m="${mt%%:*}"
    for part in ${mt#*:}; do
      python bench/analyze_step01.py --merge results/step02/${DATE}_s02_${m}_gpu[67]_${part} \
        --out "results/step02/cost_tables/parts/${m}_${part}.csv" | tee -a "$LOG"
    done
    python - "$m" <<'EOF' | tee -a "$LOG"
import glob, sys, pandas as pd
m = sys.argv[1]
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/step02/cost_tables/parts/{m}_*.csv"))])
d.to_csv(f"results/step02/cost_tables/{m}.csv", index=False); print(m, len(d), "rows", sorted(d.op.unique()))
EOF
  done
  say "§2.5 validation against vLLM prefill (Qwen3.8-27B TP1, GPU 6), clock logged"
  need_lock 6
  nvidia-smi -i 6 --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active \
    --format=csv,noheader -lms 200 > "$MARK/validate_gpu6.csv" & MON=$!
  py 6 72-95 bench/validate_forward.py --c 512,1024,2048,4096,8192 --t 0,16384,32768,65536 --repeats 5
  kill $MON
  say "oracle and token-flow simulator (CPU, NUMA 0)"
  for m in Qwen3-Next-80B-A3B-Instruct_tp2 Qwen3.8-27B_tp1; do
    nice -n 10 taskset -c 0-23,96-119 python bench/oracle.py --model "$m" > "results/step02/oracle_$m.txt" 2>&1
    nice -n 10 taskset -c 0-23,96-119 python bench/sim_tokenflow.py --model "$m" --jobs 32 > "results/step02/sim_$m.txt" 2>&1
  done
  nice -n 10 taskset -c 0-23,96-119 python bench/sim_tokenflow.py --model Qwen3-30B-A3B_tp2 --budgets 50,125 \
    --decode 8,32 --t 0,4096 --delta 8192 --jobs 32 > results/step02/sim_Qwen3-30B-A3B_tp2.txt 2>&1
  touch "$MARK/POST.done"; say "=== queue POST finished" ;;
esac
