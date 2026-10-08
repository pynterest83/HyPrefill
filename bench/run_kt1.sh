#!/usr/bin/env bash
# Routing dump (192 arXiv requests, Qwen3-Next TP2) -> KT1 (moe_mixed, real routing, 3 draws)
# -> KT1 verdict -> KT2 rerun with the measured MoE. Needs 2 free GPUs of one NUMA node, clock locked.
#   tmux new -d -s kt1 'GPUS=4,5 CPUS=48-71 bash bench/run_kt1.sh'
# Resumable: the dump and each KT1 draw are skipped when already present.
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS="${GPUS:-4,5}"; CPUS="${CPUS:-48-71}"; LOCK_MHZ="${LOCK_MHZ:-1980}"; N="${N:-192}"
GPU0="${GPUS%%,*}"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-hyprefill}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
DATE="$(date +%F)"; LOG="$HOME/hyprefill_data/kt1_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$HOME/hyprefill_data"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/*/ | head -1)
CFG=results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json

check_gpus(){  # free and locked
  for g in ${GPUS//,/ }; do
    read -r used util clk < <(nvidia-smi -i "$g" --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ',')
    (( used < 1000 && util < 5 )) || { say "GPU $g busy ($used MiB, $util%); stop"; exit 1; }
    [[ "$clk" == "$LOCK_MHZ" ]] || { say "GPU $g clock $clk != $LOCK_MHZ (not locked); stop"; exit 1; }
  done
}

say "kt1 pipeline: GPUS=$GPUS CPUS=$CPUS N=$N model=$MODEL"
check_gpus

DUMP=$(ls -d "$HOME"/hyprefill_data/step02/moe_routing/*Qwen3-Next*_arxiv 2>/dev/null | tail -1)
if [[ -n "$DUMP" && $(ls "$DUMP"/req_*.npz 2>/dev/null | wc -l) -ge $N && -f "$DUMP/meta.json" ]]; then
  say "dump exists: $DUMP"
else
  say "dumping routing ($N requests)"
  CUDA_VISIBLE_DEVICES="$GPUS" taskset -c "$CPUS" python -u bench/moe_routing_dump.py \
    --model "$MODEL" --tp 2 --dataset arxiv --n "$N" --max-prompt 16384 2>&1 | grep -vE "Warning|warn\(" | tee -a "$LOG"
  DUMP=$(ls -d "$HOME"/hyprefill_data/step02/moe_routing/*Qwen3-Next*_arxiv | tail -1)
  [[ -f "$DUMP/meta.json" ]] || { say "dump failed"; exit 1; }
fi
say "dump: $DUMP"

check_gpus
for s in 0 1 2; do
  tag="s02_Qwen3-Next-80B-A3B-Instruct_tp2_gpu${GPU0}_kt1_draw$s"
  # done = 3 decode batches x 9 chunk sizes; a crashed run leaves a short summary.csv
  done_rows=$(cat results/step0[12]/*_"$tag"/summary.csv 2>/dev/null | grep -c "^moe_mixed")
  (( done_rows >= 27 )) && { say "draw $s exists"; continue; }
  ls -d results/step0[12]/*_"$tag" >/dev/null 2>&1 && { say "draw $s: incomplete earlier run present, remove it first"; exit 1; }
  say "KT1 draw $s"
  CUDA_VISIBLE_DEVICES="$GPU0" taskset -c "$CPUS" python -u bench/op_cost.py --clock-locked \
    --config "$CFG" --tp 2 --op moe_mixed --c 0,64,128,256,512,1024,2048,4096,8192 --decode-batch 8,32,64 \
    --routing "$DUMP" --routing-seed "$s" --tag "$tag" 2>&1 | grep -vE "Warning|warn\(" | tee -a "$LOG"
  done_rows=$(cat results/step0[12]/*_"$tag"/summary.csv 2>/dev/null | grep -c "^moe_mixed")
  (( done_rows >= 27 )) || { say "draw $s incomplete ($done_rows/27 rows); stop"; exit 1; }
done

D=$(ls -d results/step01/*_gpu${GPU0}_kt1_draw0 results/step02/*_gpu${GPU0}_kt1_draw0 2>/dev/null | head -1)
PFX="${D%_draw0}_draw"
say "KT1 verdict ($PFX)"
python bench/kt1_eval.py "$PFX" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || { say "KT1 verdict failed; stop"; exit 1; }
say "KT2 with measured MoE"
python bench/kt2_capacity.py --kt1 "$PFX" 2>&1 | tee -a "$LOG"
say "all finished"
