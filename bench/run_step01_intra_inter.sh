#!/usr/bin/env bash
# plan/01 §2.4: split GDN cost into intra-chunk O(C^2 d) and inter-chunk O(C d^2) parts by
# varying heads and d_k/d_v around the Qwen3-Next TP2 per-GPU shape (k 8, v 16, d 128).
# t is fixed at 0: step 1 showed GDN cost does not depend on t.
#   tmux new -d -s step01b 'bash bench/run_step01_intra_inter.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-4}"; CPUS="${CPUS:-48-71}"; LOCK_MHZ="${LOCK_MHZ:-1980}"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hyprefill}"
LOG="$HOME/hyprefill_data/step01b_$(date +%Y%m%d_%H%M%S).log"
idle=$(nvidia-smi -i "$GPU" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
[[ "$idle" == "$LOCK_MHZ" ]] || { echo "GPU $GPU not locked at $LOCK_MHZ (idle $idle)" | tee -a "$LOG"; exit 1; }

cfg=results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json
C=64,128,256,512,1024,2048,4096,8192
run(){ echo "=== $*" | tee -a "$LOG"
       CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CPUS" python -u bench/op_cost.py --config "$cfg" --tp 2 \
         --op gdn --c "$C" --t 0 --clock-locked "$@" 2>&1 | grep -vE "Warning|warn\(" | tee -a "$LOG"; }

for be in flashinfer triton; do
  # heads (GVA ratio 2 kept), d = 128
  for hv in 8 16 32; do run --gdn-backend $be --shape "gdn.v_heads=$hv,gdn.k_heads=$((hv / 2))" --tag "intra_${be}_hv$hv"; done
  # key/value head dim, heads as served (k 8, v 16)
  for d in 64 256; do run --gdn-backend $be --shape "gdn.d_k=$d" --tag "intra_${be}_dk$d"; done
  for d in 64 256; do run --gdn-backend $be --shape "gdn.d_v=$d" --tag "intra_${be}_dv$d"; done
done
echo "=== all finished" | tee -a "$LOG"
