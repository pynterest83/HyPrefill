#!/usr/bin/env bash
# Step 1 sweep: FA and GDN cost vs (c, t) for the GDN hybrids, on one GPU.
# Run detached:  tmux new -d -s step01 'bash bench/run_step01.sh'
# GPU and NUMA-local CPUs default to GPU 4 / CPUs 48-71 (see AGENTS.md, Server).
set -uo pipefail  # pipefail: the lock check below relies on it
cd "$(dirname "$0")/.."

GPU="${GPU:-4}"
CPUS="${CPUS:-48-71}"
EXTRA=("$@")   # e.g. --clock-locked, --allow-busy
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hyprefill}"
LOG="$HOME/hyprefill_data/step01_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

# Numbers for the paper need locked clocks (docs/03_MEASUREMENT.md). Root in this container
# cannot lock, so the host admin locks the GPUs; we then only verify. If we can lock
# ourselves, lock only this GPU and give it back at the end. LOCK=0 for exploratory runs.
LOCK_MHZ="${LOCK_MHZ:-1980}"   # standard lock for all numbers (2026-09-29; 1830 before, archived)
if [[ "${LOCK:-1}" == 1 ]]; then
  if sudo nvidia-smi -i "$GPU" -lgc "$LOCK_MHZ,$LOCK_MHZ" >>"$LOG" 2>&1; then
    trap 'sudo nvidia-smi -i "$GPU" -rgc | tee -a "$LOG"' EXIT
    trap 'exit 1' HUP INT TERM
    echo "locked GPU $GPU at $LOCK_MHZ MHz ourselves" | tee -a "$LOG"
  else
    # Not allowed to lock: accept only if the GPU is already pinned at LOCK_MHZ while
    # idle (an unlocked idle H200 drops to ~345 MHz). Leave the admin's lock in place.
    idle=$(nvidia-smi -i "$GPU" --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$idle" != "$LOCK_MHZ" ]]; then
      echo "GPU $GPU idle SM clock is $idle MHz, not locked at $LOCK_MHZ; ask the admin, or LOCK=0 (exploratory only)" | tee -a "$LOG"
      exit 1
    fi
    echo "GPU $GPU already locked at $LOCK_MHZ MHz (idle clock pinned); leaving it as is" | tee -a "$LOG"
  fi
  EXTRA+=(--clock-locked)
fi

run(){ echo "=== $*" | tee -a "$LOG"
       CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CPUS" python -u bench/op_cost.py "$@" "${EXTRA[@]}" 2>&1 |
         grep -vE "Warning|warn\(" | tee -a "$LOG"; }

# model:TP as served (PROPOSAL §4.1). Kernel costs are per GPU, so measure the per-GPU
# shapes of the TP the model actually runs at; Qwen3-Next bf16 does not fit on one H200.
for mt in Qwen_Qwen3-Next-80B-A3B-Instruct:2 Qwen_Qwen3.8-27B:1; do
  m="${mt%%:*}"; tp="${mt##*:}"
  cfg="results/step00/configs/$m.json"
  run --config "$cfg" --tp "$tp" --op fa,gdn,dense_attn,dense_gdn,gdn_conv,dense_mlp --tag "${m#Qwen_}_tp${tp}_gpu$GPU"
  # vLLM's alternative GDN backend, to see how much of the GDN cost is the FlashInfer wrapper
  run --config "$cfg" --tp "$tp" --op gdn --gdn-backend triton --tag "${m#Qwen_}_tp${tp}_gdn-triton_gpu$GPU"
done
echo "=== all finished" | tee -a "$LOG"
