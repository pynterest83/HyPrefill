#!/usr/bin/env bash
# Re-measure everything that was CPU-bound before (after a CPU / container change), with the same
# settings as the original runs, so the numbers can be compared one to one:
#   A (GPUs 6-7, CPUs 72-95): host_ms of the eager GDN / FA calls (cost-table cells, Qwen3-Next TP2),
#     §2.5 per-step wall vs GPU busy sweep for Qwen3.8-27B TP1 (as 2026-09-29) and Qwen3-Next TP2
#   B (GPUs 4-5, CPUs 48-71): nsys profile of the layered-prefill fork, chunked vs layered (as 2026-09-30)
#   tmux new -d -s cpuA 'QUEUE=A bash bench/run_cpu_recheck.sh'
#   tmux new -d -s cpuB 'QUEUE=B bash bench/run_cpu_recheck.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
Q="${QUEUE:?QUEUE=A|B}"; LOCK_MHZ="${LOCK_MHZ:-1980}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}" LOCK_MHZ
LOG="$HOME/hyprefill_data/cpu_recheck_${Q}_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$HOME/hyprefill_data"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
need(){ for g in "$@"; do  # free and locked
  read -r used util clk < <(nvidia-smi -i "$g" --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ',')
  (( used < 1000 && util < 5 )) || { say "GPU $g busy ($used MiB, $util%); stop"; exit 1; }
  [[ "$clk" == "$LOCK_MHZ" ]] || { say "GPU $g clock $clk != $LOCK_MHZ; stop"; exit 1; }
done; }
snap(){ ls -d "$HF_HOME"/hub/models--"$1"/snapshots/*/ | head -1; }
say "queue $Q, cpu.max $(cat /sys/fs/cgroup/cpu.max), governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"

case "$Q" in
A)
  source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
  CPUS=72-95
  need 6
  say "host_ms: GDN / FA eager calls, Qwen3-Next TP2 (cost-table cells)"
  CUDA_VISIBLE_DEVICES=6 taskset -c $CPUS python -u bench/op_cost.py --clock-locked \
    --config results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json --tp 2 --op fa,gdn,gdn_decode,fa_decode \
    --c 64,512,2048,8192 --t 0,65536,262144 --tag cpu_recheck_Qwen3-Next-80B-A3B-Instruct_tp2_gpu6 2>&1 |
    grep -vE "Warning|warn\(" | tee -a "$LOG"
  need 6
  say "§2.5 sweep, Qwen3.8-27B TP1 (as 2026-09-29)"
  CUDA_VISIBLE_DEVICES=6 taskset -c $CPUS python -u bench/profile_vllm_step.py --model "$(snap Qwen--Qwen3.8-27B)" --tp 1 \
    --sweep 512,1024,2048,4096,8192 --t-list 0,16384,65536 --repeats 5 --tag Qwen3.8-27B_tp1 2>&1 |
    grep -vE "Warning|warn\(|INFO|DEBUG" | tee -a "$LOG"
  need 6 7
  say "§2.5 sweep, Qwen3-Next TP2 (first run)"
  CUDA_VISIBLE_DEVICES=6,7 taskset -c $CPUS python -u bench/profile_vllm_step.py --model "$(snap Qwen--Qwen3-Next-80B-A3B-Instruct)" \
    --tp 2 --sweep 512,1024,2048,4096,8192 --t-list 0,16384,65536 --repeats 5 --tag Qwen3-Next-80B-A3B-Instruct_tp2 2>&1 |
    grep -vE "Warning|warn\(|INFO|DEBUG" | tee -a "$LOG"
  ;;
B)
  need 4 5
  say "layered-prefill fork profile, chunked vs layered (as 2026-09-30)"
  GPUS=4,5 CPUS=48-71 bash bench/step00_layered_profile.sh 2>&1 | tee -a "$LOG"
  ;;
esac
say "queue $Q finished"
