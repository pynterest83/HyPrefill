#!/usr/bin/env bash
# Snapshot the software and GPU state behind a measurement campaign (reproducibility):
# pip freeze + conda list of both envs, driver/CUDA, GPU clocks/limits, repo commits.
#   bash scripts/record_env.sh [out_dir]      # default results/env/<date>
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/env/$(date +%F)}"; mkdir -p "$OUT"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
for env in hyprefill layered-prefill; do
  conda list -n "$env" --export > "$OUT/conda_$env.txt" 2>/dev/null
  conda run -n "$env" pip freeze > "$OUT/pip_$env.txt" 2>/dev/null
done
{ nvidia-smi --query-gpu=index,name,driver_version,vbios_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,power.max_limit,persistence_mode,memory.total --format=csv
  echo; /usr/local/cuda/bin/nvcc --version | tail -2
  echo; uname -a; lscpu | grep -E "Model name|^CPU\(s\)|NUMA node[0-9] CPU|max MHz"
  # single-thread speed of the engine loop depends on these (governor differed per NUMA node, 2026-10-09)
  echo; echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  echo "intel_pstate no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
  for n in /sys/devices/system/node/node[0-9]*; do
    c=$(cut -d- -f1 "$n/cpulist" | cut -d, -f1)
    echo "$(basename "$n") cpus $(cat "$n/cpulist"): governor $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null)," \
         "epp $(cat /sys/devices/system/cpu/cpu$c/cpufreq/energy_performance_preference 2>/dev/null)"
  done
} > "$OUT/system.txt" 2>&1
{ echo "repo $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
  for d in third_party/vllm third_party/layered-prefill third_party/layered-prefill/flash-attention; do
    [[ -d $d/.git || -f $d/.git ]] && echo "$d $(git -C "$d" rev-parse HEAD)"
  done
} > "$OUT/commits.txt"
echo "wrote $OUT"
