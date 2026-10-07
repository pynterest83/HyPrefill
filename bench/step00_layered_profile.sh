#!/usr/bin/env bash
# Step 0 check: where does the step time go in the layered-prefill fork on H200, chunked vs
# layered? One server per mode on GPUs 4-5, arXiv trace at RATE req/s for DURATION s; the
# server runs under Nsight Systems (PROFILER=nsys, default: CUDA + NVTX trace of every process,
# window --delay NSYS_DELAY s after launch for NSYS_DURATION s, during the benchmark; CPU
# sampling is unavailable in this container) or with the sitecustomize torch-profiler hook
# (PROFILER=torch: ModelRunner.run calls START..START+CALLS in rank 0). Observation only.
#   tmux new -d -s lpprof 'bash bench/step00_layered_profile.sh'
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; LP="$REPO/third_party/layered-prefill"
GPUS="${GPUS:-4,5}"; CPUS="${CPUS:-48-71}"; RATE="${RATE:-2.5}"; DURATION="${DURATION:-240}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
MODEL="$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)"
OUT="${OUT_DIR:-$HOME/hyprefill_data/step00/layered_profile/$(date +%F_%H%M)}"; mkdir -p "$OUT"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; set +u; conda activate layered-prefill; set -u
export PATH="$PATH:$CONDA_PREFIX/nvvm/bin" CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux" CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO/bench/pyhooks"
cd "$LP"
for mode in ${MODES:-chunked layered}; do
  if [[ $mode == layered ]]; then mnbt=8192; stages=16; else mnbt=512; stages=1; fi
  port=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
  nccl=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
  if [[ "${PROFILER:-nsys}" == nsys ]]; then
    mkdir -p "$OUT/$mode"
    # --kill none: by default nsys kills the traced processes when --duration ends; the fork's
    # engine processes died, the api_server survived and the client hung (2026-09-30)
    # --cuda-graph-trace node: the fork runs MoE and GEMMs inside CUDA graphs; without it nsys
    # records each graph launch as one block and the per-kernel summary misses them (2026-09-30)
    WRAP=(nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --kill none --cuda-graph-trace node --delay "${NSYS_DELAY:-200}" \
          --duration "${NSYS_DURATION:-60}" -o "$OUT/$mode/trace" --force-overwrite true)
  else
    WRAP=(env HYP_PROFILE_DIR="$OUT/$mode" HYP_PROFILE_START="${START:-1500}" HYP_PROFILE_CALLS="${CALLS:-300}")
  fi
  setsid taskset -c "$CPUS" "${WRAP[@]}" python -m nanovllm.entrypoints.api_server --model "$MODEL" --max-num-batched-tokens $mnbt \
    --max-num-seqs 256 --max-model-len 32768 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --log-level info \
    --host localhost --port $port --nccl-port $nccl --schedule-mode "$mode-prefill" --num-stages $stages > "$OUT/server_$mode.log" 2>&1 &
  srv=$!
  until curl -sf -o /dev/null -X POST "localhost:$port/generate" -H 'Content-Type: application/json' \
      -d '{"model":"","prompt":"hi","max_tokens":1,"temperature":0.0,"stream":false}'; do
    kill -0 $srv 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
  n=$(python -c "print(int($DURATION * $RATE))")
  python "$REPO/bench/cgroup_cpu.py" snap > "$OUT/.cg_$mode.json"  # container CPU throttling, see bench/cgroup_cpu.py
  taskset -c "$CPUS" python benchmarks/benchmark_serving.py --model "$MODEL" --endpoint /generate --backend nano-vllm \
    --port $port --dataset-name arxiv --request-rate "$RATE" --num-prompts $n --seed 0 > "$OUT/client_$mode.log" 2>&1
  python "$REPO/bench/cgroup_cpu.py" delta "$OUT/.cg_$mode.json" --tag "$mode" --csv "$OUT/cgroup_cpu.csv"; rm -f "$OUT/.cg_$mode.json"
  sleep 30  # let nsys finish writing the report
  ls "$OUT/$mode"/* >/dev/null 2>&1 && echo "$mode: trace written" || echo "$mode: NO trace"
  if ls "$OUT/$mode"/*.nsys-rep >/dev/null 2>&1; then
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$OUT/$mode/kern" "$OUT/$mode"/*.nsys-rep >/dev/null 2>&1
    nsys stats -q -r nvtx_sum -f csv -o "$OUT/$mode/nvtx" "$OUT/$mode"/*.nsys-rep >/dev/null 2>&1
  fi
  kill -TERM -- -"$srv" 2>/dev/null; sleep 20; kill -KILL -- -"$srv" 2>/dev/null; wait $srv 2>/dev/null
  # under nsys the server runs in nsys' own process group, out of reach of the kill above
  pkill -TERM -f "^[^ ]*python -m nanovllm.entrypoints.api_server.*--port $port" 2>/dev/null; sleep 10
  pkill -KILL -f "^[^ ]*python -m nanovllm.entrypoints.api_server.*--port $port" 2>/dev/null
  pkill -KILL -f "^/home/quangch1/miniforge3/envs/layered-prefill/bin/python" 2>/dev/null
  for _ in $(seq 60); do [[ $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]] && break; sleep 5; done
done
conda activate hyprefill
python "$REPO/bench/analyze_fork_profile.py" "$OUT"
