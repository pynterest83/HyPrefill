#!/usr/bin/env bash
# Step 0 demo: scale-snu/layered-prefill on Qwen3-30B-A3B, chunked vs layered prefill in the
# same engine, same GPUs, reproducing the fork's own protocol (benchmarks/benchmark_batch.py):
#   TP=2, max-model-len 32768, gpu-memory-utilization 0.85, max-num-seqs 256
#   chunked: max-num-batched-tokens 512,  num-stages 1
#   layered: max-num-batched-tokens 8192, num-stages 16
#   arXiv trace, 60 s warmup, then 600 s at the request rate (paper Table 6: 1.3 req/s)
# Purpose (plan/00 §2.4): does the fork run here, and is the Layered/chunked improvement
# on this machine close to the paper (Table 6: mean TTFT -56%)? Relative numbers only.
#
#   tmux new -d -s lp-demo 'bash bench/step00_layered_demo.sh'
#   MODES="chunked layered chunked layered" RATE=1.3 bash bench/step00_layered_demo.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LP="$REPO/third_party/layered-prefill"
GPUS="${GPUS:-4,5}"            # same NUMA node (CPUs 48-71)
CPUS="${CPUS:-48-71}"
RATES="${RATES:-${RATE:-1.3}}"   # one server per mode, rates swept on it (as the fork's driver)
MODES="${MODES-chunked layered}"   # set but empty = run no mode
WARMUP_S=60; BENCH_S=600
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
MODEL="$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)"

# OUT_DIR appends modes to an existing run (same day, same setup); otherwise a new dir
if [[ -n "${OUT_DIR:-}" ]]; then OUT="$(cd "$REPO" && realpath -m "$OUT_DIR")"; else  # absolute: we cd into the fork below
  OUT="$REPO/results/step00/layered_demo/$(date +%F)"
  i=2; base="$OUT"; while [[ -e "$OUT" ]]; do OUT="${base}_run$i"; i=$((i + 1)); done
fi
mkdir -p "$OUT"
LOG="$OUT/driver.log"

source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
set +u; conda activate layered-prefill; set -u
# the fork's own runtime environment (benchmark_batch.py ENV)
export PATH="$PATH:$CONDA_PREFIX/nvvm/bin" CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export CUDA_VISIBLE_DEVICES="$GPUS" TORCH_CUDA_ARCH_LIST="9.0"

say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
free_port(){ python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])"; }

# record the setup (docs/03_MEASUREMENT.md §5)
[[ -e "$OUT/config.json" ]] || python - "$OUT/config.json" <<EOF
import json, subprocess, sys, torch
q = "index,name,clocks.sm,clocks.max.sm,power.limit,memory.used,utilization.gpu"
gpus = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i", "$GPUS"],
                      capture_output=True, text=True).stdout.strip().splitlines()
commit = lambda d: subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"date": "$(date -Is)", "model": "$MODEL", "gpus": "$GPUS", "cpus": "$CPUS", "gpu_state": gpus,
           "torch": torch.__version__, "cuda": torch.version.cuda, "fork_commit": commit("$LP"),
           "repo_commit": commit("$REPO"), "rates": "$RATES", "warmup_s": $WARMUP_S, "bench_s": $BENCH_S,
           "modes": "$MODES", "tp": 2, "dynamo_recompile_limit": "${HYP_DYNAMO_RECOMPILE_LIMIT:-default (8)}", "pythonpath": "${PYTHONPATH:-}", "max_model_len": 32768, "gpu_memory_utilization": 0.85,
           "clock_idle_pinned_mhz": "see gpu_state (admin lock, 1980 from 2026-09-29)"}, open(sys.argv[1], "w"), indent=2)
EOF

cd "$LP"
for mode in $MODES; do
  if [[ "$mode" == layered ]]; then mnbt=8192; stages=16; else mnbt=512; stages=1; fi
  port=$(free_port); nccl=$(free_port)
  tag="${mode}_$(date +%H%M%S)"
  say "=== $tag: server (mnbt=$mnbt, stages=$stages)"
  # own session/process group: the engine spawns worker processes (multiprocessing.spawn)
  # that survive a plain kill of the server and keep ~120 GB on the GPUs (2026-09-28)
  setsid taskset -c "$CPUS" python -m nanovllm.entrypoints.api_server --model "$MODEL" \
    --max-num-batched-tokens $mnbt --max-num-seqs 256 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --log-level info \
    --host localhost --port $port --nccl-port $nccl \
    --schedule-mode "$mode-prefill" --num-stages $stages > "$OUT/server_$tag.log" 2>&1 &
  srv=$!
  until curl -sf -o /dev/null -X POST "localhost:$port/generate" -H 'Content-Type: application/json' \
        -d '{"model":"","prompt":"hi","max_tokens":1,"temperature":0.0,"stream":false}'; do
    kill -0 $srv 2>/dev/null || { say "server died, see server_$tag.log"; continue 2; }
    sleep 5
  done
  say "server ready"
  nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active \
    --format=csv,noheader -lms 200 > "$OUT/gpu_$tag.csv" &
  mon=$!
  for RATE in $RATES; do
    rtag="${tag}_r$RATE"
    n_warm=$(python -c "print(int($WARMUP_S * $RATE))"); n=$(python -c "print(int($BENCH_S * $RATE))")
    common=(--model "$MODEL" --endpoint /generate --backend nano-vllm --port $port --dataset-name arxiv
            --request-rate "$RATE" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
            --goodput ttft:200 tpot:20 e2el:20000 --seed 0)
    say "warmup: $n_warm requests at $RATE req/s"
    taskset -c "$CPUS" python benchmarks/benchmark_serving.py "${common[@]}" --num-prompts $n_warm >> "$OUT/client_${rtag}_warmup.log" 2>&1
    say "benchmark: $n requests at $RATE req/s"
    taskset -c "$CPUS" python benchmarks/benchmark_serving.py "${common[@]}" --num-prompts $n \
      --save-result --save-detailed --result-dir "$OUT" --result-filename "$rtag.json" > "$OUT/client_$rtag.log" 2>&1
    say "client exit $?"
    sleep 30   # let the queue drain before the next rate
  done
  kill $mon
  kill -TERM -- -"$srv" 2>/dev/null; sleep 20; kill -KILL -- -"$srv" 2>/dev/null; wait $srv 2>/dev/null
  for _ in $(seq 60); do
    [[ $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]] && break
    sleep 5
  done
  if [[ $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -ge 1000 ]]; then
    say "GPUs not released after 5 min; aborting"; exit 1
  fi
  say "GPUs released; relax 60 s"; sleep 60
done
say "=== all finished"
