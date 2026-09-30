#!/usr/bin/env bash
# Step 0 follow-up: the demo at 1.3 req/s (paper Table 6) showed no Layered gain on 2xH200,
# where 1.3 req/s is far from saturation. Sweep the request rate as the fork's own driver does,
# chunked and layered on the same GPUs, one server per mode, into a fresh results dir.
#   tmux new -d -s rates 'bash bench/run_queue_step00_rates.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
while tmux has-session -t queue 2>/dev/null; do sleep 30; done   # wait for the step 0-1 queue
RATES="${RATES:-1.3 2.0 2.5 3.0 3.5}" MODES="${MODES:-chunked layered}" bash bench/step00_layered_demo.sh
