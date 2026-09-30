#!/usr/bin/env bash
# Step 0 follow-up (plan/00 R5): recheck Layered vs chunked on 2xH200 with a
# finer rate grid around the knee to measure goodput (PROPOSAL §4.4). 1.3 req/s repeats the
# earlier run.
#   tmux new -d -s lpcheck 'bash bench/run_step00_layered_check.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
# Do NOT raise torch._dynamo's recompile limit (bench/pyhooks): with 256 the fork recompiled
# forward_attention 256 times and then paid 256 guard checks per call x 48 layers, doubling TPOT
# for both modes (2026-09-29, results in ~/hyprefill_data/invalid/). Run the fork as its authors
# do, with the default limit (8), under which it falls back to eager for forward_attention.
RATES="${RATES:-1.3 2.6 2.7 2.8 2.9 3.0}" MODES="${MODES:-chunked layered}" bash bench/step00_layered_demo.sh
