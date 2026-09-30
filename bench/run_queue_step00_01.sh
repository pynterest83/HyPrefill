#!/usr/bin/env bash
# Everything left for steps 0-1 that needs the GPUs, run back to back so it survives a
# disconnect (tmux). Order matters only for GPU use: the demo uses GPUs 4-5, the rest GPU 4.
#   tmux new -d -s queue 'bash bench/run_queue_step00_01.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="$HOME/hyprefill_data/queue_$(date +%Y%m%d_%H%M%S).log"
say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

DEMO_DIR="${DEMO_DIR:-$(ls -dt results/step00/layered_demo/*/ | head -1)}"
say "1/4 step 0 demo: layered in $DEMO_DIR (chunked already there), then chunked, layered (interleaved A-B-A-B with the existing chunked run)"
OUT_DIR="$DEMO_DIR" MODES="layered chunked layered" bash bench/step00_layered_demo.sh; say "demo exit $?"

say "2/4 step 1 sweep with dense ops, pass A"
bash bench/run_step01.sh; say "exit $?"
say "3/4 step 1 sweep with dense ops, pass B"
bash bench/run_step01.sh; say "exit $?"
say "4/4 step 1 §2.4 intra/inter split"
bash bench/run_step01_intra_inter.sh; say "exit $?"
say "=== queue finished"
