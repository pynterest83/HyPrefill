#!/usr/bin/env bash
# Download HyPrefill models into HF_HOME, retrying until each one completes.
# Meant to run detached so it survives the VS Code / SSH session:
#   tmux new -d -s hyprefill-dl 'bash scripts/download_models.sh'
#   tmux attach -t hyprefill-dl          # watch; Ctrl-b d to detach again
# Pass repo ids to override the default list. Safe to re-run: hf download resumes.
set -uo pipefail

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
LOG_DIR="${LOG_DIR:-$HOME/hyprefill_data}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
MAX_TRIES="${MAX_TRIES:-20}"
mkdir -p "$HF_HOME" "$LOG_DIR"
LOG="$LOG_DIR/download_$(date +%Y%m%d_%H%M).log"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hyprefill}"

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(Qwen/Qwen3-Next-80B-A3B-Instruct Qwen/Qwen3.8-27B Qwen/Qwen3-30B-A3B)
fi

say(){ printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
say "HF_HOME=$HF_HOME  models: ${MODELS[*]}"
for m in "${MODELS[@]}"; do
  for ((i = 1; i <= MAX_TRIES; i++)); do
    say "--- $m (try $i/$MAX_TRIES)"
    if hf download "$m" >>"$LOG" 2>&1; then
      say "DONE $m  ($(du -shL "$HF_HOME"/hub/models--${m//\//--}/snapshots/*/ | cut -f1))"
      break
    fi
    say "retry $m in 60 s"
    sleep 60
  done
  ((i > MAX_TRIES)) && say "FAILED $m after $MAX_TRIES tries"
done
say "all finished"
