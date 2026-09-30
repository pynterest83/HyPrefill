#!/usr/bin/env bash
# Collect step 2 op_cost runs (written under results/step01 with an s02 tag) into results/step02
# and merge them into per-model cost tables (median over the cards that ran them).
#   bash bench/merge_step02.sh [date]
set -uo pipefail
cd "$(dirname "$0")/.."
DATE="${1:-$(date +%F)}"
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate hyprefill
mkdir -p results/step02/cost_tables/parts
for d in results/step01/*_s02_*; do [[ -e $d ]] && mv "$d" results/step02/; done
for mt in Qwen3-Next-80B-A3B-Instruct_tp2:"moe fa_decode decode_dense" Qwen3.8-27B_tp1:"fa_decode decode_dense" \
          Qwen3-30B-A3B_tp2:"prefill moe fa_decode decode_dense"; do
  m="${mt%%:*}"
  for part in ${mt#*:}; do
    runs=(results/step02/${DATE}_s02_${m}_gpu[0-9]_${part})
    [[ -e "${runs[0]}" ]] || { echo "no runs for $m $part"; continue; }
    python bench/analyze_step01.py --merge "${runs[@]}" --out "results/step02/cost_tables/parts/${m}_${part}.csv"
  done
  python - "$m" <<'PY'
import glob, sys, pandas as pd
m = sys.argv[1]
fs = sorted(glob.glob(f"results/step02/cost_tables/parts/{m}_*.csv"))
if fs:
    d = pd.concat([pd.read_csv(f) for f in fs]); d.to_csv(f"results/step02/cost_tables/{m}.csv", index=False)
    print(m, len(d), "rows", sorted(d.op.unique()), "runs per row:", sorted(d.n_runs.unique()))
PY
done
