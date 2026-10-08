#!/usr/bin/env python3
"""KT1 verdict (PROPOSAL §5, revised 2026-10-07 before running), from op_cost.py `moe_mixed` runs.

  inc(D, c) = moe_mixed(D, c) - moe_mixed(D, 0)             ms per layer, real routing, mean over draws
  s         = 48 * [inc(D, 512)/512 - inc(D, 2048)/2048]    ms saved per prefill token, 512 -> 2048
  f         = s / cost per prefill token of a uniform 512 chunk at t = 128K
              (step 1 tables, MoE replaced by inc; no all-reduce)
  GO     : inc(D, 2048) <= 0.6 * 4 * inc(D, 512)  and  f >= 10% at D = 32
  KILL   : f < 5% at both D = 32 and D = 64
  else   : undecided at KT1, KT2 decides

  python bench/kt1_eval.py results/step02/<date>_s02_..._kt1_draw [out.csv]   # prefix: matches <prefix>*/summary.csv
"""
import datetime, glob, json, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle import Costs, layers  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = "Qwen3-Next-80B-A3B-Instruct_tp2"


def main():
    prefix = sys.argv[1]
    files = sorted(glob.glob(f"{prefix}/summary.csv") + glob.glob(f"{prefix}*/summary.csv"))
    assert files, f"no summary.csv under {prefix}*"
    m = pd.concat([pd.read_csv(f).assign(run=f) for f in files])
    m = m[m.op == "moe_mixed"]
    print(f"{len(files)} runs; draws per (D, c): {m.groupby(['t', 'c']).size().min()}")
    cfg = json.load(open(REPO / "results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json"))
    cfg = cfg.get("text_config", cfg)
    n_attn, n_gdn, n = layers(cfg)
    tab = pd.concat([pd.read_csv(REPO / f"results/step0{i}/cost_tables/{MODEL}.csv") for i in (1, 2)],
                    ignore_index=True).drop_duplicates(["op", "c", "t"])
    cost = Costs(tab)

    rows, verdict = [], {}
    for d in sorted(m.t.unique()):
        g = m[m.t == d].groupby("c").p50_ms.agg(["mean", "std", "count"])
        base = g.loc[0, "mean"]
        inc = lambda c: float(g.loc[c, "mean"] - base)
        # non-MoE cost per prefill token of a uniform 512 chunk at t = 128K, whole model
        non_moe = (n_attn * (cost("fa", 512, 131072) + cost("dense_attn", 512))
                   + n_gdn * (cost("gdn", 512) + cost("dense_gdn", 512) + cost("gdn_conv", 512))) / 512
        per_tok = non_moe + n * inc(512) / 512
        s = n * (inc(512) / 512 - inc(2048) / 2048)
        f = s / per_tok
        ratio = inc(2048) / (4 * inc(512))
        rows.append(dict(D=int(d), moe0_ms=base, inc256=inc(256), inc512=inc(512), inc2048=inc(2048),
                         ratio_2048_vs_4x512=ratio, s_us_per_tok=s * 1e3, f_pct=100 * f))
        verdict[int(d)] = (ratio, f)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    go = verdict.get(32, (9, 0))[0] <= 0.6 and verdict.get(32, (9, 0))[1] >= 0.10
    kill = all(verdict[d][1] < 0.05 for d in (32, 64) if d in verdict) and 32 in verdict and 64 in verdict
    print("\nKT1:", "GO" if go else "KILL (MoE amortization branch)" if kill else "undecided -> KT2 decides")
    dst = sys.argv[2] if len(sys.argv) > 2 else REPO / f"results/step02/kt1_eval_{MODEL}_{datetime.date.today()}.csv"
    out.to_csv(dst, index=False, float_format="%.5f")
    print("->", dst)


if __name__ == "__main__":
    main()
