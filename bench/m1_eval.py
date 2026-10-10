#!/usr/bin/env python3
"""M1 verdict (PROPOSAL §5, criteria fixed 2026-10-10 before M1 ran) from bench/hyprefill_emulator.py runs.

Per cell (B, D, t, delta): each policy's R = best R among its configurations whose iterations stay
within B (p99 <= B) in graph mode; ratio = HyPrefill / max(chunked, Layered); only cells where that
baseline reaches R >= 100 tokens/iteration count.
  GO:   ratio >= 1.30 in at least one cell AND >= 1.15 in at least three cells
  KILL: ratio < 1.10 in every cell
  else: in between (go on to M2, flagged as unlikely to reach an A* result)
Eager-mode numbers are reported next to it, not used for the verdict.

  python bench/m1_eval.py results/step05/<date>_m1_*      # one or more run directories
"""
import pathlib, sys

import numpy as np
import pandas as pd


def best(d, mode):
    x = d[(d["mode"] == mode) & (d.policy != "decode_only")]
    ok = x[x.within_B == 1]
    g = ok.groupby(["B", "D", "t", "delta", "policy"]).R.max().unstack("policy")
    for p in ("chunked", "layered", "hyprefill"):
        if p not in g:
            g[p] = np.nan
    g = g.fillna(0.0)
    g["hyprefill"] = g[["hyprefill", "layered"]].max(axis=1)  # HyPrefill's configuration space includes k = 1
    g["baseline"] = g[["chunked", "layered"]].max(axis=1)
    g["ratio"] = g.hyprefill / g.baseline.where(g.baseline > 0)
    return g


def main():
    dirs = [pathlib.Path(x) for x in sys.argv[1:]]
    d = pd.concat([pd.read_csv(p / "summary.csv") for p in dirs], ignore_index=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    dec = d[d.policy == "decode_only"][["D", "t", "iter_p50_ms", "decode_ms"]]
    print("decode-only iteration (GPU ms):"); print(dec.to_string(index=False))
    g = best(d, "graph")
    e = best(d, "eager") if (d["mode"] == "eager").any() else None
    if e is not None:
        g["ratio_eager"] = e.ratio
    print("\nbest R per policy within B (graph mode), ratio = HyPrefill / max(chunked, Layered):")
    print(g.round(3).to_string())
    v = g[g.baseline >= 100].ratio.dropna()
    n130, n115, n110 = int((v >= 1.30).sum()), int((v >= 1.15).sum()), int((v >= 1.10).sum())
    if n130 >= 1 and n115 >= 3:
        verdict = "GO"
    elif n110 == 0:
        verdict = "KILL"
    else:
        verdict = "in between"
    print(f"\ncells counted (baseline R >= 100): {len(v)}; max ratio {v.max():.3f}; "
          f">= 1.30: {n130}, >= 1.15: {n115}, >= 1.10: {n110}")
    print(f"M1: {verdict}")
    out = dirs[0].parent / f"m1_eval_{dirs[0].name.split('_D')[0]}.csv"  # one file per run family, never overwritten
    g.to_csv(out, float_format="%.3f")
    print("->", out)


if __name__ == "__main__":
    main()
