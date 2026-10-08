#!/usr/bin/env python3
"""Why HyPrefill barely beats Layered in KT2 (diagnosis, 2026-10-08, after KT1/KT2 ran).
Same steady-state capacity model as bench/kt2_capacity.py, MoE from KT1; no new criteria.

  phase  : HyPrefill / Layered over (t, P = B - D) at fixed decode batch, P swept finely (h0 = 0)
  group  : g decoder layers per firing (1 = finest, as in KT2), on the KT2 grid
  fire   : CPU ms per firing h_fire (scheduling one chunk segment), with g in {1, 4}
  pertok : per-token GPU cost per group (attention / GDN / MoE) against the chunk size

  python bench/kt2_diag.py --kt1 results/step02/<date>_s02_..._kt1_draw
Output: results/step02/kt2_diag_<model>_<date>/{phase,group,fire,pertok}.csv
"""
import argparse, datetime, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kt2_capacity import capacity, load  # noqa: E402
from oracle import layers  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--kt1", required=True)
    ap.add_argument("--h0", type=float, default=10.0)
    a = ap.parse_args()
    cfg, cost, allreduce, moe, host, src = load(a.model, a.kt1)
    out = REPO / f"results/step02/kt2_diag_{a.model}_{datetime.date.today()}"
    out.mkdir(parents=True, exist_ok=False)
    cap = lambda **k: capacity(cfg, cost, allreduce, moe, host, **k)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

    # per-token GPU cost of each group (whole model) against the chunk size
    n_attn, n_gdn, n_lay = layers(cfg)
    c = lambda op, n, t=None: float(cost(op, n, t))
    rows = []
    for bd in (8, 32, 64):
        for t in (65536, 131072, 262144):
            for n in (256, 512, 1024, 2048, 4096, 8192):
                ar = allreduce(bd + n) - allreduce(bd)
                A = n_attn * (c("fa", n, t) + c("dense_attn", bd + n) - c("dense_attn", bd) + ar)
                G = n_gdn * (c("gdn", n) + c("gdn_conv", n) + c("dense_gdn", bd + n) - c("dense_gdn", bd) + ar)
                M = n_lay * (moe(bd, n) - moe(bd, 0) + ar)
                rows.append(dict(decode_batch=bd, t=t, n=n, A_us_tok=1e3 * A / n, G_us_tok=1e3 * G / n,
                                 M_us_tok=1e3 * M / n, total_us_tok=1e3 * (A + G + M) / n, A_share=A / (A + G + M),
                                 one_attn_layer_ms=A / n_attn, one_gdn_layer_ms=G / n_gdn, one_moe_layer_ms=M / n_lay))
    pertok = pd.DataFrame(rows)
    pertok.to_csv(out / "pertok.csv", index=False, float_format="%.4f")

    # phase map: P swept by moving B at fixed decode batch
    rows = []
    for bd in (8, 32, 64):
        for t in (65536, 131072, 262144):
            D = cap(B=1e6, bd=bd, t=t, delta=8192, h0=0.0)["P_ms"]  # P at huge B gives D
            D = 1e6 - D
            for P in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
                for delta in (2048, 8192):
                    r = cap(B=D + P, bd=bd, t=t, delta=delta, h0=0.0)  # GPU geometry only; CPU in group_fire
                    if r.get("layered_R"):
                        rows.append(dict(decode_batch=bd, t=t, delta=delta, D_ms=round(D, 2), P_ms=P, B_ms=round(D + P, 2),
                                         layered_R=r["layered_R"], layered_n=r["layered_n"], hy_R=r["hy_R"],
                                         hy_nA=r["hy_nA"], hy_nG=r["hy_nG"], hy_nM=r["hy_nM"],
                                         hy_over_lay=r["hy_R"] / r["layered_R"]))
    phase = pd.DataFrame(rows)
    phase.to_csv(out / "phase.csv", index=False, float_format="%.3f")

    # layer-group size and per-firing CPU cost, on the KT2 grid
    grid = [(B, bd, t, dl) for B in (25, 50, 100) for bd in (8, 32, 64)
            for t in (65536, 131072, 262144) for dl in (2048, 4096, 8192)]
    rows = []
    for g in (1, 2, 4, 8, 12, 24, 48):
        for hf in ((0.0,) if g not in (1, 4) else (0.0, 0.02, 0.05, 0.1, 0.2)):
            for B, bd, t, dl in grid:
                r = cap(B=B, bd=bd, t=t, delta=dl, h0=a.h0, g=g, h_fire=hf)
                if r.get("hy_over_best") is not None:
                    rows.append(dict(g=g, h_fire_ms=hf, B_ms=B, decode_batch=bd, t=t, delta=dl,
                                     **{k: r[k] for k in ("uniform_R", "layered_R", "layered_n", "hy_R", "hy_nA",
                                                          "hy_nG", "hy_nM", "hy_over_best", "pipe_over_uni")}))
    gf = pd.DataFrame(rows)
    gf.to_csv(out / "group_fire.csv", index=False, float_format="%.3f")

    print("MoE:", src)
    print("\n== per-token GPU cost (us/token, whole model), decode batch 32")
    print(pertok[pertok.decode_batch == 32].round(3).to_string(index=False))
    print("\n== HyPrefill / Layered against P (rows: decode batch, t; columns: P ms), delta = 8192")
    print(phase[phase.delta == 8192].pivot_table(index=["decode_batch", "t"], columns="P_ms", values="hy_over_lay").round(3).to_string())
    print("\n== layered chunk n chosen, delta = 8192")
    print(phase[phase.delta == 8192].pivot_table(index=["decode_batch", "t"], columns="P_ms", values="layered_n").to_string())
    s = gf.groupby(["g", "h_fire_ms"]).agg(cells=("hy_over_best", "count"), max_hy_over_best=("hy_over_best", "max"),
                                          median=("hy_over_best", "median"),
                                          cells_ge_110=("hy_over_best", lambda x: int((x >= 1.10).sum())),
                                          cells_ge_125=("hy_over_best", lambda x: int((x >= 1.25).sum())),
                                          max_pipe_over_uni=("pipe_over_uni", "max"))
    print("\n== layer group g and per-firing CPU h_fire (KT2 grid, h0 = %g ms)" % a.h0)
    print(s.round(3).to_string())
    print("->", out)


if __name__ == "__main__":
    main()
