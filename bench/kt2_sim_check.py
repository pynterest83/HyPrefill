#!/usr/bin/env python3
"""Independent check of bench/kt2_capacity.py (2026-10-09): simulate the depth-pipelined schedule
iteration by iteration instead of using the steady-state formula R = min(P / per_token, n_s, ...).

Same per-layer cost inputs (cost tables, all-reduce, KT1 MoE), different logic:
  sublayers in model order (Qwen3-Next: 48 x [mixer (GDN or attention), MoE]); a queue of tokens in
  front of each; an unlimited supply of prefill tokens in front of the first. Every iteration has a
  GPU budget P = B - D; sublayers are visited from the deepest to the first (drain the pipeline
  first) and fire on n_s tokens if enough are queued and the firing still fits the budget; a sublayer
  fires at most once per iteration. Throughput = tokens leaving the last sublayer per iteration, after
  a warmup. Attention is charged at context t for every firing, as kt2_capacity.py does.
GPU only (no CPU term): compare with kt2_capacity at h0 = 0.

  python bench/kt2_sim_check.py --kt1 results/step02/<date>_s02_..._kt1_draw
"""
import argparse, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kt2_capacity import capacity, load  # noqa: E402  (load: same inputs; capacity: the formula under test)


def per_layer_inc(cfg, cost, allreduce, moe, bd, t):
    c = lambda op, n, tt=None: float(cost(op, n, tt)) if n > 0 else 0.0
    ar = lambda n: allreduce(bd + n) - allreduce(bd)
    return {
        "A": lambda n: c("fa", n, t) + c("dense_attn", bd + n) - c("dense_attn", bd) + ar(n),
        "G": lambda n: c("gdn", n) + c("gdn_conv", n) + c("dense_gdn", bd + n) - c("dense_gdn", bd) + ar(n),
        "M": lambda n: moe(bd, n) - moe(bd, 0) + ar(n),
    }


def simulate(kinds, inc, n, P, iters=4000, warm=1000):
    S = len(kinds)
    cost = [inc[k](n[k]) for k in kinds]
    if max(cost) > P:
        return 0.0
    q = [0] * (S + 1)
    q[0] = 10**12
    done = 0
    for it in range(iters):
        left = P
        for s in range(S - 1, -1, -1):
            if q[s] >= n[kinds[s]] and cost[s] <= left:
                left -= cost[s]; q[s] -= n[kinds[s]]; q[s + 1] += n[kinds[s]]
        if it >= warm:
            done += q[S]
        q[S] = 0
    return done / (iters - warm)


GRID = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def best_cell(args):
    """Best throughput in the simulation for layered (one n) and hyprefill (n_A, n_G = n_M), each
    searched on GRID (n <= delta), for one KT2 cell."""
    kinds, B, bd, t, delta = args
    r = capacity(CTX["cfg"], CTX["cost"], CTX["allreduce"], CTX["moe"], CTX["host"], B, bd, t, delta, 0.0)
    if not r.get("layered_R"):
        return None
    inc = per_layer_inc(CTX["cfg"], CTX["cost"], CTX["allreduce"], CTX["moe"], bd, t)
    P, g = r["P_ms"], [n for n in GRID if n <= delta]
    lay = max((simulate(kinds, inc, {k: n for k in "AGM"}, P, 2500, 500), n) for n in g)
    hy = max((simulate(kinds, inc, {"A": na, "G": nb, "M": nb}, P, 2500, 500), na, nb) for na in g for nb in g)
    return dict(B=B, bd=bd, t=t, delta=delta, P=P, R_lay_formula=r["layered_R"], n_lay_formula=r["layered_n"],
                R_hy_formula=r["hy_R"], R_lay_sim=round(lay[0]), n_lay_sim=lay[1], R_hy_sim=round(hy[0]),
                n_hy_sim=f"{hy[1]}/{hy[2]}", ratio_formula=r["hy_R"] / r["layered_R"],
                ratio_sim=hy[0] / lay[0] if lay[0] else np.nan)


CTX = {}


def per_layer_inc2(cfg, cost, allreduce, moe, bd, t, delta):
    """As per_layer_inc, but an attention firing of n tokens is charged the average FA cost of the
    delta / n chunks of one append (contexts t, t + n, ...), not FA(n, t): the causal work of an
    append does not depend on how it is chunked, and charging every chunk at t favored small n_A."""
    c = lambda op, n, tt=None: float(cost(op, n, tt)) if n > 0 else 0.0
    ar = lambda n: allreduce(bd + n) - allreduce(bd)
    fa = lambda n: float(np.mean([c("fa", n, t + j * n) for j in range(max(1, delta // n))]))
    return {
        "A": lambda n: fa(n) + c("dense_attn", bd + n) - c("dense_attn", bd) + ar(n),
        "G": lambda n: c("gdn", n) + c("gdn_conv", n) + c("dense_gdn", bd + n) - c("dense_gdn", bd) + ar(n),
        "M": lambda n: moe(bd, n) - moe(bd, 0) + ar(n),
    }


def simulate2(kinds, inc, host, n, P, cpu, order, iters=2500, warm=500):
    """simulate() with a CPU budget too (cpu ms per iteration left for prefill calls; GDN and attention
    run eagerly, MoE inside CUDA graphs) and a visiting order: deepest-first or shallowest-first."""
    S = len(kinds)
    cost = [inc[k](n[k]) for k in kinds]
    hcost = [host.get(k, 0.0) for k in kinds]
    if max(cost) > P or cpu <= 0:
        return 0.0
    rng = range(S - 1, -1, -1) if order == "deep" else range(S)
    q = [0] * (S + 1); q[0] = 10**12
    done = 0
    for it in range(iters):
        left, cleft = P, cpu
        for s in rng:
            if q[s] >= n[kinds[s]] and cost[s] <= left and hcost[s] <= cleft:
                left -= cost[s]; cleft -= hcost[s]; q[s] -= n[kinds[s]]; q[s + 1] += n[kinds[s]]
        if it >= warm:
            done += q[S]
        q[S] = 0
    return done / (iters - warm)


def best_cell2(args):
    kinds, B, bd, t, delta, h0 = args
    cfg = CTX["cfg"]
    r = capacity(cfg, CTX["cost"], CTX["allreduce"], CTX["moe"], CTX["host"], B, bd, t, delta, h0)
    D = B - r["P_ms"]
    P = r["P_ms"]
    n_attn = kinds.count("A"); n_gdn = kinds.count("G")
    hostd = {"A": CTX["host"]["fa"], "G": CTX["host"]["gdn"]}
    cpu = B - h0 - n_attn * hostd["A"] - n_gdn * hostd["G"]  # decode calls every iteration
    base = dict(B=B, bd=bd, t=t, delta=delta, h0=h0, P=round(P, 2), cpu_left=round(cpu, 2))
    if P <= 0 or cpu <= 0:
        return dict(base, R_lay=0, R_hy=0, ratio=np.nan)
    inc = per_layer_inc2(cfg, CTX["cost"], CTX["allreduce"], CTX["moe"], bd, t, delta)
    g = [n for n in GRID if n <= delta]
    lay = max((simulate2(kinds, inc, hostd, {k: n for k in "AGM"}, P, cpu, o), n, o) for n in g for o in ("deep", "shallow"))
    hy = max((simulate2(kinds, inc, hostd, {"A": na, "G": nb, "M": nb}, P, cpu, o), na, nb, o)
             for na in g for nb in g for o in ("deep", "shallow"))
    return dict(base, R_lay=round(lay[0]), n_lay=lay[1], order_lay=lay[2], R_hy=round(hy[0]), n_hy=f"{hy[1]}/{hy[2]}",
                order_hy=hy[3], ratio=hy[0] / lay[0] if lay[0] else np.nan)


def main_best2(a):
    import multiprocessing as mp
    cfg, cost, allreduce, moe, host, _ = load(a.model, a.kt1)
    if a.host_from:
        h = pd.read_csv(a.host_from)
        host = {op: float(h[(h.op == op) & (h.t > 0)].host_ms.median()) for op in ("gdn", "fa")}
    CTX.update(cfg=cfg, cost=cost, allreduce=allreduce, moe=moe, host=host)
    n_lay, fai = cfg["num_hidden_layers"], cfg.get("full_attention_interval", 1)
    lt = cfg.get("layer_types") or ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(n_lay)]
    kinds = [x for l in lt for x in (("A" if l == "full_attention" else "G"), "M")]
    cells = [(kinds, B, bd, t, dl, h0) for h0 in (0.0, 10.0) for B in (25, 50, 100) for bd in (8, 32, 64)
             for t in (65536, 131072, 262144) for dl in (2048, 8192)]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        d = pd.DataFrame(pool.map(best_cell2, cells))
    pd.set_option("display.width", 230); pd.set_option("display.max_rows", 300)
    print("host ms per call:", host)
    for h0 in (0.0, 10.0):
        x = d[(d.h0 == h0) & (d.R_lay > 0)]
        print(f"\n== h0 = {h0:g} ms (CPU budget per iteration = B - h0 - decode calls)")
        print(x.sort_values("ratio", ascending=False).head(20).round(3).to_string(index=False))
        print(f"max HyPrefill/Layered {x.ratio.max():.3f}; cells >= 1.10: {(x.ratio >= 1.10).sum()}, "
              f">= 1.25: {(x.ratio >= 1.25).sum()} of {len(x)}")
    if a.out:
        d.to_csv(a.out, index=False, float_format="%.3f"); print("->", a.out)


def main_best(a):
    import multiprocessing as mp
    cfg, cost, allreduce, moe, host, _ = load(a.model, a.kt1)
    CTX.update(cfg=cfg, cost=cost, allreduce=allreduce, moe=moe, host=host)
    n_lay, fai = cfg["num_hidden_layers"], cfg.get("full_attention_interval", 1)
    lt = cfg.get("layer_types") or ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(n_lay)]
    kinds = [x for l in lt for x in (("A" if l == "full_attention" else "G"), "M")]
    cells = [(kinds, B, bd, t, dl) for B in (25, 50, 100) for bd in (8, 32, 64) for t in (65536, 131072, 262144)
             for dl in (2048, 8192)]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        rows = [x for x in pool.map(best_cell, cells) if x]
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(d.round(3).to_string(index=False))
    print(f"\nmax HyPrefill/Layered: formula {d.ratio_formula.max():.3f}, simulation (both tuned) {d.ratio_sim.max():.3f}; "
          f"cells >= 1.10: formula {(d.ratio_formula >= 1.10).sum()}, sim {(d.ratio_sim >= 1.10).sum()}; "
          f">= 1.25: formula {(d.ratio_formula >= 1.25).sum()}, sim {(d.ratio_sim >= 1.25).sum()} of {len(d)}")
    if a.out:
        d.to_csv(a.out, index=False, float_format="%.3f"); print("->", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--kt1", required=True)
    ap.add_argument("--best", action="store_true", help="tune n per policy inside the simulation (KT2 grid)")
    ap.add_argument("--best2", action="store_true", help="--best with FA averaged over the append, a CPU budget and two orders")
    ap.add_argument("--host-from", default=None, help="op_cost.py summary.csv with GDN / FA host_ms")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.best2:
        return main_best2(a)
    if a.best:
        return main_best(a)
    cfg, cost, allreduce, moe, host, _ = load(a.model, a.kt1)
    n_lay, fai = cfg["num_hidden_layers"], cfg.get("full_attention_interval", 1)
    lt = cfg.get("layer_types") or ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(n_lay)]
    kinds = [x for l in lt for x in (("A" if l == "full_attention" else "G"), "M")]
    rows = []
    for B in (50, 100):
        for bd in (8, 32, 64):
            for t in (65536, 131072):
                for delta in (2048, 8192):
                    r = capacity(cfg, cost, allreduce, moe, host, B, bd, t, delta, 0.0)
                    if not r.get("layered_R"):
                        continue
                    inc = per_layer_inc(cfg, cost, allreduce, moe, bd, t)
                    P = r["P_ms"]
                    lay = simulate(kinds, inc, {k: r["layered_n"] for k in "AGM"}, P)
                    hy = simulate(kinds, inc, {"A": r["hy_nA"], "G": r["hy_nG"], "M": r["hy_nM"]}, P)
                    rows.append(dict(B=B, bd=bd, t=t, delta=delta, P=P, layered_n=r["layered_n"],
                                     hy_n=f'{r["hy_nA"]}/{r["hy_nG"]}/{r["hy_nM"]}',
                                     R_lay_formula=r["layered_R"], R_lay_sim=round(lay), R_hy_formula=r["hy_R"],
                                     R_hy_sim=round(hy), ratio_formula=r["hy_R"] / r["layered_R"],
                                     ratio_sim=hy / lay if lay else np.nan))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(d.round(3).to_string(index=False))
    print(f"\nsim / formula: layered {np.median(d.R_lay_sim / d.R_lay_formula):.3f} (median), "
          f"hyprefill {np.median(d.R_hy_sim / d.R_hy_formula):.3f}; max ratio formula {d.ratio_formula.max():.3f}, "
          f"sim {d.ratio_sim.max():.3f}")


if __name__ == "__main__":
    main()
