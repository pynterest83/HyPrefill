#!/usr/bin/env python3
"""KT2 (PROPOSAL §5, revised 2026-10-07 before running): steady-state prefill capacity per
iteration under a TBT budget B, for three policies on the same per-layer cost tables.

  uniform  : one chunk c per iteration through the full depth (tuned per t; at a single t in
             steady state SLOWeave's adaptive chunk equals it)
  layered  : depth-pipelined, one chunk n for every sublayer (Layered Prefill, k = 1)
  hyprefill: depth-pipelined, own chunk per group: n_A (attention), n_G (GDN), n_M (MoE)

Steady state, prefill throughput R tokens/iteration: every sublayer s processes R tokens per
iteration on average, firing on n_s tokens every n_s / R iterations (staggered). Constraints:
  GPU:  D(bd, t) + R * sum_s inc_s(n_s) / n_s <= B      (inc: cost added to the decode-only call)
  CPU:  h0 + host calls of the iteration <= B              (iteration time = max(GPU, CPU))
  R <= n_s (a sublayer runs once per iteration),  inc_s(n_s) of ONE layer <= B - D (one firing
  fits; finest layer groups, optimistic for layered and hyprefill alike),
  n_s <= delta (chunks of one append request; conservative for all policies alike).
Uniform is the special case n_s = R for all s, so layered >= uniform and hyprefill >= layered.

Costs per layer (per GPU, cold L2, 1980 MHz tables), decode and prefill share one call where
vLLM does (validated mixed model, plan/01 §2.5): dense GEMMs, MoE and all-reduce add
f(bd + n) - f(bd); FA and GDN prefill are separate kernels added on top of decode.
MoE with real routing: from KT1 (`--kt1 <op_cost dir>`: moe_mixed(bd, n) - moe_mixed(bd, 0)),
or before KT1 an estimate a0 + alpha * experts + eta * tokens fitted on the random-routing
table, with the experts touched taken from the real-routing overlap (bench/moe_overlap.py).
CPU: h0 per iteration + host_ms of the eager calls (GDN core, attention; vLLM runs both
outside CUDA graphs): decode GDN and attention every iteration, prefill GDN 36 R / n_G.
GPU kernel time is an optimistic bound for every policy (no launch gaps inside graphs).

  python bench/kt2_capacity.py                       # MoE estimate (before KT1)
  python bench/kt2_capacity.py --kt1 results/step02/<date>_s02_..._kt1
"""
import argparse, glob, itertools, json, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle import Costs, layers  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
GRID = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def load(model, kt1):
    name = model.split("_tp")[0]
    cfg = json.load(open(REPO / f"results/step00/configs/Qwen_{name}.json")); cfg = cfg.get("text_config", cfg)
    tab = pd.concat([pd.read_csv(REPO / f"results/step0{i}/cost_tables/{model}.csv") for i in (1, 2)],
                    ignore_index=True).drop_duplicates(["op", "c", "t"])
    import functools
    cost = functools.lru_cache(maxsize=None)(lambda op, n, t=None: float(Costs(tab)(op, n, t)))
    ar = pd.read_csv(glob.glob(str(REPO / f"results/step01/*_allreduce_{model.replace('_tp2', '')}_tp2/summary.csv"))[0])
    allreduce = lambda n: float(np.interp(n, ar.n, ar.p50_ms)) if n > 0 else 0.0
    E, k = cfg["num_experts"], cfg["num_experts_per_tok"]
    ov = pd.read_csv(REPO / f"results/step02/moe_overlap_{name}.csv")
    host = {op: float(tab[(tab.op == op) & (tab.t > 0)].host_ms.median()) for op in ("gdn", "fa")}

    if kt1:  # measured: moe_mixed with real routing, column t holds bd, averaged over routing draws
        m = pd.concat([pd.read_csv(f) for f in glob.glob(f"{kt1}/summary.csv") + glob.glob(f"{kt1}*/summary.csv")])
        m = m[m.op == "moe_mixed"].groupby(["t", "c"]).p50_ms.mean().reset_index()

        @functools.lru_cache(maxsize=None)
        def moe(bd, n):
            r = m[m.t == bd].sort_values("c")
            if r.empty:
                raise SystemExit(f"KT1 has no decode batch {bd}")
            return float(np.interp(n, r.c, r.p50_ms))
        src = f"KT1 {kt1}"
    else:  # estimate: fit a0 + alpha * E_random(c) + eta * c on the random-routing table
        r = tab[(tab.op == "moe")].sort_values("c")
        e_rand = E * (1 - (1 - k / E) ** r.c.to_numpy(float))
        X = np.stack([np.ones(len(r)), e_rand, r.c.to_numpy(float)], 1)
        a0, alpha, eta = np.linalg.lstsq(X, r.p50_ms.to_numpy(), rcond=None)[0]

        @functools.lru_cache(maxsize=None)
        def moe(bd, n):
            o = ov[ov.decode_batch == bd].sort_values("c")
            if o.empty:
                raise SystemExit(f"no real-routing overlap for decode batch {bd} (dump more requests)")
            experts = float(np.interp(n, o.c, o.union_experts))
            return a0 + alpha * experts + eta * (bd + n)
        src = f"estimate a0={a0 * 1e3:.1f} us, alpha={alpha * 1e3:.3f} us/expert, eta={eta * 1e3:.3f} us/token"
    return cfg, cost, allreduce, moe, host, src


def capacity(cfg, cost, allreduce, moe, host, B, bd, t, delta, h0, g=1, h_fire=0.0):
    """g: decoder layers per firing (1 = finest; a firing of a group type covers the layers of that
    type among g consecutive layers, at least one); h_fire: CPU ms per firing (scheduling and
    metadata of one chunk segment), 0 = as before."""
    n_attn, n_gdn, n_lay = layers(cfg)
    gl = {"A": max(1, round(g * n_attn / n_lay)), "G": max(1, round(g * n_gdn / n_lay)), "M": g}
    nl = {"A": n_attn, "G": n_gdn, "M": n_lay}
    c = lambda op, n, tt=None: float(cost(op, n, tt)) if n > 0 else 0.0
    # decode-only iteration (all layers)
    D = (n_attn * (c("fa_decode", bd, t) + c("dense_attn", bd)) + n_gdn * (c("gdn_decode", bd) + c("dense_gdn", bd))
         + n_lay * (moe(bd, 0) + 2 * allreduce(bd)))
    P = B - D
    # cost one firing of n prefill tokens adds, per group (all layers of that group)
    inc = {
        "A": lambda n: n_attn * (c("fa", n, t) + c("dense_attn", bd + n) - c("dense_attn", bd) + allreduce(bd + n) - allreduce(bd)),
        "G": lambda n: n_gdn * (c("gdn", n) + c("gdn_conv", n) + c("dense_gdn", bd + n) - c("dense_gdn", bd)
                                + allreduce(bd + n) - allreduce(bd)),
        "M": lambda n: n_lay * (moe(bd, n) - moe(bd, 0) + allreduce(bd + n) - allreduce(bd)),
    }
    memo = {}
    I = lambda g, n: memo.setdefault((g, n), inc[g](n))
    cpu_base = h0 + n_attn * host["fa"] + n_gdn * host["gdn"]  # decode calls every iteration

    def best(cfgs, uniform=False):
        top = (0.0, None, "")
        for nA, nG, nM in cfgs:
            # one firing = one sublayer of one layer when pipelined (finest layer groups, same for
            # layered and hyprefill); uniform runs the full depth in its iteration, checked by R = c
            ns = {"A": nA, "G": nG, "M": nM}
            if max(nA, nG, nM) > delta or max(I(x, ns[x]) / nl[x] * gl[x] for x in ns) > P:
                continue
            per_tok = I("A", nA) / nA + I("G", nG) / nG + I("M", nM) / nM
            # prefill GDN calls cost host time: n_gdn * R / nG per iteration
            # and every firing h_fire: (layers / g) firings of each group type per n_s / R iterations
            cpu_per_tok = (n_gdn * host["gdn"] / nG if n_gdn else 0.0) + h_fire * (1 / nA if uniform else sum(nl[x] / gl[x] / ns[x] for x in ns))
            r_cpu = (B - cpu_base) / cpu_per_tok if cpu_per_tok > 0 else np.inf
            R = min(P / per_tok, r_cpu, nA, nG, nM)
            lim = min((P / per_tok, "gpu"), (r_cpu, "cpu"), (min(nA, nG, nM), "chunk"))[1]
            if uniform:  # the chunk crosses the full depth in one iteration: R = c, or infeasible
                if R < nA:
                    continue
                lim = "budget"
            if R > top[0]:
                top = (R, (nA, nG, nM), lim)
        return top
    if P <= 0 or cpu_base >= B:
        return dict(P_ms=round(P, 2), cpu_base_ms=round(cpu_base, 2))
    # uniform: every sublayer fires every iteration on the same c (fine grid)
    uni = best([(n, n, n) for n in range(64, min(delta, 8192) + 1, 64)], uniform=True)
    lay = max(best([(n, n, n) for n in GRID]), uni, key=lambda x: x[0])
    hy = max(best(itertools.product(GRID, GRID, GRID)), lay, key=lambda x: x[0])
    return dict(P_ms=round(P, 2), cpu_base_ms=round(cpu_base, 2),
                uniform_R=round(uni[0]), uniform_c=uni[1][0] if uni[1] else 0, uniform_lim=uni[2],
                layered_R=round(lay[0]), layered_n=lay[1][0] if lay[1] else 0, layered_lim=lay[2],
                hy_R=round(hy[0]), hy_nA=hy[1][0] if hy[1] else 0, hy_nG=hy[1][1] if hy[1] else 0,
                hy_nM=hy[1][2] if hy[1] else 0, hy_lim=hy[2],
                lay_over_uni=lay[0] / uni[0] if uni[0] else np.nan,
                # uniform infeasible (R = 0) still counts: the best baseline is then layered
                hy_over_best=hy[0] / max(uni[0], lay[0]) if max(uni[0], lay[0]) else np.nan,
                pipe_over_uni=max(lay[0], hy[0]) / uni[0] if uni[0] else np.nan)


_LOADED = {}


def _cell(model, kt1, h0, B, bd, t, delta):
    if (model, kt1) not in _LOADED:
        _LOADED[(model, kt1)] = load(model, kt1)
    cfg, cost, allreduce, moe, host, _ = _LOADED[(model, kt1)]
    try:
        return capacity(cfg, cost, allreduce, moe, host, B, bd, t, delta, h0)
    except SystemExit as e:
        print(f"skip bd={bd}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--kt1", default=None, help="op_cost.py output dir(s) prefix of the KT1 moe_mixed run")
    ap.add_argument("--budgets", default="25,50,100")
    ap.add_argument("--decode", default="8,32,64")
    ap.add_argument("--t", default="65536,131072,262144")
    ap.add_argument("--delta", default="2048,4096,8192")
    ap.add_argument("--h0", default="10,0,20", help="CPU ms per iteration besides eager calls; first is the decision value")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    cfg, cost, allreduce, moe, host, src = load(a.model, a.kt1)
    print(f"MoE: {src}; host ms per call: {host}")
    grid = list(itertools.product(*[[float(x) for x in a.h0.split(",")]] + [
        [float(x) for x in a.budgets.split(",")], [int(x) for x in a.decode.split(",")],
        [int(x) for x in a.t.split(",")], [int(x) for x in a.delta.split(",")]]))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.starmap(_cell, [(a.model, a.kt1, *g) for g in grid])
    rows = [dict(h0_ms=g[0], B_ms=g[1], decode_batch=g[2], t=g[3], delta=g[4], **r) for g, r in zip(grid, res) if r is not None]
    df = pd.DataFrame(rows)
    out = pathlib.Path(a.out or REPO / f"results/step02/kt2_capacity_{a.model}{'_kt1' if a.kt1 else '_estimate'}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.3f")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["h0_ms", "B_ms", "decode_batch", "t", "delta", "P_ms", "uniform_R", "uniform_lim", "layered_R", "layered_n",
            "layered_lim", "hy_R", "hy_nA", "hy_nG", "hy_nM", "lay_over_uni", "hy_over_best", "pipe_over_uni"]
    print(df[[c for c in cols if c in df]].round(2).to_string(index=False))
    d = df[df.h0_ms == float(a.h0.split(",")[0])].dropna(subset=["hy_over_best"])
    for h0 in sorted(df.h0_ms.unique()):
        x = df[df.h0_ms == h0].dropna(subset=["hy_over_best"])
        print(f"h0={h0:g} ms: KT2-a max HyPrefill/max(uniform, layered) = {x.hy_over_best.max():.3f}; "
              f"KT2-b max pipelined/uniform (t >= 64K) = {x[x.t >= 65536].pipe_over_uni.max():.3f}; "
              f"layered/uniform range {x.lay_over_uni.min():.2f}-{x.lay_over_uni.max():.2f}")
    print(f"decision (h0={a.h0.split(',')[0]} ms): KT2-a "
          + ("GO" if d.hy_over_best.max() >= 1.25 else "KILL" if d.hy_over_best.max() < 1.10 else "in between")
          + ", KT2-b " + ("GO" if d[d.t >= 65536].pipe_over_uni.max() >= 1.20 else
                         "KILL" if d[d.t >= 65536].pipe_over_uni.max() < 1.10 else "in between")
          + ("" if a.kt1 else "  [MoE estimate: early warning only, rerun with --kt1]"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
