#!/usr/bin/env python3
"""Step 2 oracle (PROPOSAL §2.2, plan/02 §2.4): largest prefill chunk per iteration under a
TBT budget B, uniform chunk vs per-operator-group chunks. Pure arithmetic on the measured
cost tables (no GPU, no depth pipelining: that is the token-flow simulator's job).

  P = B - D                  D = decode step of Bd sequences at context t (all layers)
  uniform:   r_u(t) = max c  s.t.  sum_g cost_g(c, t) <= P
  decoupled: r_d(t) = max c  s.t.  cost_A(c, t) + sum_{g != A} cost_g(k c) / k <= P   (amortized)
                                and cost_A(c, t) + max_{g != A} cost_g(k c)     <= P   (peak, staggered)
  Gain(t) = r_d(t) / r_u(t),  k in {1, 2, 4, 8, 16}

Groups, per layer type, all layers of that type together (per GPU, TP as served):
  A   (attention): fa(c, t) + dense_attn(c)            x n_attn layers
  GDN            : gdn(c) + dense_gdn(c) + gdn_conv(c)  x n_gdn layers
  FFN            : moe(c) (MoE models) or dense_mlp(c)  x n_layers
Costs are GPU kernel time (cold L2); the CPU-bound regime at small chunks (plan/01 §2.5)
and the TP all-reduce are not in the tables yet, so this is an optimistic bound for both
policies. Decode is added as a separate D, not merged into the prefill kernels (plan/02
Việc 4 checks how much MoE expert reads decode and prefill actually share).

  python bench/oracle.py --model Qwen3-Next-80B-A3B-Instruct_tp2
"""
import argparse, json, pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
C_GRID = np.arange(64, 32768 + 1, 64)  # GDN chunks are multiples of 64


class Costs:
    def __init__(self, table):
        self.t = table

    def _curve(self, op, t=None):
        r = self.t[self.t.op == op]
        if t is not None and r.t.nunique() > 1:
            # linear in t between measured contexts, per measured c
            cs = sorted(r.c.unique())
            vals = []
            for c in cs:
                rc = r[r.c == c].sort_values("t")
                vals.append(np.interp(t, rc.t, rc.p50_ms))
            return np.array(cs, float), np.array(vals)
        r = r[r.t == r.t.min()].sort_values("c")
        return r.c.to_numpy(float), r.p50_ms.to_numpy()

    def __call__(self, op, c, t=None):
        """ms per layer; c may be an array; linear in c inside the grid, linear extrapolation
        from the last two points above it (kernels are linear in c at large c)."""
        xs, ys = self._curve(op, t)
        m = xs % 64 == 0 if (xs % 64 == 0).sum() >= 2 else np.ones_like(xs, bool)
        xs, ys = xs[m], ys[m]
        c = np.asarray(c, float)
        out = np.interp(c, xs, ys)
        hi = c > xs[-1]
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return np.where(hi, ys[-1] + slope * (c - xs[-1]), out)


def layers(cfg):
    lt = cfg.get("layer_types")
    n = cfg["num_hidden_layers"]
    n_attn = sum(x == "full_attention" for x in lt) if lt else n // cfg.get("full_attention_interval", 1)
    return n_attn, n - n_attn, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tables", nargs="+", default=None, help="cost tables (default: step 1 + step 2 for --model)")
    ap.add_argument("--budgets", default="25,50,100", help="TBT SLO B in ms")
    ap.add_argument("--decode", default="8,32,64", help="decode batch Bd")
    ap.add_argument("--t", default="4096,16384,32768,65536,131072,262144")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--batch-model", choices=["additive", "mixed"], default="mixed",
                    help="additive: D + prefill costs (PROPOSAL §2.2 as first written); mixed: GEMM/MoE layers "
                         "run decode and prefill tokens in one call (cost at bd + tokens), FA/GDN kernels separate")
    a = ap.parse_args()

    name = a.model.split("_tp")[0]
    cfg = json.load(open(a.config or REPO / f"results/step00/configs/Qwen_{name}.json"))
    cfg = cfg.get("text_config", cfg)
    tables = a.tables or sorted(str(p) for p in (REPO / "results/step01/cost_tables").glob(f"{a.model}.csv")) + \
        sorted(str(p) for p in (REPO / "results/step02/cost_tables").glob(f"{a.model}*.csv"))
    tab = pd.concat([pd.read_csv(p) for p in tables], ignore_index=True)
    tab = tab.drop_duplicates(["op", "c", "t"], keep="first")
    cost = Costs(tab)
    ops = set(tab.op)
    n_attn, n_gdn, n = layers(cfg)
    ffn = "moe" if "moe" in ops else "dense_mlp"

    A = lambda c, t: n_attn * (cost("fa", c, t) + cost("dense_attn", c))
    G = lambda c: n_gdn * (cost("gdn", c) + cost("dense_gdn", c) + cost("gdn_conv", c))
    F = lambda c: n * cost(ffn, c)
    groups = [G, F] if n_gdn else [F]

    def D(bd, t):
        gdn = n_gdn * (cost("gdn_decode", bd) + cost("dense_gdn", bd)) if n_gdn else 0.0
        return n_attn * (cost("fa_decode", bd, t) + cost("dense_attn", bd)) + gdn + n * cost(ffn, bd)

    # mixed batch: prefill-exclusive kernels (FA, GDN, conv) vs layers whose GEMMs take decode
    # and prefill tokens in one call (attention/GDN projections, FFN/MoE)
    D_ex = lambda bd, t: n_attn * cost("fa_decode", bd, t) + (n_gdn * cost("gdn_decode", bd) if n_gdn else 0.0)
    A_ex = lambda c, t: n_attn * cost("fa", c, t)
    A_sh = lambda m: n_attn * cost("dense_attn", m)
    G_ex = lambda c: n_gdn * (cost("gdn", c) + cost("gdn_conv", c))
    G_sh = lambda m: n_gdn * cost("dense_gdn", m)
    F_sh = lambda m: n * cost(ffn, m)
    nonfa = [(G_ex, G_sh), (lambda c: 0.0 * c, F_sh)] if n_gdn else [(lambda c: 0.0 * c, F_sh)]

    def iter_cost(bd, t, c, k):
        """(amortized, peak) iteration cost, attention chunk c, non-attention groups k*c every k-th
        iteration (k = 1: uniform chunk)."""
        c = np.asarray(c, float)
        if a.batch_model == "additive":
            base = float(D(bd, t)) + A(c, t)
            return base + sum(g(k * c) / k for g in groups), base + np.max([g(k * c) for g in groups], axis=0)
        base = D_ex(bd, t) + A_ex(c, t) + A_sh(bd + c)
        idle = sum(sh(bd) for _, sh in nonfa)              # a group's shared layers when it has no prefill work
        big = [ex(k * c) + sh(bd + k * c) for ex, sh in nonfa]
        amort = base + sum((b + (k - 1) * sh(bd)) / k for b, (_, sh) in zip(big, nonfa))
        peak = base + np.max([b + idle - sh(bd) for b, (_, sh) in zip(big, nonfa)], axis=0)  # staggered: one big group
        return amort, peak

    rows = []
    for B in [float(x) for x in a.budgets.split(",")]:
        for bd in [int(x) for x in a.decode.split(",")]:
            for t in [int(x) for x in a.t.split(",")]:
                P = B - float(D(bd, t))
                c = C_GRID
                am, pk = iter_cost(bd, t, c, 1)
                ok = am <= B
                r_u = int(c[ok].max()) if ok.any() else 0
                best = (r_u, 1)
                for k in (2, 4, 8, 16):
                    am, pk = iter_cost(bd, t, c, k)
                    ok = (am <= B) & (pk <= B)
                    if ok.any() and int(c[ok].max()) > best[0]:
                        best = (int(c[ok].max()), k)
                r_d, k = best
                rows.append(dict(B_ms=B, decode_batch=bd, t=t, D_ms=round(float(D(bd, t)), 3), P_ms=round(P, 3),
                                 r_u=r_u, r_d=r_d, k=k, gain=(r_d / r_u) if r_u else np.nan))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"model {a.model}: {n_attn} attention, {n_gdn} GDN, {n} {ffn} layers; batch model {a.batch_model}; tables {tables}")
    print(df.to_string(index=False))
    out = a.out or REPO / f"results/step02/oracle_{a.model}_{a.batch_model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
