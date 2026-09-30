#!/usr/bin/env python3
"""Step 2 Việc 2-3: sim/hyprefill_sim.js ported to Python, with the measured cost tables
instead of its illustrative constants.

Separates the two mechanisms (PROPOSAL §5, "Vì sao tách G1"):
  sarathi  : one chunk c_u, every chunk crosses the full depth in one iteration
  layered  : depth-pipelined, the same chunk for every sublayer (k = 1)
  hyprefill: depth-pipelined, attention chunk c, every other sublayer k*c (k >= 2)
One request of delta new tokens at context t (append-prefill), decode of Bd sequences at
context t running every iteration, per-iteration prefill budget P = B - D. Result: iterations
until all delta tokens exit the stack. The ratio of iterations is the ratio of sustainable
prefill throughput under the TBT budget, i.e. a capacity proxy for the goodput ratio that
gates G1a/G1b are defined on (PROPOSAL §4.4); the request-level goodput needs the step 3
simulator (arrivals, queueing, SLO attainment).

Same greedy firing rule as the JS: each iteration spends P firing sublayers deep-first; a
sublayer fires on `rate` tokens, or flushes its remainder once everything upstream is empty.
Costs per sublayer (per GPU, ms): attention fa(n, t) + dense_attn(n); GDN gdn(n) + dense_gdn(n)
+ gdn_conv(n); FFN moe(n) or dense_mlp(n). GPU kernel time only: the CPU-bound regime at
small chunks (plan/01 §2.5) and the TP all-reduce are not modelled, and they penalise small
chunks, i.e. sarathi and layered more than hyprefill. Indexer memory caps are not modelled
either: G1c showed the QSA logits buffer is bounded at ~1 GiB, which does not bind on H200.

  python bench/sim_tokenflow.py --model Qwen3-Next-80B-A3B-Instruct_tp2
"""
import argparse, functools, json, math, pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle import Costs, layers  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
C_GRID = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
K_GRID = [2, 4, 8, 16]


def build(model, cfg):
    tables = [REPO / f"results/step01/cost_tables/{model}.csv", REPO / f"results/step02/cost_tables/{model}.csv"]
    tab = pd.concat([pd.read_csv(p) for p in tables if p.exists()], ignore_index=True).drop_duplicates(["op", "c", "t"])
    cost = Costs(tab)
    ffn = "moe" if "moe" in set(tab.op) else "dense_mlp"
    every = cfg.get("full_attention_interval", 1)  # 1: every layer is full attention (e.g. Qwen3-30B-A3B)
    lt = cfg.get("layer_types") or ["full_attention" if (i + 1) % every == 0 else "linear_attention"
                                    for i in range(cfg["num_hidden_layers"])]
    layout = []
    for x in lt:  # mixer then FFN, per decoder layer
        layout += ["A" if x == "full_attention" else "G", "M"]
    n_attn, n_gdn, n = layers(cfg)

    @functools.lru_cache(maxsize=None)
    def sub(ty, n_tok, t):
        if ty == "A":
            return float(cost("fa", n_tok, t) + cost("dense_attn", n_tok))
        if ty == "G":
            return float(cost("gdn", n_tok) + cost("dense_gdn", n_tok) + cost("gdn_conv", n_tok))
        return float(cost(ffn, n_tok))

    def D(bd, t):
        gdn = n_gdn * (cost("gdn_decode", bd) + cost("dense_gdn", bd)) if n_gdn else 0.0
        return float(n_attn * (cost("fa_decode", bd, t) + cost("dense_attn", bd)) + gdn + n * cost(ffn, bd))
    return layout, sub, D


def simulate(layout, sub, P, delta, t, rate_of, max_it=20000):
    S = len(layout)
    buf = [0] * (S + 1)
    buf[0] = delta
    it = 0
    while buf[S] < delta and it < max_it:
        it += 1
        budget = P
        progressed = True
        while progressed:
            progressed = False
            first_nonempty = next((u for u in range(S) if buf[u] > 0), S)
            for s in range(S - 1, -1, -1):
                r = rate_of(layout[s])
                up_empty = first_nonempty >= s
                n = r if buf[s] >= r else (buf[s] if buf[s] > 0 and up_empty else 0)
                if n <= 0:
                    continue
                c = sub(layout[s], n, t)
                if c <= budget:
                    budget -= c
                    buf[s] -= n
                    buf[s + 1] += n
                    progressed = True
                    if s == first_nonempty and buf[s] == 0:
                        first_nonempty = next((u for u in range(s, S) if buf[u] > 0), S)
    return it if buf[S] >= delta else math.inf


def run(model, B, bd, t, delta):
    name = model.split("_tp")[0]
    cfg = json.load(open(REPO / f"results/step00/configs/Qwen_{name}.json")); cfg = cfg.get("text_config", cfg)
    layout, sub, D = build(model, cfg)
    P = B - D(bd, t)
    if P <= 0:
        return dict(P_ms=P)
    # sarathi: largest uniform chunk whose full-depth pass fits in P
    c_u = max((c for c in range(64, 32769, 64) if sum(sub(ty, c, t) for ty in layout) <= P), default=0)
    sar = math.ceil(delta / c_u) if c_u else math.inf
    lay = min(((simulate(layout, sub, P, delta, t, lambda ty, c=c: c), c) for c in C_GRID if c <= max(delta, 64)),
              default=(math.inf, 0))
    hy = min(((simulate(layout, sub, P, delta, t, lambda ty, c=c, k=k: c if ty == "A" else k * c), c, k)
              for c in C_GRID if c <= max(delta, 64) for k in K_GRID), default=(math.inf, 0, 0))
    hy = min(hy, (lay[0], lay[1], 1))  # k = 1 is layered; hyprefill never does worse than it
    return dict(P_ms=round(P, 2), sarathi_iters=sar, sarathi_c=c_u, layered_iters=lay[0], layered_c=lay[1],
                hyprefill_iters=hy[0], hyprefill_c=hy[1], hyprefill_k=hy[2],
                lay_over_sar=sar / lay[0], hy_over_lay=lay[0] / hy[0], hy_over_sar=sar / hy[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--budgets", default="50,100")
    ap.add_argument("--decode", default="8,32")
    ap.add_argument("--t", default="16384,65536,131072,262144")
    ap.add_argument("--delta", default="1024,4096,16384")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    grid = [(float(B), int(bd), int(t), int(delta)) for B in a.budgets.split(",") for bd in a.decode.split(",")
            for t in a.t.split(",") for delta in a.delta.split(",")]
    import multiprocessing as mp
    with mp.Pool(a.jobs) as pool:
        res = pool.starmap(run, [(a.model, *g) for g in grid])
    rows = [dict(B_ms=g[0], decode_batch=g[1], t=g[2], delta=g[3], **r) for g, r in zip(grid, res)]
    df = pd.DataFrame(rows)
    out = REPO / f"results/step02/sim_tokenflow_{a.model}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
