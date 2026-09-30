#!/usr/bin/env python3
"""Step 2 (plan/02, Việc 4), offline: how many experts a prefill chunk adds on top of what the
decode batch of the same iteration already touches, on real routing (bench/moe_routing_dump.py).

In vLLM decode and prefill tokens share one fused MoE call, so expert weights read for decode
are read once for both. The "expert reload tax" a prefill chunk pays per iteration is only the
experts it touches that decode does not: |D ∪ P| - |D| per layer.

Sampling per (decode batch bd, prefill chunk c), `--samples` times:
  decode set D: one generated token from each of bd distinct requests (random position)
  prefill set P: c consecutive prompt tokens of one other request (random offset)
Per layer: |D|, |P|, |D ∪ P|; reported as mean over layers and samples, next to the
random-routing value E(1 - (1 - k/E)^n).

  python bench/moe_overlap.py ~/hyprefill_data/step02/moe_routing/<dir> --out results/step02/moe_overlap_<model>.csv
"""
import argparse, glob, json, pathlib

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--decode", default="8,32,64,128")
ap.add_argument("--c", default="0,64,128,256,512,1024,2048,4096,8192")
ap.add_argument("--samples", type=int, default=200)
ap.add_argument("--experts", type=int, default=None, help="number of routed experts (default: from max id + 1)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=pathlib.Path, required=True)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
reqs = []
for f in sorted(glob.glob(f"{a.dump}/req_*.npz")):
    z = np.load(f)
    reqs.append((z["experts"].astype(np.int32), int(z["prompt_len"])))
meta = json.load(open(f"{a.dump}/meta.json"))
E = a.experts or int(max(r[0].max() for r in reqs)) + 1
k = reqs[0][0].shape[2]
L = reqs[0][0].shape[1]
print(f"{len(reqs)} requests, {L} layers, E={E}, top-{k}")


def touched(tokens):  # tokens: [n, L, k] -> distinct experts per layer, [L]
    if len(tokens) == 0:
        return np.zeros(L, int), [set() for _ in range(L)]
    sets = [set(np.unique(tokens[:, l, :]).tolist()) for l in range(L)]
    return np.array([len(s) for s in sets]), sets


rows = []
for bd in [int(x) for x in a.decode.split(",")]:
    if bd > len(reqs) - 1:
        print(f"skip bd={bd}: only {len(reqs)} requests"); continue
    for c in [int(x) for x in a.c.split(",")]:
        d_n, p_n, u_n = [], [], []
        for _ in range(a.samples):
            idx = rng.permutation(len(reqs))
            pre, dec = idx[0], idx[1:bd + 1]
            D = np.stack([reqs[i][0][rng.integers(reqs[i][1], reqs[i][0].shape[0])] for i in dec])
            ex, plen = reqs[pre]
            if c > plen:
                continue
            s = rng.integers(0, plen - c + 1)
            P = ex[s:s + c]
            dn, ds = touched(D)
            pn, ps = touched(P)
            un = np.array([len(ds[l] | ps[l]) for l in range(L)])
            d_n.append(dn.mean()); p_n.append(pn.mean()); u_n.append(un.mean())
        if not d_n:
            continue
        rnd = lambda n: E * (1 - (1 - k / E) ** n)
        rows.append(dict(decode_batch=bd, c=c, samples=len(d_n),
                         decode_experts=np.mean(d_n), prefill_experts=np.mean(p_n), union_experts=np.mean(u_n),
                         added_by_prefill=np.mean(u_n) - np.mean(d_n),
                         added_frac_of_E=(np.mean(u_n) - np.mean(d_n)) / E,
                         random_decode=rnd(bd), random_prefill=rnd(c), random_union=rnd(bd + c)))
        r = rows[-1]
        print(f"bd={bd:3d} c={c:5d}: decode {r['decode_experts']:6.1f} (random {r['random_decode']:6.1f})  "
              f"prefill {r['prefill_experts']:6.1f} (random {r['random_prefill']:6.1f})  "
              f"prefill adds {r['added_by_prefill']:6.1f} = {100 * r['added_frac_of_E']:4.1f}% of E")
a.out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, index=False, float_format="%.3f")
print(f"-> {a.out}  (dump: {a.dump}, model {meta['model']}, dataset {meta['dataset']})")
