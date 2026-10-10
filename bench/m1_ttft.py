#!/usr/bin/env python3
"""TTFT of one append under the M1 schedules (CPU only, from the M1 calibration and best configurations).

For each cell and policy (best configuration within B from bench/m1_eval.py's inputs), one append of
delta tokens arrives at an idle prefill pipeline (decode keeps running): count iterations until its last
token leaves the last sublayer, with the same greedy deep-first chooser and calibrated per-sublayer costs
as bench/hyprefill_emulator.py. The tail of the append (fewer than n tokens) may fire once nothing is left
upstream. TTFT ~ iterations x B (iterations fill the budget; M1 measured p50 at 0.96-0.99 B).
Also reports the steady-state service time delta / R (back-to-back appends).

  python bench/m1_ttft.py results/step05/2026-10-10_m1v2_D* results/step05/2026-10-10_m1v2s*
"""
import json, pathlib, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hyprefill_emulator import CAL_N  # noqa: E402


def kinds_qwen3_next(n_lay=48, fai=4):
    k = []
    for i in range(n_lay):
        k += ["A" if (i + 1) % fai == 0 else "G", "M"]
    return k


def one_append(kinds2, n, inc_l, inc_a, host_pre, P, cpu_left, delta, max_it=20000):
    S = len(kinds2)
    q = [0] * (S + 1); q[0] = delta
    proc = [0] * S
    for it in range(1, max_it + 1):
        left, cl = P, cpu_left
        for s in range(S - 1, -1, -1):
            need = n[kinds2[s]]
            upstream_empty = all(q[j] == 0 for j in range(s))
            take = need if q[s] >= need else (q[s] if (q[s] > 0 and upstream_empty) else 0)
            if not take:
                continue
            c = inc_a(s, take, proc[s] % delta) if kinds2[s] == "A" else inc_l(s, take)
            h = host_pre.get(kinds2[s], 0.0)
            if c <= left and h <= cl:
                left -= c; cl -= h; q[s] -= take; q[s + 1] += take; proc[s] += take
        if q[S] >= delta:
            return it
    return np.nan


def main():
    dirs = [pathlib.Path(x) for x in sys.argv[1:]]
    kinds2 = kinds_qwen3_next()
    rows = []
    for dr in dirs:
        cfg = json.load(open(dr / "config.json"))
        a = cfg["args"]; host_pre = cfg["host_pre"]; host_dec = cfg["host_dec"]
        safety = float(a["safety"]); h0 = float(a["h0"])
        cal = [json.loads(l) for l in open(dr / "calibration.jsonl")]
        summ = pd.read_csv(dr / "summary.csv")
        summ = summ[(summ["mode"] == "graph") & (summ.within_B == 1) & (summ.policy != "decode_only")]
        dec = pd.read_csv(dr / "summary.csv"); dec = dec[dec.policy == "decode_only"]
        for c in cal:
            D, t, delta = c["D"], c["t"], c["delta"]
            gm = {int(k): v for k, v in c["gdn_moe"].items()}
            a0 = {int(k): v for k, v in c["attn_start"].items()}
            a1 = {int(k): v for k, v in c["attn_end"].items()}
            inc_a = lambda s, nn, off, a0=a0, a1=a1, dl=delta: safety * float(
                np.interp(nn, CAL_N, a0[s]) + (np.interp(nn, CAL_N, a1[s]) - np.interp(nn, CAL_N, a0[s])) * min(1.0, off / max(1, dl - nn)))
            inc_l = lambda s, nn, gm=gm: safety * float(np.interp(nn, CAL_N, gm[s]))
            D_ms = float(dec[(dec.D == D) & (dec.t == t)].iter_p50_ms.iloc[0])
            x = summ[(summ.D == D) & (summ.t == t) & (summ.delta == delta)]
            for (B, pol), g in x.groupby(["B", "policy"]):
                r = g.loc[g.R.idxmax()]
                nA, nG, nM = map(int, r.n.split("/"))
                P = B - D_ms
                cpu_left = B - h0 - 12 * host_dec["A"] - 36 * host_dec["G"]
                if pol == "chunked":
                    its = int(np.ceil(delta / nA))
                else:
                    its = one_append(kinds2, {"A": nA, "G": nG, "M": nM}, inc_l, inc_a, host_pre, P, cpu_left, delta)
                rows.append(dict(B=B, D=D, t=t, delta=delta, policy=pol, n=r.n, safety=safety, R=float(r.R),
                                 ttft_iters=its, ttft_ms=its * B, service_ms=delta / float(r.R) * B))
    d = pd.DataFrame(rows)
    # per cell and policy: the configuration with the best R (as in m1_eval), with its TTFT
    d = d.loc[d.groupby(["B", "D", "t", "delta", "policy"]).R.idxmax()]
    p = d.pivot_table(index=["B", "D", "t", "delta"], columns="policy", values=["ttft_ms", "service_ms"])
    p[("ttft_ms", "hy/lay")] = p[("ttft_ms", "hyprefill")] / p[("ttft_ms", "layered")]
    p[("service_ms", "hy/lay")] = p[("service_ms", "hyprefill")] / p[("service_ms", "layered")]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(p.round(2).to_string())
    out = dirs[0].parent / "m1_ttft_2026-10-10.csv"
    d.to_csv(out, index=False, float_format="%.3f"); print("->", out)


if __name__ == "__main__":
    main()
