#!/usr/bin/env python3
"""Request-level simulation of chunked / SLOWeave / Layered / HyPrefill on the coding-agent trace
(plan/03-04 support; PROPOSAL §4.4 metric), with costs calibrated by the M1 emulator.

Workload: Poisson arrivals at rate lam; each request takes (context, new tokens after the per-session
prefix cache, output tokens) from the cc-traces-weka requests (bench/trace_stats.py). A request is
admitted when its KV fits (all admitted contexts + outputs <= KV capacity of 2 x H200), prefills its
new tokens, then decodes `out` tokens, one per iteration.

Iteration time = max(GPU, CPU). GPU = decode(D, mean context) + prefill firings. Prefill costs per
sublayer come from the M1 calibration (calibration.jsonl): GDN / MoE per sublayer and chunk size;
attention per layer fitted as a + b n + c n kv (n tokens, kv = keys up to the last one). Decode from the
step 1/2 tables (fa_decode at the mean context, GDN decode, dense, MoE at n = 0, all-reduce), scaled to the
emulator's decode-only measurement. CPU = h0 + decode calls + one host_ms per eager prefill call.

Policies (B = TBT SLO, each tuned over its parameter; PROPOSAL §4.4 / docs/03 §7):
  chunked c   : every iteration up to c prefill tokens (FCFS, may span requests) through the full depth
                (vLLM's fixed chunk; may exceed B when contexts are long)
  sloweave d  : per iteration the largest c (multiple of the unit) with T(D, c) + d <= B (SLOWeave §3)
  layered n   : depth-pipelined, one chunk n per sublayer, greedy deep-first under budget B
  hyprefill   : as layered with n_A for attention, n_G = n_M for GDN / MoE
Staged policies fire a partial chunk when nothing is left upstream (tail of the queue).
Metrics: TTFT (arrival -> last prefill token out), TBT per decode iteration; SLO attainment = share of
requests with TTFT <= SLO_TTFT and every TBT <= B; goodput = largest lam with attainment >= 90%.

  python bench/reqsim.py --calib results/step05/<run dirs> --B 50 --unit 544
"""
import argparse, collections, datetime, glob, json, math, multiprocessing as mp, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kt2_capacity import load  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = "Qwen3-Next-80B-A3B-Instruct_tp2"
KINDS = [k for i in range(48) for k in (("A" if (i + 1) % 4 == 0 else "G"), "M")]
S = len(KINDS)


class Costs:
    def __init__(self, calib_dirs, D_ref, host_from, h0):
        recs, dec_meas = [], []
        for d in calib_dirs:
            d = pathlib.Path(d)
            recs += [json.loads(l) for l in open(d / "calibration.jsonl")]
            sm = pd.read_csv(d / "summary.csv")
            dec_meas += [(int(r.D), int(r.t), float(r.iter_p50_ms), float(r.decode_ms)) for r in sm[sm.policy == "decode_only"].itertuples()]
        Ds = sorted({r["D"] for r in recs})
        self.D_cal = min(Ds, key=lambda x: abs(x - D_ref))
        rs = [r for r in recs if r["D"] == self.D_cal]
        self.ns = rs[0]["n"]
        # GDN / MoE: mean over the calibrations of this D (they do not depend on t)
        gm = collections.defaultdict(list)
        for r in rs:
            for s, row in r["gdn_moe"].items():
                gm[int(s)].append(np.interp(self.ns, r["n"], row))
        self.gm = {s: np.mean(v, axis=0) for s, v in gm.items()}
        # attention: per layer, least squares on (1, n, n * kv)
        self.att = {}
        for s in [i for i, k in enumerate(KINDS) if k == "A"]:
            X, y = [], []
            for r in rs:
                for j, n in enumerate(r["n"]):
                    for kv, val in ((r["t"] + n, r["attn_start"][str(s)][j]), (r["t"] + r["delta"], r["attn_end"][str(s)][j])):
                        X.append([1.0, n, n * kv / 1e6]); y.append(val)
            self.att[s] = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
        cfg, cost, allreduce, moe, host, _ = load(MODEL, str(REPO / "results/step02/2026-10-08_s02_Qwen3-Next-80B-A3B-Instruct_tp2_gpu4_kt1_draw"))
        self.cost, self.allreduce, self.moe = cost, allreduce, moe
        hh = pd.read_csv(host_from)
        hmed = lambda op: float(hh[hh.op == op].host_ms.median())
        self.host_dec = {"A": hmed("fa_decode"), "G": hmed("gdn_decode")}
        self.host_pre = {"A": hmed("fa"), "G": hmed("gdn"), "M": 0.0}
        self.h0 = h0
        # decode model scaled to the emulator's decode-only measurements (heterogeneous contexts)
        ratios = [m / max(p, 1e-6) for (_, _, m, p) in dec_meas if p > 0]
        self.dec_scale = float(np.median(ratios)) if ratios else 1.0
        self._dec = {}

    def decode(self, D, mean_ctx):
        if D == 0:
            return 0.0
        key = (D, int(mean_ctx // 4096))
        if key not in self._dec:
            c = lambda op, n, t=None: float(self.cost(op, n, t))
            t = max(1, key[1] * 4096 + 2048)
            v = (12 * (c("fa_decode", D, t) + c("dense_attn", D)) + 36 * (c("gdn_decode", D) + c("dense_gdn", D))
                 + 48 * (self.moe_dec(D) + 2 * self.allreduce(D)))
            self._dec[key] = self.dec_scale * v
        return self._dec[key]

    def moe_dec(self, D):
        """MoE of a decode-only batch: KT1 (real routing) up to 64 tokens; above, the random-router table
        scaled to match at 64."""
        bds = [8, 32, 64]
        vals = [self.moe(b, 0) for b in bds]
        if D <= 64:
            return float(np.interp(D, bds, vals))
        return vals[-1] * float(self.cost("moe", D)) / float(self.cost("moe", 64))

    def inc_gm(self, s, n):
        return float(np.interp(n, self.ns, self.gm[s])) if n > 0 else 0.0

    def inc_a(self, s, n, kv):
        if n <= 0:
            return 0.0
        a, b, c = self.att[s]
        return max(0.0, a + b * n + c * n * kv / 1e6)

    def cpu_base(self):
        return self.h0 + 12 * self.host_dec["A"] + 36 * self.host_dec["G"]


def workload(trace_csv, lam, n_req, seed):
    d = pd.read_csv(trace_csv)
    rng = np.random.default_rng(seed)
    rows = d.sample(n_req, replace=True, random_state=seed).reset_index(drop=True)
    arr = np.cumsum(rng.exponential(1.0 / lam, n_req))
    reqs = [dict(id=i, arr=float(arr[i]), ctx=int(r.ctx), new=max(64, int(r.new)), out=max(1, int(r.out))) for i, r in rows.iterrows()]
    return reqs


def simulate(policy, params, reqs, C, B, kv_cap, unit, max_time=None):
    """Event loop over iterations. Returns per-request TTFT and worst TBT."""
    pending = collections.deque(sorted(reqs, key=lambda r: r["arr"]))
    admitted_kv = 0
    now = 0.0
    waiting = collections.deque()    # admitted, prefill not started
    # prefill token stream: per sublayer a deque of [req_id, ntok]; for chunked/sloweave a single FCFS queue
    q = [collections.deque() for _ in range(S + 1)]
    left_up = collections.Counter()  # tokens of a request not yet out of the last sublayer
    proc_att = collections.defaultdict(int)  # (req, sublayer) -> tokens done at that attention layer
    info = {r["id"]: r for r in reqs}
    decoding = {}                    # id -> remaining tokens
    ttft, worst_tbt = {}, collections.defaultdict(float)
    st = collections.defaultdict(float)  # diagnostics: where the iteration time goes
    admit_t = {}
    cpu0 = C.cpu_base()
    nA, nG = (params if policy == "hyprefill" else (params, params)) if policy in ("layered", "hyprefill") else (None, None)
    n_of = {"A": nA, "G": nG, "M": nG}
    total = len(reqs)
    while len(ttft) < total or decoding:
        # arrivals and admission (KV of context + output must fit)
        while pending and pending[0]["arr"] <= now:
            waiting.append(pending.popleft())
        while waiting and admitted_kv + waiting[0]["ctx"] + waiting[0]["out"] <= kv_cap:
            r = waiting.popleft(); admitted_kv += r["ctx"] + r["out"]; admit_t[r["id"]] = now
            q[0].append([r["id"], r["new"]]); left_up[r["id"]] = r["new"]
        has_prefill = any(q[s] for s in range(S))
        if not decoding and not has_prefill:
            if not pending:
                break
            now = max(now, pending[0]["arr"]); continue
        D = len(decoding)
        mean_ctx = np.mean([info[i]["ctx"] + info[i]["out"] - decoding[i] for i in decoding]) if D else 0.0
        dec = C.decode(D, mean_ctx)
        P = B - dec
        cpu_left = B - cpu0
        gpu, cpu = dec, cpu0
        finished = []
        if policy in ("chunked", "sloweave"):
            avail = sum(n for _, n in q[0])
            if avail:
                def full_cost(c):
                    tot, segs, take = 0.0, [], c
                    for rid, n in q[0]:
                        if take <= 0:
                            break
                        m = min(n, take); segs.append((rid, m)); take -= m
                    n_tot = c - take
                    for s, k in enumerate(KINDS):
                        if k == "A":
                            tot += sum(C.inc_a(s, m, info[rid]["ctx"] - info[rid]["new"] + (info[rid]["new"] - left_up[rid]) + m) for rid, m in segs)
                        else:
                            tot += C.inc_gm(s, n_tot)
                    return tot, n_tot
                if policy == "chunked":
                    c = params
                else:  # SLOWeave: largest multiple of the unit that fits B - delta
                    lo, hi, c = unit, max(unit, min(avail, 65536)), 0
                    while lo <= hi:
                        mid = (lo + hi) // (2 * unit) * unit or unit
                        cost_mid, _ = full_cost(mid)
                        if cost_mid + params <= P:
                            c, lo = mid, mid + unit
                        else:
                            hi = mid - unit
                    if c == 0 and D == 0:
                        c = unit  # always make progress when idle
                if c:
                    cst, n_tot = full_cost(c)
                    gpu += cst; cpu += 12 * C.host_pre["A"] + 36 * C.host_pre["G"]
                    take = n_tot
                    while take > 0:
                        rid, n = q[0][0]; m = min(n, take); take -= m
                        left_up[rid] -= m
                        if m == n:
                            q[0].popleft()
                        else:
                            q[0][0][1] -= m
                        if left_up[rid] == 0:
                            finished.append(rid)
        else:
            left, cl = P, cpu_left
            fired = [False] * S
            acc = {"left": left, "cl": cl, "gpu": gpu, "cpu": cpu}

            def self_fire(s):
                k = KINDS[s]
                have = sum(n for _, n in q[s])
                upstream_empty = all(not q[j] for j in range(s))
                take = n_of[k] if have >= n_of[k] else (have if upstream_empty else 0)
                if not take:
                    return False
                segs, tt = [], take
                for rid, n in q[s]:
                    if tt <= 0:
                        break
                    m = min(n, tt); segs.append((rid, m)); tt -= m
                if k == "A":
                    cst = sum(C.inc_a(s, m, info[rid]["ctx"] - info[rid]["new"] + proc_att[(rid, s)] + m) for rid, m in segs)
                else:
                    cst = C.inc_gm(s, take)
                h = C.host_pre[k]
                if cst > acc["left"] or h > acc["cl"]:
                    return False
                acc["left"] -= cst; acc["cl"] -= h; acc["gpu"] += cst; acc["cpu"] += h
                tt = take
                while tt > 0:
                    rid, n = q[s][0]; m = min(n, tt); tt -= m
                    if k == "A":
                        proc_att[(rid, s)] += m
                    if m == n:
                        q[s].popleft()
                    else:
                        q[s][0][1] -= m
                    if q[s + 1] and q[s + 1][-1][0] == rid:
                        q[s + 1][-1][1] += m
                    else:
                        q[s + 1].append([rid, m])
                return True
            progress = True
            while progress:  # repeated deep-first sweeps: tokens moved by a firing can go on through the next
                progress = False  # sublayers in the same iteration while the budget allows (each fires once)
                for s in range(S - 1, -1, -1):
                    if fired[s] or not q[s]:
                        continue
                    if not self_fire(s):
                        continue
                    fired[s] = True; progress = True
            gpu, cpu = acc["gpu"], acc["cpu"]
            while q[S]:
                rid, m = q[S].popleft(); left_up[rid] -= m
                if left_up[rid] == 0:
                    finished.append(rid)
        it = max(gpu, cpu)
        st["iters"] += 1; st["time_ms"] += it; st["decode_ms"] += dec; st["prefill_ms"] += gpu - dec
        st["D_sum"] += D; st["kv_sum"] += admitted_kv; st["waiting_sum"] += len(waiting)
        st["cpu_bound"] += cpu > gpu; st["over_B"] += it > B * 1.0001
        st["decode_over_B"] += dec > B
        now += it / 1e3
        for rid in list(decoding):
            worst_tbt[rid] = max(worst_tbt[rid], it)
            decoding[rid] -= 1
            if decoding[rid] == 0:
                del decoding[rid]; admitted_kv -= info[rid]["ctx"] + info[rid]["out"]
        for rid in finished:
            ttft[rid] = now - info[rid]["arr"]
            decoding[rid] = info[rid]["out"]
        if max_time and now > max_time:
            break
    st["admit_wait_s"] = float(np.mean([admit_t[i] - info[i]["arr"] for i in admit_t])) if admit_t else 0.0
    simulate.stats = dict(st)
    return ttft, worst_tbt


def attainment(policy, params, lam, args, C, slo_ttft):
    reqs = workload(args.trace_csv, lam, args.n_req, args.seed)
    ttft, tbt = simulate(policy, params, reqs, C, args.B, args.kv_cap, args.unit, max_time=args.max_time)
    ok = sum(1 for r in reqs if r["id"] in ttft and ttft[r["id"]] <= slo_ttft and tbt.get(r["id"], 0.0) <= args.B * 1.0001)
    t = np.array([ttft[r["id"]] for r in reqs if r["id"] in ttft])
    return ok / len(reqs), (np.percentile(t, 50) if len(t) else np.nan), (np.percentile(t, 99) if len(t) else np.nan)


def goodput(job):
    policy, params, args, slo_ttft = job
    C = CTX["C"]
    lo, hi = 0.0, args.lam_max
    best = (0.0, np.nan, np.nan)
    for _ in range(args.bisect):
        mid = (lo + hi) / 2
        att, p50, p99 = attainment(policy, params, mid, args, C, slo_ttft)
        if att >= 0.9:
            lo, best = mid, (mid, p50, p99)
        else:
            hi = mid
    return dict(policy=policy, params=str(params), goodput=lo, ttft_p50_at_goodput=best[1], ttft_p99_at_goodput=best[2])


CTX = {}


def unloaded_ttft(C, trace_csv, n=4000, c=2048, seed=0):
    """TTFT of a request alone with vLLM's default chunked prefill (2048 tokens), warm prefix cache."""
    d = pd.read_csv(trace_csv).sample(n, replace=True, random_state=seed)
    out = []
    for r in d.itertuples():
        t, done, base = 0.0, 0, int(r.ctx) - int(r.new)
        while done < r.new:
            m = min(c, r.new - done)
            g = sum(C.inc_a(s, m, base + done + m) if k == "A" else C.inc_gm(s, m) for s, k in enumerate(KINDS))
            cpu = C.cpu_base() + 12 * C.host_pre["A"] + 36 * C.host_pre["G"]
            t += max(g, cpu); done += m
        out.append(t / 1e3)
    return float(np.percentile(out, 90)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="+", required=True, help="M1 run directories (calibration.jsonl, summary.csv)")
    ap.add_argument("--host-from", default=str(REPO / "results/step01/2026-10-09_cpu_recheck_Qwen3-Next-80B-A3B-Instruct_tp2_gpu6/summary.csv"))
    ap.add_argument("--trace-csv", default=str(REPO / "results/step04/2026-10-10_trace_stats/requests_sample.csv"))
    ap.add_argument("--B", type=float, default=50.0)
    ap.add_argument("--D-ref", type=int, default=32, help="calibration decode batch to use")
    ap.add_argument("--h0", type=float, default=10.0)
    ap.add_argument("--unit", type=int, default=544)
    ap.add_argument("--kv-cap", type=float, default=4.5e6)
    ap.add_argument("--n-req", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-max", type=float, default=8.0)
    ap.add_argument("--bisect", type=int, default=7)
    ap.add_argument("--max-time", type=float, default=3600.0)
    ap.add_argument("--slo-ttft", type=float, default=None, help="seconds; default 5 x unloaded P90 (PROPOSAL §4.4)")
    ap.add_argument("--only-slo", action="store_true", help="print the unloaded TTFT P90 and SLO_TTFT, then stop")
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    C = Costs(a.calib, a.D_ref, a.host_from, a.h0)
    p90, _ = unloaded_ttft(C, a.trace_csv)
    slo = a.slo_ttft or 5 * p90
    print(f"calibration D = {C.D_cal}, chunk grid {C.ns}; decode scale {C.dec_scale:.3f}")
    print(f"unloaded TTFT P90 (chunked 2048, alone) = {p90:.3f} s -> SLO_TTFT = {slo:.3f} s; SLO_TBT = B = {a.B} ms")
    if a.only_slo:
        return
    CTX["C"] = C
    u = a.unit
    grid = [n for n in C.ns]
    jobs = [("chunked", c, a, slo) for c in grid if c <= 16384]
    jobs += [("sloweave", d, a, slo) for d in (0.0, 1.0, 2.0)]
    jobs += [("layered", n, a, slo) for n in grid]
    jobs += [("hyprefill", (na, nb), a, slo) for na in grid for nb in grid if nb > na]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(goodput, jobs)
    d = pd.DataFrame(res)
    best = d.loc[d.groupby("policy").goodput.idxmax()].sort_values("goodput", ascending=False)
    pd.set_option("display.width", 200)
    print(best.round(3).to_string(index=False))
    base = best[best.policy != "hyprefill"].goodput.max()
    hy = best[best.policy == "hyprefill"].goodput.max()
    print(f"HyPrefill goodput / best baseline = {hy / base if base else float('nan'):.3f}")
    out = pathlib.Path(a.out or REPO / f"results/step04/{datetime.date.today()}_reqsim_B{int(a.B)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, float_format="%.4f"); print("->", out)


if __name__ == "__main__":
    main()
