#!/usr/bin/env python3
"""M1 (PROPOSAL §5, criteria fixed 2026-10-10 before this code ran): run the chunked / Layered /
HyPrefill prefill schedules on a real-kernel emulation of Qwen3-Next-80B-A3B (TP2 shapes per GPU)
and measure prefill tokens per iteration R under a TBT budget B.

Stack: 48 layers x [mixer (36 GDN, 12 gated attention), MoE], each layer with its own weights and
state, built from the kernels vLLM 0.30 serves with (bench/op_cost.py): FA3 on paged KV, FlashInfer
GDN with an initial state, vLLM causal conv1d, fused_experts with real routing (per-layer routing
from the 192-request dump), cuBLAS GEMMs and vLLM norm ops. Weights are random.
Every iteration: D decode tokens go through all 48 layers; prefill tokens go through the sublayers
the schedule fires this iteration. GEMMs and MoE run decode and prefill tokens in one call; FA, GDN
and conv run separate decode and prefill kernels (as vLLM does). The TP all-reduce (2 per layer) is
a GPU sleep of the measured vLLM all-reduce duration (one GPU, no peer).

Costs used by the scheduler are calibrated on this stack (one sublayer with D + n tokens minus with D,
mean of a few layers per kind; the step 1/2 tables under-predict MoE for layers other than the one KT1
used and over-predict attention), the decode part D is measured. Schedules: the greedy deep-first
chooser of bench/kt2_sim_check.py with a GPU budget P = B - D and a CPU budget. chunked = every
sublayer runs the same c each iteration (full depth); Layered = one chunk n for every sublayer,
staggered; HyPrefill = chunk n_A for attention, n_G = n_M for GDN / MoE. R comes from the long schedule;
the GPU executes a window of it to check that iterations stay within B (p99 <= B). Each policy's best
configuration among the top ones of the simulation that stays within B is its result.

Modes:
  graph: each distinct iteration pattern is captured once as a CUDA graph and replayed; GPU time per
         iteration from CUDA events; CPU time modelled as in vLLM's piecewise graphs (h0 + host_ms of
         every eager FA / GDN call). Iteration time = max(GPU, CPU). Decides M1.
  eager: every op launched from Python; wall time per iteration (host perf_counter + sync).

  CUDA_VISIBLE_DEVICES=4 taskset -c 48-71 python bench/hyprefill_emulator.py --kt1 <dir prefix> --routing <dump> --host-from <op_cost summary>
Output: results/step05/<date>_m1_emulator/{config.json, raw.csv, summary.csv}
"""
import argparse, csv, datetime, glob, json, os, pathlib, shlex, subprocess, sys, time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cgroup_cpu  # noqa: E402
import kt2_sim_check as sim  # noqa: E402
from kt2_capacity import load  # noqa: E402
from op_cost import ClockSampler, gpu_state, load_shapes, versions  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
F = torch.nn.functional
DT = torch.bfloat16
DEV = "cuda:0"


def sleep_ms(ms, mhz):
    if ms > 0:
        torch.cuda._sleep(int(ms * mhz * 1e3))


class Stack:
    """Weights, states, KV and persistent input buffers for 48 layers; run(pattern) issues one iteration."""

    def __init__(self, shapes, cfg, D, t, delta, pmax, routing, allreduce, conv_cost, mhz, seed=0):
        from vllm import _custom_ops as ops
        self.ops, self.D, self.t, self.delta, self.allreduce, self.conv_cost, self.mhz = ops, D, t, delta, allreduce, conv_cost, mhz
        g = torch.Generator(device="cpu").manual_seed(seed)
        n_lay, fai = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.kinds = ["A" if (i + 1) % fai == 0 else "G" for i in range(n_lay)]
        a, gd, m, h = shapes["dense_attn"], shapes["dense_gdn"], shapes["moe"], cfg["hidden_size"]
        self.h, self.a, self.gd, self.m = h, a, gd, m
        N = D + pmax  # max tokens of one call
        w = lambda o, i: torch.randn(o, i, device=DEV, dtype=DT) * 0.02
        self.x = torch.randn(N, h, device=DEV, dtype=DT)
        self.res = torch.randn_like(self.x)
        self.ln = torch.ones(h, device=DEV, dtype=DT)
        # attention shapes
        hq, hkv, hd, blk = a["heads"], a["kv_heads"], a["head_dim"], a["kv_block"]
        self.hq, self.hkv, self.hd = hq, hkv, hd
        self.q_out = hq * hd * 2
        self.hn = torch.ones(hd, device=DEV, dtype=DT)
        self.qn = torch.empty(N * hq, hd, device=DEV, dtype=DT)
        self.kn = torch.empty(N * hkv, hd, device=DEV, dtype=DT)
        self.attn_out = torch.randn(N, hq * hd, device=DEV, dtype=DT)
        self.q_pre = torch.randn(pmax, hq, hd, device=DEV, dtype=DT)
        self.q_dec = torch.randn(D, hq, hd, device=DEV, dtype=DT)
        # GDN shapes
        hk, hv, dk, dv, kw = gd["k_heads"], gd["v_heads"], gd["d_k"], gd["d_v"], gd["conv_kernel"]
        self.key_dim, self.value_dim, self.conv_dim, self.hv, self.dv, self.dk = hk * dk, hv * dv, 2 * hk * dk + hv * dv, hv, dv, dk
        self.core = torch.randn(N * hv, dv, device=DEV, dtype=DT)
        self.mixed_pre = torch.randn(pmax, self.conv_dim, device=DEV, dtype=DT)
        self.gq = torch.randn(1, pmax, hk, dk, device=DEV, dtype=DT)
        self.gk = torch.randn(1, pmax, hk, dk, device=DEV, dtype=DT)
        self.gv = torch.randn(1, pmax, hv, dv, device=DEV, dtype=DT)
        self.gg = F.logsigmoid(torch.randn(1, pmax, hv, device=DEV, dtype=torch.float32))
        self.gbeta = torch.rand(1, pmax, hv, device=DEV, dtype=DT).sigmoid()
        self.x_dec_conv = torch.randn(D, self.conv_dim, device=DEV, dtype=DT)
        self.a_dec = torch.randn(D, hv, device=DEV, dtype=DT)
        self.b_dec = torch.randn(D, hv, device=DEV, dtype=DT)
        self.out_dec = torch.empty(D, 1, hv, dv, device=DEV, dtype=DT)
        self.idx_dec = torch.arange(D, device=DEV, dtype=torch.int32)
        self.idx0 = torch.zeros(1, device=DEV, dtype=torch.int32)
        self.has1 = torch.ones(1, device=DEV, dtype=torch.bool)
        from vllm.model_executor.layers.layernorm import RMSNormGated
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            self.gnorm = RMSNormGated(dv, eps=1e-6, group_size=None, norm_before_gate=True, device=DEV, dtype=DT)
        # MoE shapes
        E, k, inter, sh = m["experts"], m["topk"], m["inter"], m["shared_inter"]
        self.s_act = torch.empty(N, sh, device=DEV, dtype=DT)
        # KV: one physical sequence of t + delta tokens shared by every decode row and the prefill
        # (identical compute, a fraction of the memory); block size as vLLM picks it
        nb = -(-(t + delta + pmax) // blk)
        self.L = []
        for i, kind in enumerate(self.kinds):
            L = {"kind": kind}
            if kind == "A":
                L["w_qkv"], L["w_o"] = w(self.q_out + 2 * hkv * hd, h), w(h, hq * hd)
                L["kc"] = torch.randn(nb, blk, hkv, hd, device=DEV, dtype=DT)
                L["vc"] = torch.randn_like(L["kc"])
                L["bt"] = torch.arange(nb, device=DEV, dtype=torch.int32).unsqueeze(0)
                L["bt_dec"] = L["bt"].expand(D, nb).contiguous()
            else:
                L["w_qkvz"], L["w_ba"], L["w_out"] = w(2 * self.key_dim + 2 * self.value_dim, h), w(2 * hv, h), w(h, self.value_dim)
                L["w_conv"] = torch.randn(self.conv_dim, kw, device=DEV, dtype=DT)
                L["conv_state"] = torch.zeros(1, kw - 1, self.conv_dim, device=DEV, dtype=DT).transpose(1, 2)
                L["conv_dec"] = torch.zeros(D, kw - 1, self.conv_dim, device=DEV, dtype=DT).transpose(-1, -2)
                L["A_log"] = torch.randn(hv, device=DEV, dtype=torch.float32)
                L["dt_bias"] = torch.randn(hv, device=DEV, dtype=torch.float32)
                ssm = getattr(torch, gd["ssm_dtype"])
                L["h0"] = (torch.randn(1, hv, dk, dv, device=DEV, dtype=torch.float32) * 0.02).to(ssm)
                L["state_dec"] = torch.randn(D, hv, dv, dk, device=DEV, dtype=ssm) * 0.02
            L["w_router"] = w(E, h)
            L["w1"] = torch.randn(E, 2 * inter, h, device=DEV, dtype=DT) * 0.02
            L["w2"] = torch.randn(E, h, inter, device=DEV, dtype=DT) * 0.02
            L["w_sgu"], L["w_sd"], L["w_sg"] = w(2 * sh, h), w(h, sh), w(1, h)
            # routing ids: D decode tokens (one generated position of D distinct requests), then pmax
            # consecutive prompt tokens of another request, layer i of the dump
            reqs = routing
            perm = torch.randperm(len(reqs), generator=g).tolist()
            rows = []
            for j in perm[1:D + 1]:
                ex, plen = reqs[j]
                rows.append(ex[int(torch.randint(plen, ex.shape[0], (1,), generator=g)), i])
            ex, plen = reqs[perm[0]]
            s0 = int(torch.randint(0, max(1, plen - pmax), (1,), generator=g))
            span = np.resize(ex[s0:s0 + pmax, i], (pmax, k))
            L["ti"] = torch.tensor(np.concatenate([np.stack(rows).reshape(D, k), span]), device=DEV, dtype=torch.int32)
            L["tw"] = torch.softmax(torch.randn(N, k, device=DEV), -1)
            self.L.append(L)
        # per-call device scalars for FA prefill (seqused) and GDN prefill (cu_seqlens), updated before replay
        self.used_pre = {i: torch.zeros(1, device=DEV, dtype=torch.int32) for i, k_ in enumerate(self.kinds) if k_ == "A"}
        self.used_dec = torch.full((D,), t, device=DEV, dtype=torch.int32)
        self.cu_dec = torch.arange(D + 1, device=DEV, dtype=torch.int32)
        self.cu_pre = {}
        self.qsl = {}
        self.max_k = t + delta + pmax

    # -- sublayers ---------------------------------------------------------------------------------
    def attn(self, i, p, conv_real=True):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        L, ops, D, n = self.L[i], self.ops, self.D, self.D + p
        x, res = self.x[:n], self.res[:n]
        hq, hkv, hd, q_out = self.hq, self.hkv, self.hd, self.q_out
        ops.fused_add_rms_norm(x, res, self.ln, 1e-6)
        qkv = F.linear(x, L["w_qkv"])
        ops.rms_norm(self.qn[:n * hq], qkv[:, :hq * hd].reshape(-1, hd), self.hn, 1e-6)
        ops.rms_norm(self.kn[:n * hkv], qkv[:, q_out:q_out + hkv * hd].reshape(-1, hd), self.hn, 1e-6)
        flash_attn_varlen_func(self.q_dec, L["kc"], L["vc"], max_seqlen_q=1, cu_seqlens_q=self.cu_dec,
                               max_seqlen_k=self.max_k, seqused_k=self.used_dec, block_table=L["bt_dec"],
                               causal=True, fa_version=3)
        if p:
            flash_attn_varlen_func(self.q_pre[:p], L["kc"], L["vc"], max_seqlen_q=p, cu_seqlens_q=self._cu(p),
                                   max_seqlen_k=self.max_k, seqused_k=self.used_pre[i], block_table=L["bt"],
                                   causal=True, fa_version=3)
        o = self.attn_out[:n] * torch.sigmoid(qkv[:, hq * hd:q_out])
        y = F.linear(o, L["w_o"])
        sleep_ms(self.allreduce(n), self.mhz)
        ops.fused_add_rms_norm(y, res, self.ln, 1e-6)

    def gdn(self, i, p, conv_real=True):
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
        L, ops, D, n = self.L[i], self.ops, self.D, self.D + p
        x, res = self.x[:n], self.res[:n]
        ops.fused_add_rms_norm(x, res, self.ln, 1e-6)
        qkvz = F.linear(x, L["w_qkvz"])
        F.linear(x, L["w_ba"])
        # decode: conv update + fused recurrent decode on D sequences
        y = m.causal_conv1d_update(self.x_dec_conv, L["conv_dec"], L["w_conv"], None, "silu",
                                   conv_state_indices=self.idx_dec, validate_data=False)
        m.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=y, a=self.a_dec, b=self.b_dec, A_log=L["A_log"], dt_bias=L["dt_bias"], scale=self.dk ** -0.5,
            initial_state=L["state_dec"], out=self.out_dec, ssm_state_indices=self.idx_dec, use_qk_l2norm_in_kernel=True)
        if p:
            if conv_real:  # builds launch metadata on the host: not capturable, eager mode only
                causal_conv1d_fn(self.mixed_pre[:p].transpose(0, 1), L["w_conv"], None, activation="silu",
                                 conv_states=L["conv_state"], has_initial_state=self.has1, cache_indices=self.idx0,
                                 query_start_loc=self._qsl(p), metadata=None)
            else:
                sleep_ms(self.conv_cost(p), self.mhz)
            m.fi_chunk_gated_delta_rule(q=self.gq[:, :p], k=self.gk[:, :p], v=self.gv[:, :p], g=self.gg[:, :p],
                                        beta=self.gbeta[:, :p], initial_state=L["h0"], output_final_state=True,
                                        cu_seqlens=self._qsl(p), use_qk_l2norm_in_kernel=True)
        z = qkvz[:, self.conv_dim:self.conv_dim + self.value_dim].reshape(-1, self.dv)
        o = self.gnorm.forward_cuda(self.core[:n * self.hv], z)
        y2 = F.linear(o.reshape(n, self.value_dim), L["w_out"])
        sleep_ms(self.allreduce(n), self.mhz)
        ops.fused_add_rms_norm(y2, res, self.ln, 1e-6)

    def moe(self, i, p):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        L, n = self.L[i], self.D + p
        x = self.x[:n]
        F.linear(x, L["w_router"])
        y = fused_experts(x, L["w1"], L["w2"], L["tw"][:n], L["ti"][:n])
        torch.ops._C.silu_and_mul(self.s_act[:n], F.linear(x, L["w_sgu"]))
        y = y + torch.sigmoid(F.linear(x, L["w_sg"])) * F.linear(self.s_act[:n], L["w_sd"])
        sleep_ms(self.allreduce(n), self.mhz)

    def _cu(self, p):
        if p not in self.cu_pre:
            self.cu_pre[p] = torch.tensor([0, p], device=DEV, dtype=torch.int32)
        return self.cu_pre[p]

    _qsl = _cu

    def run(self, pattern, conv_real=True):
        """pattern: tuple of prefill tokens per sublayer (2 per layer: mixer, MoE)."""
        for i, kind in enumerate(self.kinds):
            pm, pe = pattern[2 * i], pattern[2 * i + 1]
            (self.attn if kind == "A" else self.gdn)(i, pm, conv_real)
            self.moe(i, pe)

    def set_offsets(self, offsets):
        """Context of each attention firing this iteration: seqused = t + offset + p (bottom-right causal)."""
        for i, used in offsets.items():
            self.used_pre[i].fill_(used)


CAL_N = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def gtime(fn, n=15):
    """GPU time of fn as one CUDA graph replay (median), warmed up on a side stream."""
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    del g
    return float(np.median(ts))


def calibrate(stack, kinds2, t, delta, ns, cache=None):
    """Per-sublayer cost added by a prefill firing of n tokens: GPU time of the sublayer with D + n tokens
    minus with D (CUDA graph, median). MoE and GDN do not depend on the context: measured once per stack
    (cache); attention at the start (seqused t + n) and the end (t + delta) of the append.
    Returns per-sublayer tables inc[s] (list over ns) and, for attention sublayers, inc0 / inc1."""
    call = {"A": lambda i, p: stack.attn(i, p, False), "G": lambda i, p: stack.gdn(i, p, False), "M": stack.moe}
    if cache is None:
        cache = {}
        for s_, k in enumerate(kinds2):
            if k == "A":
                continue
            i = s_ // 2
            base = gtime(lambda: call[k](i, 0))
            cache[s_] = [gtime(lambda: call[k](i, n)) - base for n in ns]
    inc0, inc1 = {}, {}
    for s_, k in enumerate(kinds2):
        if k != "A":
            continue
        i = s_ // 2
        base = gtime(lambda: call["A"](i, 0))
        for ctx, dst in (("start", inc0), ("end", inc1)):
            row = []
            for n in ns:
                stack.used_pre[i].fill_(t + (min(n, delta) if ctx == "start" else delta))
                row.append(gtime(lambda: call["A"](i, n)) - base)
            dst[s_] = row
    return cache, inc0, inc1


def schedule(kinds2, n, inc, host_pre, P, cpu_left, iters, warm, inc_a=None, delta=None):
    """Greedy deep-first firings (kt2_sim_check.simulate2) with GPU budget P and CPU budget cpu_left
    per iteration; returns per-iteration patterns and tokens leaving the last sublayer. With inc_a(n, off),
    an attention firing is charged at its offset within the current append (cost grows with context)."""
    S = len(kinds2)
    cost = [inc(s, n[k]) for s, k in enumerate(kinds2)]  # inc(sublayer, tokens)
    hc = [host_pre.get(k, 0.0) for k in kinds2]
    q = [0] * (S + 1); q[0] = 10 ** 12
    proc = [0] * S
    pats, done = [], []
    for it in range(iters):
        left, cl, pat = P, cpu_left, [0] * S
        for s in range(S - 1, -1, -1):
            c = inc_a(s, n["A"], proc[s] % delta) if (inc_a and kinds2[s] == "A") else cost[s]
            if q[s] >= n[kinds2[s]] and c <= left and hc[s] <= cl:
                left -= c; cl -= hc[s]; q[s] -= n[kinds2[s]]; q[s + 1] += n[kinds2[s]]; pat[s] = n[kinds2[s]]
                proc[s] += n[kinds2[s]]
        pats.append(tuple(pat)); done.append(q[S]); q[S] = 0
    return pats, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct_tp2")
    ap.add_argument("--config", default=str(REPO / "results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json"))
    ap.add_argument("--kt1", required=True, help="moe_mixed results prefix (predictions)")
    ap.add_argument("--routing", required=True, help="routing dump dir")
    ap.add_argument("--host-from", required=True, help="op_cost.py summary with host_ms of fa, gdn, fa_decode, gdn_decode")
    ap.add_argument("--B", default="25,50,100")
    ap.add_argument("--D", default="8,32")
    ap.add_argument("--t", default="131072,262144")
    ap.add_argument("--delta", default="2048,8192")
    ap.add_argument("--h0", type=float, default=10.0)
    ap.add_argument("--topk", type=int, default=5, help="configurations per policy taken from the simulation")
    ap.add_argument("--iters", type=int, default=200, help="measured iterations (executed on the GPU)")
    ap.add_argument("--warm", type=int, default=600, help="iterations scheduled (not executed) to fill the pipeline")
    ap.add_argument("--modes", default="graph,eager")
    ap.add_argument("--safety", type=float, default=1.04, help="multiply the calibrated costs when scheduling; same for every policy")
    ap.add_argument("--long", type=int, default=3000, help="scheduled iterations for R (after --warm)")
    ap.add_argument("--clock-locked", action="store_true")
    ap.add_argument("--mhz", type=float, default=1980.0, help="SM clock for the all-reduce sleep")
    ap.add_argument("--tag", default="m1_emulator")
    a = ap.parse_args()

    cfg, cost, allreduce, moe_pred, host, _ = load(a.model, a.kt1)
    import pandas as pd
    hh = pd.read_csv(a.host_from)
    hmed = lambda op: float(hh[hh.op == op].host_ms.median())
    # CPU per iteration (vLLM piecewise graphs run FA and GDN eagerly): every layer's decode call, plus one
    # prefill call per firing; decode and prefill host costs measured separately (op_cost host_ms)
    host_dec = {"A": hmed("fa_decode"), "G": hmed("gdn_decode")}
    host_pre = {"A": hmed("fa"), "G": hmed("gdn")}
    from kt2_capacity import capacity
    shapes = load_shapes(a.config, 2)
    from op_cost import load_routing
    routing = load_routing(a.routing)["reqs"]
    conv_cost = lambda p: float(cost("gdn_conv", p))
    kinds2 = []
    n_lay, fai = cfg["num_hidden_layers"], cfg["full_attention_interval"]
    lt = ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(n_lay)]
    for l in lt:
        kinds2 += ["A" if l == "full_attention" else "G", "M"]
    n_attn, n_gdn = kinds2.count("A"), kinds2.count("G")

    torch.zeros(1, device=DEV)
    before = gpu_state()
    busy = [g for g in before if int(g["utilization.gpu"]) > 5 or int(g["memory.used"]) * 10 > int(g["memory.total"])]
    if busy:
        sys.exit(f"GPU busy: {busy}")
    out = REPO / "results/step05" / f"{datetime.date.today()}_{a.tag}"
    i = 2
    while out.exists():
        out = out.with_name(f"{datetime.date.today()}_{a.tag}_run{i}"); i += 1
    out.mkdir(parents=True)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    meta = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), args=vars(a), host_dec=host_dec, host_pre=host_pre, shapes=shapes,
                gpu=before, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), **versions(),
                repo_commit=git("rev-parse", "HEAD"), repo_dirty=bool(git("status", "--porcelain")),
                cmd=" ".join(shlex.quote(x) for x in sys.argv))
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    rawf = open(out / "raw.csv", "w", newline=""); raw = csv.writer(rawf)
    raw.writerow(["B", "D", "t", "delta", "policy", "n", "mode", "iter", "gpu_ms", "cpu_model_ms", "wall_ms", "iter_ms", "tokens_done"])
    sumf = open(out / "summary.csv", "w", newline=""); summ = csv.writer(sumf)
    summ.writerow(["B", "D", "t", "delta", "policy", "n", "mode", "R", "unused", "iter_p50_ms", "iter_p99_ms",
                   "frac_over_B", "within_B", "gpu_p50_ms", "decode_ms", "n_patterns", "sm_clock_min_mhz"])
    calf = open(out / "calibration.jsonl", "w")
    sampler = ClockSampler()
    cg0 = cgroup_cpu.snapshot()
    pmax = max(int(x) for x in a.delta.split(","))

    for D in map(int, a.D.split(",")):
        for t in map(int, a.t.split(",")):
            torch.cuda.empty_cache()
            stack = Stack(shapes, cfg, D, t, pmax, pmax, routing, allreduce, conv_cost, a.mhz)
            zero = tuple(0 for _ in kinds2)
            for i in stack.used_pre:
                stack.used_pre[i].fill_(t + 1)
            D_ms = gtime(lambda: stack.run(zero, conv_real=False))
            d_pred = 100 - capacity(cfg, cost, allreduce, moe_pred, host, 100, D, t, pmax, a.h0)["P_ms"]
            print(f"D={D} t={t}: decode-only iteration GPU {D_ms:.2f} ms measured ({d_pred:.2f} predicted from tables)", flush=True)
            cal_gm = None  # GDN / MoE calibration, reused across delta
            summ.writerow([0, D, t, 0, "decode_only", "-", "graph", 0, 0, f"{D_ms:.3f}", "", 0, 1, f"{D_ms:.3f}", f"{d_pred:.3f}", 1, ""])
            for delta in map(int, a.delta.split(",")):
                # per-sublayer cost added by a prefill firing of n tokens, measured on this stack (mean of a few
                # layers per kind: routing differs by layer), attention at the mid-append context
                cal_gm, inc0, inc1 = calibrate(stack, kinds2, t, delta, CAL_N, cal_gm)
                calf.write(json.dumps(dict(D=D, t=t, delta=delta, n=CAL_N, gdn_moe=cal_gm, attn_start=inc0, attn_end=inc1)) + "\n"); calf.flush()

                def inc_a(s_, nn, off, dl=delta):  # attention sublayer s_ firing at offset off of the append
                    c0, c1 = np.interp(nn, CAL_N, inc0[s_]), np.interp(nn, CAL_N, inc1[s_])
                    return a.safety * float(c0 + (c1 - c0) * min(1.0, off / max(1, dl - nn)))

                def inc_l(s_, nn):  # per sublayer; attention at mid-append
                    if kinds2[s_] == "A":
                        return inc_a(s_, nn, (delta - nn) / 2)
                    return a.safety * float(np.interp(nn, CAL_N, cal_gm[s_]))
                # per-kind mean, used only to rank candidate configurations in the simulation
                inc_s = {k: (lambda k: (lambda nn: float(np.mean([inc_l(s_, nn) for s_, x in enumerate(kinds2) if x == k]))))(k) for k in "AGM"}
                for B in map(int, a.B.split(",")):
                    P = B - D_ms
                    cpu_left = B - a.h0 - n_attn * host_dec["A"] - n_gdn * host_dec["G"]
                    if P <= 0 or cpu_left <= 0:
                        print(f"B={B} D={D} t={t}: no room (P {P:.1f} ms, CPU left {cpu_left:.1f} ms)", flush=True); continue
                    g = [n for n in sim.GRID if n <= delta]
                    cands = {"chunked": [], "layered": [], "hyprefill": []}
                    for n in range(256, delta + 1, 256):  # chunked: the chunk crosses the full depth every iteration
                        if (sum(inc_l(s_, n) for s_, k in enumerate(kinds2) if k != "A")
                                + sum(inc_a(s_, n, delta - n) for s_, k in enumerate(kinds2) if k == "A") <= P) and n_attn * host_pre["A"] + n_gdn * host_pre["G"] <= cpu_left:
                            cands["chunked"].append((n, (n, n, n)))
                    cands["chunked"] = sorted(cands["chunked"], reverse=True)[:a.topk]
                    lay = sorted(((sim.simulate2(kinds2, inc_s, host_pre, {k: n for k in "AGM"}, P, cpu_left, "deep"), (n, n, n)) for n in g), reverse=True)
                    hy = sorted(((sim.simulate2(kinds2, inc_s, host_pre, {"A": na, "G": nb, "M": nb}, P, cpu_left, "deep"), (na, nb, nb))
                                 for na in g for nb in g), reverse=True)
                    cands["layered"] = [x for x in lay if x[0] > 0][:a.topk]
                    # HyPrefill's configuration space includes k = 1: always try Layered's candidates too, so a
                    # near-miss of the budget by its own top configurations cannot rank it below Layered
                    hy_top = [x for x in hy if x[0] > 0][:a.topk]
                    cands["hyprefill"] = hy_top + [x for x in cands["layered"] if x[1] not in {c[1] for c in hy_top}]
                    for pol, lst in cands.items():
                        for _, (nA, nG, nM) in lst:
                            n = {"A": nA, "G": nG, "M": nM}
                            if pol == "chunked":
                                pats = [tuple(nA for _ in kinds2)] * (a.warm + a.iters)
                                R_long = float(nA)
                            else:
                                pats, done = schedule(kinds2, n, inc_l, host_pre, P, cpu_left, a.warm + a.long, a.warm, inc_a, delta)
                                R_long = float(np.mean(done[a.warm:]))
                                pats = pats[:a.warm + a.iters]
                            for mode in a.modes.split(","):
                                res = run_policy(stack, kinds2, pats, mode, a, host_dec, host_pre, sampler, t, delta)
                                if res is None:
                                    continue
                                it_ms, gpu_ms, cpu_ms, wall_ms, npat, clk = res
                                for j, (x1, x2, x3, x4) in enumerate(zip(gpu_ms, cpu_ms, wall_ms, it_ms)):
                                    raw.writerow([B, D, t, delta, pol, f"{nA}/{nG}/{nM}", mode, j, f"{x1:.4f}", f"{x2:.4f}", f"{x3:.4f}", f"{x4:.4f}", ""])
                                p50, p99 = np.percentile(it_ms, 50), np.percentile(it_ms, 99)
                                over = float(np.mean(np.array(it_ms) > B))
                                summ.writerow([B, D, t, delta, pol, f"{nA}/{nG}/{nM}", mode, f"{R_long:.1f}", "", f"{p50:.3f}", f"{p99:.3f}",
                                               f"{over:.3f}", int(p99 <= B), f"{np.percentile(gpu_ms, 50):.3f}", f"{D_ms:.3f}", npat, clk])
                                sumf.flush(); rawf.flush()
                                print(f"B={B} D={D} t={t} delta={delta} {pol:9s} n={nA}/{nG}/{nM} {mode}: R={R_long:7.1f} "
                                      f"iter p50 {p50:6.2f} p99 {p99:6.2f} ms (GPU p50 {np.percentile(gpu_ms, 50):6.2f}), "
                                      f"over B {100 * over:4.1f}%, patterns {npat}", flush=True)
            del stack
    meta["cgroup_cpu"] = cgroup_cpu.delta(cg0); cgroup_cpu.warn(meta["cgroup_cpu"])
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    print("wrote", out.relative_to(REPO))


def run_policy(stack, kinds2, pats, mode, a, host_dec, host_pre, sampler, t, delta):
    """Execute the iterations after a.warm; returns per-iteration times. Attention firings get context
    t + offset within the current append."""
    att = [i // 2 for i, k in enumerate(kinds2) if k == "A"]
    processed = {i: 0 for i in att}
    for pat in pats[:a.warm]:  # warmup iterations are scheduled only: advance the append offsets
        for li in att:
            processed[li] += pat[2 * li]
    graphs, it_ms, gpu_ms, cpu_ms, wall_ms = {}, [], [], [], []
    n_attn_layers = len(att)
    n_gdn_layers = kinds2.count("G")
    try:
        with sampler:
            for j, pat in enumerate(pats):
                if j < a.warm:
                    continue
                offs = {}
                for li in att:
                    p = pat[2 * li]
                    if p:
                        offs[li] = t + (processed[li] % delta) + p
                        processed[li] += p
                stack.set_offsets(offs)
                n_fa = sum(1 for li in att if pat[2 * li])
                n_g = sum(1 for i, k in enumerate(kinds2) if k == "G" and pat[i])
                cpu_model = (a.h0 + n_attn_layers * host_dec["A"] + n_gdn_layers * host_dec["G"]
                             + n_fa * host_pre["A"] + n_g * host_pre["G"])
                if mode == "graph":
                    if pat not in graphs:
                        if len(graphs) > 64:
                            graphs.clear(); torch.cuda.empty_cache()
                        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            stack.run(pat, conv_real=False)
                        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
                        gr = torch.cuda.CUDAGraph()
                        if not hasattr(stack, "pool"):
                            stack.pool = torch.cuda.graph_pool_handle()
                        with torch.cuda.graph(gr, pool=stack.pool):
                            stack.run(pat, conv_real=False)
                        graphs[pat] = gr
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    w0 = time.perf_counter(); e0.record(); graphs[pat].replay(); e1.record(); torch.cuda.synchronize()
                    w = (time.perf_counter() - w0) * 1e3
                    g_ms = e0.elapsed_time(e1)
                    it = max(g_ms, cpu_model)
                else:
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    w0 = time.perf_counter(); e0.record(); stack.run(pat, conv_real=True); e1.record(); torch.cuda.synchronize()
                    w = (time.perf_counter() - w0) * 1e3
                    g_ms = e0.elapsed_time(e1)
                    it = w
                if j >= a.warm:
                    it_ms.append(it); gpu_ms.append(g_ms); cpu_ms.append(cpu_model); wall_ms.append(w)
        _, clk_min, _ = sampler.summary()
    except torch.cuda.OutOfMemoryError:
        print("OOM, skipped", flush=True); torch.cuda.empty_cache(); return None
    return it_ms, gpu_ms, cpu_ms, wall_ms, len(graphs), clk_min


if __name__ == "__main__":
    main()
