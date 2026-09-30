#!/usr/bin/env python3
"""Step 1 micro-benchmark: per-layer cost of full attention and GDN vs (c, t).

Calls the same kernels vLLM 0.30 serves with on H200, directly (no engine), with
shapes read from a model config.json.

  FA : vllm_flash_attn.flash_attn_varlen_func, q_len = c, kv_len = t + c, causal.
       fa_version=3 by default: vLLM picks FA3 on SM90 (v1/attention/backends/fa_utils.py).
       KV is contiguous, not paged.
  GDN: vLLM's GDN prefill wrapper on c tokens with a non-None initial_state.
       Default backend is FlashInfer, which vLLM selects on SM90
       (_resolve_gdn_prefill_backend in layers/mamba/gdn/qwen_gdn_linear_attn.py);
       --gdn-backend triton uses vLLM's vendored FLA instead. The wrapper includes
       the q/k l2norm and dtype casts the model runs. The state is built by running
       t tokens first (--gdn-state prefill), or random (--gdn-state random).

Timing. Each call is captured in a CUDA graph and the replay is timed with CUDA events:
that is GPU time with no host gaps, which is what a layer costs when the CPU runs ahead of
the GPU. Eager event timing of one isolated call is host-bound for these kernels (the
FlashInfer GDN wrapper takes ~0.2 ms of CPU per call vs ~0.04 ms of GPU at c=64), and CPU
speed on this shared machine drifts, so eager numbers are recorded but are not the cost.
host_ms is CPU time to enqueue one call; it matters because vLLM runs GDN and attention
outside its CUDA graphs.

Follows docs/03_MEASUREMENT.md: CUDA events, >=10 warmup, >=30 iterations,
p50/p99/min/max, peak memory, results/step01/<date>/ never overwritten.

Examples:
  CUDA_VISIBLE_DEVICES=4 taskset -c 48-71 python bench/op_cost.py \
      --config results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json --op fa,gdn
  python bench/op_cost.py --config ... --op gdn --c 64,65,128 --t 0,65536 --tag smoke
"""
import argparse, csv, datetime, json, os, pathlib, shlex, subprocess, sys, time

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_C = [64, 65, 128, 129, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_T = [0, 4096, 16384, 32768, 65536, 131072, 262144]


def load_shapes(path, tp=1):
    """Per-GPU layer shapes under tensor parallel size tp, split the way vLLM splits them:
    attention heads / tp, kv heads / tp but at least 1 (replicated when tp > kv heads),
    GDN k and v heads / tp."""
    cfg = json.load(open(path))
    cfg = cfg.get("text_config", cfg)
    mt = cfg["model_type"]
    if mt not in ("qwen3_next", "qwen3_5_text", "qwen3_5", "qwen3_5_moe", "qwen4_exp_text", "qwen3_moe"):
        raise NotImplementedError(f"model_type {mt}: only Qwen GDN hybrids and qwen3_moe are wired up")
    n = cfg["num_hidden_layers"]
    if "layer_types" in cfg:
        n_attn = sum(x == "full_attention" for x in cfg["layer_types"])
    elif "full_attention_interval" in cfg:
        n_attn = n // cfg["full_attention_interval"]
    else:
        n_attn = n
    h, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    # vLLM keeps the GDN recurrent state in the config's mamba_ssm_dtype (Qwen3.5-family
    # configs such as Qwen3.8 say float32), otherwise in the model dtype (Qwen3-Next: bf16)
    ssm_dtype = cfg.get("mamba_ssm_dtype") or "bfloat16"
    assert h % tp == 0 and (hkv % tp == 0 if hkv >= tp else tp % hkv == 0), "heads not divisible by tp"
    shapes = {
        "model_type": mt, "tp": tp,
        "attn": {"layers": n_attn, "heads": h // tp, "kv_heads": max(1, hkv // tp),
                 "head_dim": cfg.get("head_dim", cfg["hidden_size"] // h)},
        "gdn": None,
    }
    if n_attn != n:
        hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        assert hk % tp == 0 and hv % tp == 0, "GDN heads not divisible by tp"
        shapes["gdn"] = {"layers": n - n_attn, "k_heads": hk // tp, "v_heads": hv // tp,
                         "d_k": cfg["linear_key_head_dim"], "d_v": cfg["linear_value_head_dim"],
                         "conv_kernel": cfg.get("linear_conv_kernel_dim", 4), "ssm_dtype": ssm_dtype}
    shapes["attn"]["kv_block"] = kv_block_size(shapes)
    hidden = cfg["hidden_size"]
    # dense (non-kernel) parts of each layer, per GPU: column-parallel outputs and
    # row-parallel inputs are split by tp; the all-reduce after row-parallel is NOT here
    shapes["dense_attn"] = dict(shapes["attn"], hidden=hidden, gated=mt != "qwen3_moe")
    if shapes["gdn"]:
        shapes["dense_gdn"] = dict(shapes["gdn"], hidden=hidden)
        shapes["gdn_conv"] = shapes["gdn"]
    if cfg.get("num_experts") and "linear_attn_config" not in cfg:  # Qwen MoE (not Kimi)
        inter, sh = cfg["moe_intermediate_size"], cfg.get("shared_expert_intermediate_size", 0)
        assert inter % tp == 0 and sh % tp == 0
        # vLLM default is TP for MoE: every GPU holds all experts, intermediate split by tp
        shapes["moe"] = {"layers": n // cfg.get("decoder_sparse_step", 1), "hidden": hidden,
                         "experts": cfg["num_experts"], "topk": cfg["num_experts_per_tok"],
                         "inter": inter // tp, "shared_inter": sh // tp,
                         "renormalize": int(cfg.get("norm_topk_prob", True))}
    if not (cfg.get("num_experts") or cfg.get("n_routed_experts")):
        inter = cfg["intermediate_size"]
        assert inter % tp == 0
        shapes["dense_mlp"] = {"layers": n, "hidden": hidden, "inter": inter // tp}
    return shapes


def kv_block_size(shapes, align=16):
    """Attention KV block size vLLM 0.30 picks (platforms/interface.py, mamba_cache_mode
    'none', the default): for hybrids the smallest multiple of the FA kernel alignment (16)
    whose attention page is >= one GDN state page; otherwise the default block size 16.
    Checked against the engine log: Qwen3.8-27B TP1 -> 784."""
    a, g = shapes["attn"], shapes["gdn"]
    if g is None:
        return align
    attn_tok = 2 * a["kv_heads"] * a["head_dim"] * 2  # K and V, bf16
    conv_dim = 2 * g["k_heads"] * g["d_k"] + g["v_heads"] * g["d_v"]
    ssm_bytes = {"float32": 4, "bfloat16": 2, "float16": 2}[g["ssm_dtype"]]
    mamba_page = (g["conv_kernel"] - 1) * conv_dim * 2 + g["v_heads"] * g["d_k"] * g["d_v"] * ssm_bytes
    return align * -(-mamba_page // (align * attn_tok))


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def graphed(fn, k=1):
    """Capture k back-to-back calls of fn in one CUDA graph (warmup on a side stream)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k):
            fn()
    return g


FLUSH_BYTES = 256 * 2**20  # > 60 MiB L2 on H200
_flush_buf = None


def flush_l2():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def gpu_time(fn, warmup, iters, target_ms=2.0, max_k=50):
    """Cold-L2 GPU time of one call, free of host launch latency.

    A single graph replay between two events still counts the ~4 us the CPU takes to
    launch the graph, which is 5-10% of a 40 us kernel and moved run to run (2026-09-28).
    So capture k x [flush L2, fn] in one graph, with k chosen so a replay lasts about
    target_ms, and subtract a graph of k x [flush L2]. Flushing puts fn in the cold-L2
    state it has when serving (GDN state and KV come from HBM, other layers run between).
    Returns (per-call samples in ms, k, flush ms per call).
    """
    est = stats(timed(graphed(lambda: (flush_l2(), fn())).replay, 3, 5))["p50"]
    k = max(1, min(max_k, int(target_ms / max(est, 1e-3))))
    g_op = graphed(lambda: (flush_l2(), fn()), k)
    g_fl = graphed(flush_l2, k)
    fl_before = stats(timed(g_fl.replay, warmup, iters))["p50"]
    raw = timed(g_op.replay, warmup, iters)
    fl = (fl_before + stats(timed(g_fl.replay, warmup, iters))["p50"]) / 2  # flush timed around the op
    samples = [(x - fl) / k for x in raw]
    del g_op, g_fl
    return samples, k, fl / k


def kernel_time(fn, warmup, iters):
    """Sum of fn's GPU kernel durations per call, cold L2, for ops that cannot be graph-captured.

    Excludes inter-kernel gaps (graph timing includes them), so it is a slight underestimate
    relative to gpu_time; rows record which method was used.
    """
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        flush_l2(); fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        flush_l2()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        samples.append(sum(e.device_time_total for e in p.key_averages()) / 1e3)
    return samples


def host_ms(fn, n=30):
    """CPU time to enqueue one call. The GPU queue is drained first, and n is small enough
    that the queue does not fill up and block the enqueue."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e3
    torch.cuda.synchronize()
    return dt


def stats(ts):
    s = sorted(ts)
    return {"p50": s[len(s) // 2], "p99": s[min(len(s) - 1, int(len(s) * 0.99))],
            "min": s[0], "max": s[-1], "n": len(s)}


KV_LAYOUT = "paged"  # "paged" (as vLLM stores KV) or "contiguous" (for comparison only)


def _paged_kv(shape, lens, dev, dtype):
    """Paged KV cache for sequences of the given lengths, block size as vLLM picks it,
    blocks allocated in order (a fresh engine hands them out sequentially)."""
    hk, d, blk = shape["kv_heads"], shape["head_dim"], shape["kv_block"]
    per = [-(-n // blk) for n in lens]
    kc = torch.randn(sum(per), blk, hk, d, device=dev, dtype=dtype)
    vc = torch.randn_like(kc)
    bt = torch.zeros(len(lens), max(per), device=dev, dtype=torch.int32)
    o = 0
    for i, n in enumerate(per):
        bt[i, :n] = torch.arange(o, o + n, device=dev, dtype=torch.int32)
        o += n
    used = torch.tensor(lens, device=dev, dtype=torch.int32)
    return kc, vc, bt, used


def make_fa(shape, c, t, dev, dtype, fa_version):
    """Prefill chunk: c query tokens attending to t cached + c new keys, causal."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    h, hk, d = shape["heads"], shape["kv_heads"], shape["head_dim"]
    q = torch.randn(c, h, d, device=dev, dtype=dtype)
    cu_q = torch.tensor([0, c], device=dev, dtype=torch.int32)
    # causal with q_len < kv_len aligns q to the end of kv (bottom-right), as in chunked prefill
    if KV_LAYOUT == "paged":
        kc, vc, bt, used = _paged_kv(shape, [t + c], dev, dtype)
        return lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=c, cu_seqlens_q=cu_q,
                                              max_seqlen_k=t + c, seqused_k=used, block_table=bt,
                                              causal=True, fa_version=fa_version)
    k = torch.randn(t + c, hk, d, device=dev, dtype=dtype)
    v = torch.randn_like(k)
    cu_k = torch.tensor([0, t + c], device=dev, dtype=torch.int32)
    return lambda: flash_attn_varlen_func(q, k, v, max_seqlen_q=c, cu_seqlens_q=cu_q,
                                          max_seqlen_k=t + c, cu_seqlens_k=cu_k,
                                          causal=True, fa_version=fa_version)


def make_fa_decode(shape, b, t, dev, dtype, fa_version):
    """Decode attention: b sequences, 1 query token each, t cached keys each."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    h, hk, d = shape["heads"], shape["kv_heads"], shape["head_dim"]
    q = torch.randn(b, h, d, device=dev, dtype=dtype)
    cu_q = torch.arange(b + 1, device=dev, dtype=torch.int32)
    if KV_LAYOUT == "paged":
        kc, vc, bt, used = _paged_kv(shape, [t] * b, dev, dtype)
        return lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=1, cu_seqlens_q=cu_q, max_seqlen_k=t,
                                              seqused_k=used, block_table=bt, causal=True, fa_version=fa_version)
    k = torch.randn(b * t, hk, d, device=dev, dtype=dtype)
    v = torch.randn_like(k)
    cu_k = torch.arange(b + 1, device=dev, dtype=torch.int32) * t
    return lambda: flash_attn_varlen_func(q, k, v, max_seqlen_q=1, cu_seqlens_q=cu_q, max_seqlen_k=t,
                                          cu_seqlens_k=cu_k, causal=True, fa_version=fa_version)


def make_gdn_decode(shape, b, dev, dtype):
    """Decode GDN as vLLM's _forward_core_decode_non_spec runs it: causal_conv1d_update, then
    fused_recurrent_gated_delta_rule_packed_decode on b sequences; states in model dtype
    (vLLM's default mamba_ssm_cache_dtype 'auto')."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
    hk, hv, dk, dv, kw = shape["k_heads"], shape["v_heads"], shape["d_k"], shape["d_v"], shape["conv_kernel"]
    conv_dim = 2 * hk * dk + hv * dv
    x = torch.randn(b, conv_dim, device=dev, dtype=dtype)
    conv_state = torch.zeros(b, kw - 1, conv_dim, device=dev, dtype=dtype).transpose(-1, -2)
    w_conv = torch.randn(conv_dim, kw, device=dev, dtype=dtype)
    a_ = torch.randn(b, hv, device=dev, dtype=dtype)
    b_ = torch.randn(b, hv, device=dev, dtype=dtype)
    A_log = torch.randn(hv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(hv, device=dev, dtype=torch.float32)
    state = torch.randn(b, hv, dv, dk, device=dev, dtype=getattr(torch, shape["ssm_dtype"])) * 0.02
    out = torch.empty(b, 1, hv, dv, device=dev, dtype=dtype)
    idx = torch.arange(b, device=dev, dtype=torch.int32)

    def fn():
        y = m.causal_conv1d_update(x, conv_state, w_conv, None, "silu", conv_state_indices=idx, validate_data=False)
        m.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=y, a=a_, b=b_, A_log=A_log, dt_bias=dt_bias, scale=dk ** -0.5, initial_state=state,
            out=out, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    return fn


GDN_BACKEND = "flashinfer"


def gdn_call(q, k, v, g, beta, cu, h0):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
    fn = m.fi_chunk_gated_delta_rule if GDN_BACKEND == "flashinfer" else m.fla_chunk_gated_delta_rule
    return fn(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
              output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)


def gdn_inputs(shape, n, dev, dtype):
    # vLLM passes q/k with num_k_heads and v with num_v_heads (GVA); both kernels
    # handle the head ratio themselves, so no repeat here.
    hk, hv, dk, dv = shape["k_heads"], shape["v_heads"], shape["d_k"], shape["d_v"]
    q = torch.randn(1, n, hk, dk, device=dev, dtype=dtype)
    k = torch.randn(1, n, hk, dk, device=dev, dtype=dtype)
    v = torch.randn(1, n, hv, dv, device=dev, dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(1, n, hv, device=dev, dtype=torch.float32))
    beta = torch.rand(1, n, hv, device=dev, dtype=dtype).sigmoid()
    # vLLM always runs prefill in varlen mode (FlashInfer requires cu_seqlens); like vLLM's
    # attention metadata, it is built once outside the timed call
    cu = torch.tensor([0, n], device=dev, dtype=torch.int32)
    return q, k, v, g, beta, cu


def gdn_state(shape, t, mode, dev, dtype, prefill_chunk=16384):
    """Initial state as vLLM stores it (mamba_ssm_dtype): the FlashInfer wrapper casts it to
    fp32 on every call, so a bf16 state (Qwen3-Next) includes that copy in the timed cost."""
    hv, dk, dv = shape["v_heads"], shape["d_k"], shape["d_v"]
    h = torch.zeros(1, hv, dk, dv, device=dev, dtype=torch.float32)
    if t and mode == "random":
        h = torch.randn_like(h) * 0.02
    elif t:
        done = 0
        while done < t:  # accumulate a real state over t tokens, in pieces to bound memory
            n = min(prefill_chunk, t - done)
            _, h = gdn_call(*gdn_inputs(shape, n, dev, dtype), h)
            done += n
    return h.to(getattr(torch, shape["ssm_dtype"]))


def make_gdn(shape, c, t, dev, dtype, state_mode):
    h0 = gdn_state(shape, t, state_mode, dev, dtype)
    args = gdn_inputs(shape, c, dev, dtype)
    return lambda: gdn_call(*args, h0)


MOE_TOUCHED = [None]  # distinct experts routed to by one call of the moe op (random router)
ROUTING = None       # real routing dump (bench/moe_routing_dump.py), loaded by --routing


def load_routing(path, seed=0):
    import glob as _g
    import numpy as _np
    reqs = []
    for f in sorted(_g.glob(f"{path}/req_*.npz")):
        z = _np.load(f)
        reqs.append((z["experts"].astype(_np.int64), int(z["prompt_len"])))
    return {"reqs": reqs, "rng": _np.random.default_rng(seed)}


def make_moe_mixed(shape, bd, c, dev, dtype):
    """One fused MoE call over a mixed batch, as vLLM runs it: bd decode tokens (one per
    sequence) + c consecutive prefill tokens of one request, sharing the expert weight reads.
    Routing: real (--routing dump, middle MoE layer; decode tokens from bd distinct requests'
    generated positions, prefill tokens a consecutive prompt span of another request) or,
    without a dump, the random router of the plain moe op. Router GEMM and shared expert are
    included; for this op the summary column t holds bd."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    F = torch.nn.functional
    E, k, inter, sh, h = shape["experts"], shape["topk"], shape["inter"], shape["shared_inter"], shape["hidden"]
    n = bd + c
    w = lambda o, i: torch.randn(o, i, device=dev, dtype=dtype) * 0.02
    x = torch.randn(n, h, device=dev, dtype=dtype)
    w_router = w(E, h)
    w1 = torch.randn(E, 2 * inter, h, device=dev, dtype=dtype) * 0.02
    w2 = torch.randn(E, h, inter, device=dev, dtype=dtype) * 0.02
    w_sgu, w_sd, w_sg = (w(2 * sh, h), w(h, sh), w(1, h)) if sh else (None, None, None)
    s_act = torch.empty(n, sh, device=dev, dtype=dtype) if sh else None
    if ROUTING is None:
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        tw, ti, _ = fused_topk(x, F.linear(x, w_router), k, bool(shape["renormalize"]))
    else:
        import numpy as _np
        reqs, rng = ROUTING["reqs"], ROUTING["rng"]
        layer = reqs[0][0].shape[1] // 2
        idx = rng.permutation(len(reqs))
        assert bd < len(reqs), f"routing dump has {len(reqs)} requests, need > {bd}"
        rows = [reqs[i][0][rng.integers(reqs[i][1], reqs[i][0].shape[0]), layer] for i in idx[1:bd + 1]]
        ex, plen = reqs[idx[0]]
        c_eff = min(c, plen)
        s0 = int(rng.integers(0, plen - c_eff + 1))
        span = _np.resize(ex[s0:s0 + c_eff, layer], (c, k))  # repeats the span if the prompt is shorter than c
        ti = torch.tensor(_np.concatenate([_np.stack(rows).reshape(bd, k), span]), device=dev, dtype=torch.int32)
        tw = torch.softmax(torch.randn(n, k, device=dev), -1)
    MOE_TOUCHED[0] = int(torch.unique(ti).numel())

    def fn():
        F.linear(x, w_router)  # router GEMM runs every call; routing itself is fixed above
        y = fused_experts(x, w1, w2, tw, ti)
        if sh:
            torch.ops._C.silu_and_mul(s_act, F.linear(x, w_sgu))
            y = y + torch.sigmoid(F.linear(x, w_sg)) * F.linear(s_act, w_sd)
        return y
    return fn


def make_dense(op, shape, c, dev, dtype):
    """Per-layer non-kernel work as vLLM runs it (bf16 F.linear = cuBLAS, vLLM norm/conv ops).

    dense_attn: input RMSNorm, qkv_proj (q doubled when gated: Qwen3-Next gated attention),
                q/k RMSNorm per head, sigmoid output gate, o_proj, post-attention RMSNorm.
    dense_gdn:  input RMSNorm, in_proj_qkvz, in_proj_ba, gated RMSNorm per v head, out_proj,
                post RMSNorm.
    gdn_conv:   vLLM causal conv1d over the q/k/v channels (SD state layout, vLLM's default).
                It builds launch metadata on the CPU when called standalone, so it cannot be
                graph-captured; it is timed as the sum of its GPU kernels (kernel_time).
    dense_mlp:  gate_up_proj, SiluAndMul, down_proj (dense-FFN models only).
    Not included: rotary embedding, the TP all-reduce, MoE (step 2).
    """
    if op == "gdn_conv":
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
        hk, hv, dk, dv, kw = shape["k_heads"], shape["v_heads"], shape["d_k"], shape["d_v"], shape["conv_kernel"]
        conv_dim = 2 * hk * dk + hv * dv
        mixed = torch.randn(c, conv_dim, device=dev, dtype=dtype)
        w_conv = torch.randn(conv_dim, kw, device=dev, dtype=dtype)
        conv_state = torch.zeros(1, kw - 1, conv_dim, device=dev, dtype=dtype).transpose(1, 2)
        qsl = torch.tensor([0, c], device=dev, dtype=torch.int32)
        idx = torch.zeros(1, device=dev, dtype=torch.int32)
        has0 = torch.ones(1, device=dev, dtype=torch.bool)
        return lambda: causal_conv1d_fn(mixed.transpose(0, 1), w_conv, None, activation="silu",
                                        conv_states=conv_state, has_initial_state=has0,
                                        cache_indices=idx, query_start_loc=qsl, metadata=None)
    from vllm import _custom_ops as ops
    F = torch.nn.functional
    h = shape["hidden"]
    w = lambda o, i: torch.randn(o, i, device=dev, dtype=dtype) * 0.02
    x = torch.randn(c, h, device=dev, dtype=dtype)
    res = torch.randn_like(x)
    ln = torch.ones(h, device=dev, dtype=dtype)
    if op == "dense_attn":
        hq, hkv, hd = shape["heads"], shape["kv_heads"], shape["head_dim"]
        q_out = hq * hd * (2 if shape["gated"] else 1)
        w_qkv, w_o = w(q_out + 2 * hkv * hd, h), w(h, hq * hd)
        hn = torch.ones(hd, device=dev, dtype=dtype)
        qn, kn = torch.empty(c * hq, hd, device=dev, dtype=dtype), torch.empty(c * hkv, hd, device=dev, dtype=dtype)
        attn_out = torch.randn(c, hq * hd, device=dev, dtype=dtype)

        def fn():
            ops.fused_add_rms_norm(x, res, ln, 1e-6)
            qkv = F.linear(x, w_qkv)
            q, k = qkv[:, : hq * hd], qkv[:, q_out: q_out + hkv * hd]
            ops.rms_norm(qn, q.reshape(-1, hd), hn, 1e-6)
            ops.rms_norm(kn, k.reshape(-1, hd), hn, 1e-6)
            o = attn_out * torch.sigmoid(qkv[:, hq * hd: q_out]) if shape["gated"] else attn_out
            y = F.linear(o, w_o)
            ops.fused_add_rms_norm(y, res, ln, 1e-6)
        return fn
    if op == "dense_gdn":
        from vllm.model_executor.layers.layernorm import RMSNormGated
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
        hk, hv, dk, dv, kw = shape["k_heads"], shape["v_heads"], shape["d_k"], shape["d_v"], shape["conv_kernel"]
        key_dim, value_dim = hk * dk, hv * dv
        conv_dim = 2 * key_dim + value_dim
        w_qkvz, w_ba, w_out = w(2 * key_dim + 2 * value_dim, h), w(2 * hv, h), w(h, value_dim)
        w_conv = torch.randn(conv_dim, kw, device=dev, dtype=dtype)
        conv_state = torch.zeros(1, kw - 1, conv_dim, device=dev, dtype=dtype).transpose(1, 2)
        qsl = torch.tensor([0, c], device=dev, dtype=torch.int32)
        idx = torch.zeros(1, device=dev, dtype=torch.int32)
        has0 = torch.ones(1, device=dev, dtype=torch.bool)
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):  # CustomOp needs a config to pick its backend
            norm = RMSNormGated(dv, eps=1e-6, group_size=None, norm_before_gate=True, device=dev, dtype=dtype)
        core = torch.randn(c * hv, dv, device=dev, dtype=dtype)

        def fn():
            ops.fused_add_rms_norm(x, res, ln, 1e-6)
            qkvz = F.linear(x, w_qkvz)
            F.linear(x, w_ba)
            # the causal conv1d between in_proj and the GDN kernel is measured as op gdn_conv
            z = qkvz[:, conv_dim: conv_dim + value_dim].reshape(-1, dv)  # layout q, k, v, z
            o = norm.forward_cuda(core, z)
            y = F.linear(o.reshape(c, value_dim), w_out)
            ops.fused_add_rms_norm(y, res, ln, 1e-6)
        return fn
    if op == "moe":
        # router (replicated linear) + vLLM fused_topk + Triton fused_experts (vLLM's bf16
        # backend on SM90: fused_moe/oracle/unquantized.py moves FlashInfer behind Triton)
        # + shared expert MLP with its sigmoid gate (not fused into the MoE on NVIDIA)
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        E, k, inter, sh = shape["experts"], shape["topk"], shape["inter"], shape["shared_inter"]
        w_router = w(E, h)
        w1 = torch.randn(E, 2 * inter, h, device=dev, dtype=dtype) * 0.02
        w2 = torch.randn(E, h, inter, device=dev, dtype=dtype) * 0.02
        w_sgu, w_sd, w_sg = (w(2 * sh, h), w(h, sh), w(1, h)) if sh else (None, None, None)
        s_act = torch.empty(c, sh, device=dev, dtype=dtype) if sh else None
        MOE_TOUCHED[0] = None

        def fn():
            logits = F.linear(x, w_router)
            tw, ti, _ = fused_topk(x, logits, k, bool(shape["renormalize"]))
            if MOE_TOUCHED[0] is None:
                MOE_TOUCHED[0] = int(torch.unique(ti).numel())  # before capture: host sync
            y = fused_experts(x, w1, w2, tw, ti)
            if sh:
                torch.ops._C.silu_and_mul(s_act, F.linear(x, w_sgu))
                y = y + torch.sigmoid(F.linear(x, w_sg)) * F.linear(s_act, w_sd)
            return y
        fn()  # records experts touched outside any graph capture
        return fn
    if op == "dense_mlp":
        inter = shape["inter"]
        w_gu, w_d = w(2 * inter, h), w(h, inter)
        act = torch.empty(c, inter, device=dev, dtype=dtype)

        def fn():
            torch.ops._C.silu_and_mul(act, F.linear(x, w_gu))
            F.linear(act, w_d)
        return fn
    raise ValueError(op)


def gpu_state():
    q = "index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.max.sm,temperature.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    keys = q.split(",")
    rows = [dict(zip(keys, (x.strip() for x in line.split(",")))) for line in out]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = set(vis.split(","))
        rows = [r for r in rows if r["index"] in ids]
    return rows


def foreign_load():
    """GPU util and memory not owned by this process, sampled while we are idle.

    Other containers' processes are invisible to nvidia-smi, so a shared GPU can get
    busy mid-sweep; record this per row so contaminated rows can be dropped.
    """
    # utilization.gpu averages over the last sample window, so let our own kernels age out
    torch.cuda.synchronize()
    time.sleep(1.0)
    g = gpu_state()[0]
    ours = torch.cuda.memory_reserved() / 2**20 + CONTEXT_MIB
    return int(g["utilization.gpu"]), max(0, int(g["memory.used"]) - ours)


CONTEXT_MIB = 0.0  # our CUDA context, not in memory_reserved; measured at startup


class ClockSampler:
    """Sample SM clock and power of the GPU in use every ~10 ms while timing runs.

    Locked clocks still drop when the GPU hits its power cap (FA3 at long context does
    on H200), so the clock during the measurement is recorded per row, not assumed.
    """
    def __init__(self):
        import threading, pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(int(vis))
        self.threading = threading

    def __enter__(self):
        self.clk, self.pw, self.stop = [], [], False
        def run():
            while not self.stop:
                self.clk.append(self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM))
                self.pw.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000)
                time.sleep(0.01)
        self.t = self.threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop = True
        self.t.join()

    def summary(self):
        if not self.clk:
            return None, None, None
        return sum(self.clk) / len(self.clk), min(self.clk), max(self.pw)


def versions():
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    for mod in ("vllm", "flashinfer"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:
            v[mod] = f"unavailable: {e}"
    return v


def out_dir(tag):
    base = REPO / "results" / "step01"
    name = datetime.date.today().isoformat() + (f"_{tag}" if tag else "")
    base.mkdir(parents=True, exist_ok=True)
    d, i = base / name, 2
    while True:  # never overwrite: a re-run gets a new directory; mkdir is the atomic claim
        try:
            d.mkdir()
            return d
        except FileExistsError:
            d = base / f"{name}_run{i}"; i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model config.json (see results/step00/configs/)")
    ap.add_argument("--op", default="fa,gdn")
    ap.add_argument("--c", default=",".join(map(str, DEFAULT_C)))
    ap.add_argument("--t", default=",".join(map(str, DEFAULT_T)))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--gdn-state", choices=["prefill", "random"], default="prefill")
    ap.add_argument("--gdn-backend", choices=["flashinfer", "triton"], default="flashinfer")
    ap.add_argument("--fa-version", type=int, choices=[2, 3], default=3)
    ap.add_argument("--kv-layout", choices=["paged", "contiguous"], default="paged")
    ap.add_argument("--decode-batch", default="8,32,64,128", help="decode batch sizes for op moe_mixed")
    ap.add_argument("--routing", default=None, help="routing dump dir (bench/moe_routing_dump.py) for moe_mixed")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size; shapes are per GPU")
    ap.add_argument("--shape", default="", help="override per-GPU shapes, e.g. gdn.v_heads=8,gdn.d_k=64 "
                    "(plan/01 §2.4: vary heads and d_k/d_v to split intra- from inter-chunk cost)")
    ap.add_argument("--clock-locked", action="store_true", help="record that clocks were locked (nvidia-smi -lgc)")
    ap.add_argument("--allow-busy", action="store_true", help="run even if the GPU has other load")
    ap.add_argument("--tag", default="", help="suffix for the results directory")
    a = ap.parse_args()
    assert a.warmup >= 10 and a.iters >= 30, "docs/03_MEASUREMENT.md: warmup >= 10, iters >= 30"

    global GDN_BACKEND, KV_LAYOUT, DECODE_B, ROUTING
    GDN_BACKEND, KV_LAYOUT = a.gdn_backend, a.kv_layout
    DECODE_B = [int(x) for x in a.decode_batch.split(",")]
    ROUTING = load_routing(a.routing) if a.routing else None
    dev, dtype = "cuda:0", torch.bfloat16
    shapes = load_shapes(a.config, a.tp)
    for kv in filter(None, a.shape.split(",")):
        key, val = kv.split("=")
        grp, fld = key.split(".")
        assert grp in shapes and fld in shapes[grp], f"unknown shape field {key}"
        shapes[grp][fld] = int(val)
    ops = a.op.split(",")
    cs = [int(x) for x in a.c.split(",")]
    ts = [int(x) for x in a.t.split(",")]

    global CONTEXT_MIB
    used0 = int(gpu_state()[0]["memory.used"])
    torch.zeros(1, device=dev)  # create the context
    CONTEXT_MIB = int(gpu_state()[0]["memory.used"]) - used0 - torch.cuda.memory_reserved() / 2**20
    before = gpu_state()
    busy = [g for g in before if int(g["utilization.gpu"]) > 5 or int(g["memory.used"]) * 10 > int(g["memory.total"])]
    if busy and not a.allow_busy:
        sys.exit(f"GPU busy, measurements would be contaminated: {busy}\n"
                 "Pick a free GPU with CUDA_VISIBLE_DEVICES or pass --allow-busy (and do not trust the numbers).")

    d = out_dir(a.tag)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_config": a.config, "shapes": shapes, "dtype": "bfloat16",
            "gpu": before, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **versions(), "repo_commit": commit, "repo_dirty": dirty,
            "cmd": " ".join(shlex.quote(x) for x in sys.argv), "clock_locked": a.clock_locked,
            "gdn_state": a.gdn_state, "gdn_backend": a.gdn_backend, "fa_version": a.fa_version,
            "kv_layout": a.kv_layout, "lock_mhz": os.environ.get("LOCK_MHZ"),
            "routing": a.routing, "decode_batch": a.decode_batch, "warmup": a.warmup, "iters": a.iters}
    (d / "config.json").write_text(json.dumps(meta, indent=2))

    raw = csv.writer(open(d / "raw.csv", "w", newline=""))
    raw.writerow(["op", "c", "t", "mode", "iter", "ms"])
    summ_f = open(d / "summary.csv", "w", newline="")
    summ = csv.writer(summ_f)
    # p50/p99/min/max/us_per_token are GPU time from CUDA graph replay (see module docstring)
    summ.writerow(["op", "layers", "c", "t", "p50_ms", "p99_ms", "min_ms", "max_ms", "n",
                   "us_per_token_p50", "eager_p50_ms", "host_ms", "k_per_graph", "flush_ms", "timing", "experts_touched",
                   "peak_alloc_mib", "peak_reserved_mib",
                   "sm_clock_mean_mhz", "sm_clock_min_mhz", "power_max_w",
                   "gpu_util_pre", "foreign_mem_mib"])
    sampler = ClockSampler()

    for op in ops:
        shape = shapes.get({"fa": "attn", "gdn": "gdn", "fa_decode": "attn", "gdn_decode": "gdn",
                            "moe_mixed": "moe"}.get(op, op))
        if shape is None:
            print(f"skip {op}: model has no such layers"); continue
        # dense work and GDN decode do not depend on t; for *_decode ops c is the decode batch
        # moe_mixed: the t loop runs over decode batch sizes (--decode-batch), stored in column t
        for t in (ts if op in ("fa", "gdn", "fa_decode") else DECODE_B if op == "moe_mixed" else [0]):
            for c in cs:
                torch.cuda.empty_cache()
                try:
                    if op == "fa":
                        fn = make_fa(shape, c, t, dev, dtype, a.fa_version)
                    elif op == "gdn":
                        fn = make_gdn(shape, c, t, dev, dtype, a.gdn_state)
                    elif op == "fa_decode":
                        fn = make_fa_decode(shape, c, t, dev, dtype, a.fa_version)
                    elif op == "gdn_decode":
                        fn = make_gdn_decode(shape, c, dev, dtype)
                    elif op == "moe_mixed":
                        fn = make_moe_mixed(shape, t, c, dev, dtype)
                    else:
                        fn = make_dense(op, shape, c, dev, dtype)
                    torch.cuda.synchronize()
                    util_pre, foreign = foreign_load()
                    torch.cuda.reset_peak_memory_stats()
                    eager = timed(fn, a.warmup, a.iters)
                    peak_a = torch.cuda.max_memory_allocated() / 2**20
                    peak_r = torch.cuda.max_memory_reserved() / 2**20
                    h_ms = host_ms(fn)
                    with sampler:
                        if op == "gdn_conv":
                            samples, k_graph, flush_ms, method = kernel_time(fn, a.warmup, a.iters), 0, 0.0, "kernel_sum"
                        else:
                            (samples, k_graph, flush_ms), method = gpu_time(fn, a.warmup, a.iters), "graph"
                    clk_mean, clk_min, pw_max = sampler.summary()
                except torch.cuda.OutOfMemoryError:
                    print(f"{op} c={c} t={t}: OOM, skipped"); continue
                for mode, ts_ in (("graph", samples), ("eager", eager)):
                    for i, ms in enumerate(ts_):
                        raw.writerow([op, c, t, mode, i, f"{ms:.5f}"])
                s = stats(samples)
                summ.writerow([op, shape["layers"], c, t, f"{s['p50']:.5f}", f"{s['p99']:.5f}",
                               f"{s['min']:.5f}", f"{s['max']:.5f}", s["n"], f"{s['p50'] * 1000 / c:.4f}",
                               f"{stats(eager)['p50']:.5f}", f"{h_ms:.5f}", k_graph, f"{flush_ms:.5f}", method, MOE_TOUCHED[0] if op in ("moe", "moe_mixed") else "",
                               f"{peak_a:.1f}", f"{peak_r:.1f}", f"{clk_mean:.0f}", clk_min, f"{pw_max:.0f}",
                               util_pre, f"{foreign:.0f}"])
                summ_f.flush()
                print(f"{op:3s} c={c:5d} t={t:6d}  gpu p50={s['p50']:.4f} ms  p99={s['p99']:.4f}  "
                      f"{s['p50'] * 1000 / c:.3f} us/tok  eager={stats(eager)['p50']:.4f}  host={h_ms:.4f}  "
                      f"peak={peak_a:.0f} MiB  clk={clk_mean:.0f}/{clk_min} MHz  {pw_max:.0f} W"
                      + (f"  WARNING other load: util={util_pre}% mem={foreign:.0f} MiB" if util_pre > 5 or foreign > 2048 else ""))
                del fn

    meta["gpu_after"] = gpu_state()
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {d.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
