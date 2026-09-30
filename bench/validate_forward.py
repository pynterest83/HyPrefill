#!/usr/bin/env python3
"""plan/01 §2.5: does the step 1 cost table add up to a real vLLM prefill?

Qwen3.8-27B (no MoE, TP1) is the only model whose whole layer stack is in the step 1 tables;
Qwen3-Next needs the MoE costs of step 2 first.

For each chunk budget c (one vLLM engine per c, max_num_batched_tokens = c) and context t,
prefill one prompt of t + c random tokens with max_tokens = 1. vLLM's chunked prefill splits
it into ceil((t + c) / c) steps of c tokens, step k at context k*c. The measured prefill time
is the whole request latency, median of repeats (--baseline none, default). The first version
subtracted the latency of a 16-token prompt as "fixed per-request work", but that prompt also
runs a full forward that reads all weights, the main GEMM cost at small c, so the t = 0 rows
came out +9..+104% (2026-09-29). The 16-token latency is still recorded (base_ms);
--baseline short restores the old subtraction for comparison. Predicted: for every step, per layer
  16 x [FA(c, k*c) + dense_attn(c)] + 48 x [GDN(c) + dense_gdn(c) + gdn_conv(c)] + 64 x dense_mlp(c)
from results/step01/cost_tables (GPU kernel time, cold L2), FA interpolated linearly in t.
Request latency is wall clock around a blocking LLM.generate, so it includes the CPU work of
every step; the gap to the prediction is what a kernel-only cost model misses.

  CUDA_VISIBLE_DEVICES=6 taskset -c 72-95 python bench/validate_forward.py

With --decode-batch 8,32,64 (plan/01 §2.5 item 5) each (c, t, bd) also runs with bd requests
decoding while the measured prompt prefills, as on a busy node. bd requests with --decode-ctx
token prompts are prefilled first; then decode-only steps give the measured decode step D, the
prompt is added, and every engine step until it finishes is timed (host perf_counter around
LLMEngine.step(), no profiler). Each such step is predicted two ways (bench/oracle.py):
  additive: D(bd) + prefill(c)          (decode and prefill costs summed)
  mixed:    decode FA/GDN kernels + prefill FA/GDN kernels + GEMM/MoE layers at bd + c tokens
Needs the step 2 decode tables (fa_decode, gdn_decode, dense ops at batch sizes) for the model.
"""
import argparse, datetime, json, os, pathlib, statistics as st, subprocess, sys, time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent


def predicted_ms(table, cfg, c, t):
    tb = table
    n_attn = sum(x == "full_attention" for x in cfg["layer_types"])
    n_gdn = len(cfg["layer_types"]) - n_attn
    n = len(cfg["layer_types"])

    def at(op, cc, tt=0):
        r = tb[(tb.op == op) & (tb.c == cc)]
        if op != "fa":
            return float(r[r.t == 0].p50_ms.iloc[0])
        r = r.sort_values("t")
        return float(np.interp(tt, r.t, r.p50_ms))

    total, steps = 0.0, []
    L = t + c
    for k in range(-(-L // c)):
        ctx = k * c
        cc = min(c, L - ctx)
        if cc != c:
            raise ValueError("t must be a multiple of c")
        s = (n_attn * (at("fa", c, ctx) + at("dense_attn", c))
             + n_gdn * (at("gdn", c) + at("dense_gdn", c) + at("gdn_conv", c))
             + n * at("dense_mlp", c))
        steps.append(s)
        total += s
    return total, steps


def mixed_models(model_tag, cfg, repo=REPO):
    """(additive, mixed, decode_only) per-step predictors from step 1 + step 2 tables."""
    sys.path.insert(0, str(REPO / "bench"))
    from oracle import Costs, layers
    tabs = [repo / f"results/step01/cost_tables/{model_tag}.csv", repo / f"results/step02/cost_tables/{model_tag}.csv"]
    cost = Costs(pd.concat([pd.read_csv(x) for x in tabs if x.exists()]).drop_duplicates(["op", "c", "t"]))
    n_attn, n_gdn, n = layers(cfg)
    ffn = "moe" if "moe" in set(cost.t.op) else "dense_mlp"
    g = lambda op, m, t=None: float(cost(op, m, t))
    dec_ex = lambda bd, td: n_attn * g("fa_decode", bd, td) + (n_gdn * g("gdn_decode", bd) if n_gdn else 0.0)
    shared = lambda m: n_attn * g("dense_attn", m) + (n_gdn * g("dense_gdn", m) if n_gdn else 0.0) + n * g(ffn, m)
    pre_ex = lambda c, t: n_attn * g("fa", c, t) + (n_gdn * (g("gdn", c) + g("gdn_conv", c)) if n_gdn else 0.0)
    additive = lambda bd, td, c, t: dec_ex(bd, td) + shared(bd) + pre_ex(c, t) + shared(c)
    mixed = lambda bd, td, c, t: dec_ex(bd, td) + pre_ex(c, t) + shared(bd + c)
    decode_only = lambda bd, td: dec_ex(bd, td) + shared(bd)
    return additive, mixed, decode_only


def run_with_decode(llm, rng, c, t, bd, decode_ctx, repeats):
    """Engine-level timing of one (t + c)-token prefill while bd requests decode."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    eng = llm.llm_engine
    runs = []
    for rep in range(repeats + 1):
        dec_ids = [f"d{c}_{t}_{bd}_{rep}_{j}" for j in range(bd)]
        long_sp = SamplingParams(max_tokens=100000, ignore_eos=True, temperature=0.0)
        for rid in dec_ids:
            eng.add_request(rid, TokensPrompt(prompt_token_ids=rng.integers(1000, 100000, decode_ctx).tolist()), long_sp)
        for _ in range(-(-bd * decode_ctx // c) + 4):   # prefill the decode requests
            eng.step()
        dwalls = []
        for _ in range(8):                               # decode-only steps
            t0 = time.perf_counter(); eng.step(); dwalls.append((time.perf_counter() - t0) * 1e3)
        pid = f"p{c}_{t}_{bd}_{rep}"
        eng.add_request(pid, TokensPrompt(prompt_token_ids=rng.integers(1000, 100000, t + c).tolist()),
                        SamplingParams(max_tokens=1, temperature=0.0))
        walls, done = [], False
        while not done:
            t0 = time.perf_counter(); outs = eng.step(); walls.append((time.perf_counter() - t0) * 1e3)
            done = any(o.request_id == pid and o.finished for o in outs)
        eng.abort_request(dec_ids)
        while eng.has_unfinished_requests():
            eng.step()
        if rep:  # first repetition is warmup
            runs.append((st.median(dwalls), walls))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default="results/step00/configs/Qwen_Qwen3.8-27B.json")
    ap.add_argument("--table", default="results/step01/cost_tables/Qwen3.8-27B_tp1.csv")
    ap.add_argument("--c", default="512,2048,8192")
    ap.add_argument("--t", default="0,16384,65536")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--baseline", choices=["none", "short"], default="none")
    ap.add_argument("--decode-batch", default="", help="e.g. 8,32,64: also measure with concurrent decode")
    ap.add_argument("--decode-ctx", type=int, default=4096, help="prompt length of the decoding requests")
    ap.add_argument("--model-tag", default="Qwen3.8-27B_tp1", help="cost table name for the decode-batch predictors")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    bds = [int(x) for x in a.decode_batch.split(",") if x]

    hf = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / "hf_cache"))
    model = a.model or str(next((hf / "hub/models--Qwen--Qwen3.8-27B/snapshots").iterdir()))
    cfg = json.load(open(REPO / a.config))
    cfg = cfg.get("text_config", cfg)
    table = pd.read_csv(REPO / a.table)
    cs = [int(x) for x in a.c.split(",")]
    ts = [int(x) for x in a.t.split(",")]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = np.random.default_rng(0)

    out = REPO / "results" / "step01" / f"{datetime.date.today()}_validate_forward"
    i = 2
    while out.exists():
        out = out.with_name(f"{datetime.date.today()}_validate_forward_run{i}"); i += 1
    out.mkdir(parents=True)
    rows = []
    for c in cs:
        llm = LLM(model=model, tensor_parallel_size=a.tp, max_num_batched_tokens=c,
                  max_model_len=max(ts) + max(cs) + 64, enable_prefix_caching=False, gpu_memory_utilization=0.85,
                  max_num_seqs=max([8] + [b + 2 for b in bds]), seed=0)
        prompt = lambda n: TokensPrompt(prompt_token_ids=rng.integers(1000, 100000, n).tolist())

        def lat(n):
            ms = []
            for rep in range(a.repeats + 2):
                p = prompt(n)
                t0 = time.perf_counter()
                llm.generate([p], sp, use_tqdm=False)
                if rep >= 2:
                    ms.append((time.perf_counter() - t0) * 1e3)
            return st.median(ms), min(ms), max(ms)

        base, _, _ = lat(16)
        for t in ts:
            med, lo, hi = lat(t + c)
            meas = med - base if a.baseline == "short" else med
            pred, steps = predicted_ms(table, cfg, c, t)
            rows.append(dict(c=c, t=t, steps=len(steps), measured_ms=meas, request_ms=med,
                             request_min_ms=lo, request_max_ms=hi, base_ms=base, predicted_ms=pred,
                             error_pct=100 * (pred - meas) / meas))
            print(f"c={c:5d} t={t:6d} steps={len(steps):3d}  measured {meas:9.2f} ms  "
                  f"predicted {pred:9.2f} ms  error {rows[-1]['error_pct']:+6.1f}%", flush=True)
            for bd in bds:
                additive, mixed, decode_only = mixed_models(a.model_tag, cfg)
                runs = run_with_decode(llm, rng, c, t, bd, a.decode_ctx, a.repeats)
                d_meas = st.median(r[0] for r in runs)
                p_meas = st.median(sum(r[1]) for r in runs)
                n_st = len(runs[0][1])
                ctxs = [min(k * c, t) for k in range(n_st)]
                p_add = sum(additive(bd, a.decode_ctx, c, x) for x in ctxs)
                p_mix = sum(mixed(bd, a.decode_ctx, c, x) for x in ctxs)
                rows.append(dict(c=c, t=t, decode_batch=bd, steps=n_st, measured_ms=p_meas,
                                 decode_step_measured_ms=d_meas, decode_step_predicted_ms=decode_only(bd, a.decode_ctx),
                                 predicted_additive_ms=p_add, predicted_mixed_ms=p_mix,
                                 error_additive_pct=100 * (p_add - p_meas) / p_meas,
                                 error_mixed_pct=100 * (p_mix - p_meas) / p_meas))
                r = rows[-1]
                print(f"  bd={bd:3d}: {n_st} steps, measured {p_meas:9.2f} ms | additive {p_add:9.2f} "
                      f"({r['error_additive_pct']:+6.1f}%) | mixed {p_mix:9.2f} ({r['error_mixed_pct']:+6.1f}%) | "
                      f"decode step {d_meas:6.2f} measured vs {r['decode_step_predicted_ms']:6.2f} predicted", flush=True)
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.4f")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,clocks.sm,power.limit", "--format=csv,noheader",
                          "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")], capture_output=True, text=True).stdout
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import vllm, torch
    (out / "config.json").write_text(json.dumps(dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"), model=model, table=a.table,
        cs=cs, ts=ts, repeats=a.repeats, baseline=a.baseline, decode_batch=bds, decode_ctx=a.decode_ctx, model_tag=a.model_tag, tp=a.tp, gpu=gpu.strip(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        vllm=vllm.__version__, torch=torch.__version__, repo_commit=commit, cmd=" ".join(sys.argv),
        clock_locked=True, note=f"clock lock verified by the caller (idle SM clock pinned at {os.environ.get('LOCK_MHZ', '1980')} MHz)"), indent=2))
    print(f"wrote {out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
