#!/usr/bin/env python3
"""Why does the step 1 cost table under-predict a real vLLM prefill (plan/01 §2.5)?

Profile one chunked prefill of t + c tokens (max_num_batched_tokens = c) in vLLM with the
torch profiler, then from the GPU trace report: kernel time by category (to compare with
the cost-table ops), GPU busy time vs wall span (idle gaps = host-bound), and time per step.

  CUDA_VISIBLE_DEVICES=6 taskset -c 72-95 python bench/profile_vllm_step.py --c 512 --t 16384
  CUDA_VISIBLE_DEVICES=6 taskset -c 72-95 python bench/profile_vllm_step.py --sweep 512,1024,2048,4096,8192 --t 16384,65536

The torch profiler slows the host side (GPU busy ~43% of the step under the profiler vs ~66%
without, c = 512, 2026-09-29), so each configuration is measured twice in the same engine:
(1) wall time of the whole request with no profiler (blocking LLM.generate, median of
--repeats), and (2) total GPU kernel time of one profiled request. The engine core runs in its
own process in vLLM V1, so LLMEngine.step() does not map to scheduler steps (a first version
timed step() and saw one "step" per request); the number of steps is known instead,
ceil((t + c) / c). host_overhead_per_step = (wall - GPU busy) / steps is the per-iteration
CPU/idle cost the kernel-only cost model misses (plan/01 §2.5), with vLLM's default settings
(async scheduling included). --sweep writes one CSV row per
(c, t) to results/step01/<date>_vllm_step_sweep_<tag>/.
"""
import argparse, collections, datetime, glob, gzip, json, os, pathlib, re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
CATS = [  # first match wins; names as they appear in the vLLM 0.30 / H200 traces
    ("fa3", r"flash_fwd|FlashAttn|flash::"),
    ("gdn_flashinfer", r"gdn|delta_r|GatedDelta"),
    ("gdn_triton", r"chunk_gated_delta|chunk_fwd_kernel|merge_16x16|recompute_w_u|chunk_scaled_dot"),
    ("l2norm", r"l2norm"),
    ("conv1d", r"conv1d|causal_conv"),
    ("gemm", r"gemm|nvjet|sm90_xmma|cutlass.*(Gemm|gemm)|cublas|Kernel2"),
    ("rmsnorm", r"rms_norm|rmsnorm|RMSNorm|layer_norm|_layer_norm_fwd"),
    ("rotary", r"rotary|rope"),
    ("act", r"silu|act_and_mul|gelu"),
    ("copy/cast", r"copy|Copy|cast|elementwise|Memcpy|memcpy|fill|Fill"),
]


def categorize(name):
    for cat, pat in CATS:
        if re.search(pat, name):
            return cat
    return "other"


def analyze(trace_file):
    ev = json.load(gzip.open(trace_file) if trace_file.endswith(".gz") else open(trace_file))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    k.sort(key=lambda e: e["ts"])
    by_cat, by_name = collections.Counter(), collections.Counter()
    for e in k:
        by_cat[categorize(e["name"])] += e["dur"]
        by_name[e["name"][:90]] += e["dur"]
    busy = sum(e["dur"] for e in k)  # sum over kernels: counts overlapping streams twice
    union, cs, ce = 0, k[0]["ts"], k[0]["ts"] + k[0]["dur"]  # GPU busy as the union of kernel intervals
    for e in k[1:]:
        if e["ts"] > ce:
            union += ce - cs; cs, ce = e["ts"], e["ts"] + e["dur"]
        else:
            ce = max(ce, e["ts"] + e["dur"])
    union += ce - cs
    span = k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]
    # steps: split where the GPU is idle for > 2 ms (between engine steps)
    steps, cur, last_end = [], [k[0]], k[0]["ts"] + k[0]["dur"]
    for e in k[1:]:
        if e["ts"] - last_end > 2000:
            steps.append(cur); cur = []
        cur.append(e); last_end = max(last_end, e["ts"] + e["dur"])
    steps.append(cur)
    step_rows = [(s[0]["ts"], (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1e3, sum(x["dur"] for x in s) / 1e3) for s in steps]
    return dict(busy_ms=union / 1e3, kernel_sum_ms=busy / 1e3, span_ms=span / 1e3, n_kernels=len(k),
                by_cat_ms={c: v / 1e3 for c, v in by_cat.most_common()},
                top_kernels_ms=[(n, v / 1e3) for n, v in by_name.most_common(15)],
                steps=[dict(span_ms=sp, busy_ms=b) for _, sp, b in step_rows])


def step_walls(llm, prompt, sp, rid):
    """Wall time of each engine step for one request, no profiler."""
    import time
    eng = llm.llm_engine
    eng.add_request(rid, prompt, sp)
    walls = []
    while eng.has_unfinished_requests():
        t0 = time.perf_counter()
        eng.step()
        walls.append((time.perf_counter() - t0) * 1e3)
    return walls


def sweep(a):
    """One engine per c (max_num_batched_tokens is an engine setting); per t: wall per step
    without profiler (median of --repeats requests), GPU busy per step from one profiled run."""
    import csv, subprocess
    import cgroup_cpu; cg0 = cgroup_cpu.snapshot()
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm.inputs import TokensPrompt
    hf = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / "hf_cache"))
    model = a.model or str(next((hf / "hub/models--Qwen--Qwen3.8-27B/snapshots").iterdir()))
    tag = a.tag or pathlib.Path(model).parts[-3].replace("models--", "")
    base = REPO / "results/step01"; base.mkdir(parents=True, exist_ok=True)
    name, i = f"{datetime.date.today()}_vllm_step_sweep_{tag}", 2
    d = base / name
    while True:
        try:
            d.mkdir(); break
        except FileExistsError:
            d = base / f"{name}_run{i}"; i += 1
    raw = pathlib.Path.home() / "hyprefill_data/step01" / d.name
    raw.mkdir(parents=True, exist_ok=True)
    ts = [int(x) for x in a.t_list.split(",")]
    rng = np.random.default_rng(0)
    rows = []
    for c in [int(x) for x in a.sweep.split(",")]:
        prof = raw / f"prof_c{c}"
        llm = LLM(model=model, tensor_parallel_size=a.tp, max_num_batched_tokens=c, max_model_len=max(ts) + c + 64,
                  enable_prefix_caching=False, gpu_memory_utilization=0.85, max_num_seqs=8, seed=0,
                  profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=str(prof),
                                                 torch_profiler_with_stack=False))
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        for t in ts:
            import time
            p = lambda: TokensPrompt(prompt_token_ids=rng.integers(1000, 100000, t + c).tolist())
            llm.generate([p()], sp, use_tqdm=False)  # warmup
            walls, cg = [], cgroup_cpu.snapshot()
            for _ in range(a.repeats):
                q = p(); t0 = time.perf_counter(); llm.generate([q], sp, use_tqdm=False)
                walls.append((time.perf_counter() - t0) * 1e3)
            cgd = cgroup_cpu.delta(cg); cgroup_cpu.warn(cgd, f"c={c} t={t}")
            before = set(glob.glob(str(prof / "**" / "*.json*"), recursive=True))
            llm.start_profile(); llm.generate([p()], sp, use_tqdm=False); llm.stop_profile()
            new = sorted(set(glob.glob(str(prof / "**" / "*.json*"), recursive=True)) - before)
            worker = [f for f in new if "rank" in f or "worker" in f.lower() or "tp0" in f] or new
            res = analyze(worker[0]) if worker else None
            n_steps = -(-(t + c) // c)
            wall_total = float(np.median(walls))
            busy_total = res["busy_ms"] if res else float("nan")
            if busy_total > float(np.median(walls)):  # Qwen3-Next TP2, 2026-10-09: the profiled run is slower than the timed ones
                print(f"WARNING c={c} t={t}: GPU busy {busy_total:.0f} ms > wall {np.median(walls):.0f} ms; "
                      "the profiled request is not representative, do not use this row", flush=True)
            wall_step, busy_step = wall_total / n_steps, busy_total / n_steps
            rows.append(dict(c=c, t=t, steps=n_steps, wall_total_ms=wall_total, wall_min_ms=min(walls),
                             wall_max_ms=max(walls), gpu_busy_total_ms=busy_total,
                             wall_step_ms=wall_step, gpu_busy_step_ms=busy_step,
                             host_overhead_step_ms=wall_step - busy_step,
                             gpu_busy_frac=busy_step / wall_step if wall_step else float("nan"),
                             cpu_throttled_s=cgd.get("throttled_s", float("nan"))))
            r = rows[-1]
            print(f"c={c:5d} t={t:6d}: {n_steps} steps, wall/step {wall_step:7.2f} ms, GPU busy/step {busy_step:7.2f} ms "
                  f"-> host overhead {r['host_overhead_step_ms']:6.2f} ms ({100 * r['gpu_busy_frac']:.0f}% busy)", flush=True)
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,power.limit", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    json.dump({"model": model, "tp": a.tp, "sweep_c": a.sweep, "t": a.t_list, "repeats": a.repeats,
               "lock_mhz": os.environ.get("LOCK_MHZ"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpus": gpus, "cgroup_cpu": cgroup_cpu.delta(cg0), "note": "wall = blocking LLM.generate of one request without profiler (median of repeats); "
               "busy = sum of GPU kernel durations of one profiled request; per step = / ceil((t + c) / c)"}, open(d / "config.json", "w"), indent=2)
    print(f"wrote {d.relative_to(REPO)}/ (traces in {raw})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=512)
    ap.add_argument("--t", type=int, default=16384)
    ap.add_argument("--model", default=None)
    ap.add_argument("--analyze", default=None, help="only analyze an existing trace file")
    ap.add_argument("--sweep", default=None, help="comma list of c: per-step wall vs GPU busy sweep")
    ap.add_argument("--t-list", default="0,16384,65536", help="contexts for --sweep")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.analyze), indent=1)); return
    if a.sweep:
        sweep(a); return

    hf = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / "hf_cache"))
    model = a.model or str(next((hf / "hub/models--Qwen--Qwen3.8-27B/snapshots").iterdir()))
    out = pathlib.Path.home() / "hyprefill_data" / "step01" / f"{datetime.date.today()}_vllm_profile_c{a.c}_t{a.t}"
    out.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm.inputs import TokensPrompt
    llm = LLM(model=model, max_num_batched_tokens=a.c, max_model_len=a.t + a.c + 64, enable_prefix_caching=False,
              gpu_memory_utilization=0.85, max_num_seqs=8, seed=0,
              profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=str(out),
                                             torch_profiler_with_stack=False))
    rng = np.random.default_rng(0)
    p = lambda: TokensPrompt(prompt_token_ids=rng.integers(1000, 100000, a.t + a.c).tolist())
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    for _ in range(2):
        llm.generate([p()], sp, use_tqdm=False)
    llm.start_profile()
    llm.generate([p()], sp, use_tqdm=False)
    llm.stop_profile()
    del llm
    traces = sorted(glob.glob(str(out / "**" / "*.json*"), recursive=True))
    worker = [f for f in traces if "rank" in f or "worker" in f.lower() or "tp0" in f] or traces
    print("traces:", traces)
    res = analyze(worker[0])
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "steps"}, indent=1))
    st = res["steps"]
    print(f"steps: {len(st)}; per step span median {np.median([s['span_ms'] for s in st]):.2f} ms, "
          f"busy median {np.median([s['busy_ms'] for s in st]):.2f} ms")


if __name__ == "__main__":
    main()
