#!/usr/bin/env python3
"""plan/01 §2.5 for Qwen3-Next TP2 (MoE + GDN + all-reduce): do the per-op cost tables add up to a
real vLLM prefill? Never checked before 2026-10-09; KT2 rests on these tables.

Per chunk budget c (one engine, max_num_batched_tokens = c, prefix caching off so chunks are exactly
c) and context t (multiple of c): one prompt of t + c tokens, max_tokens = 1.
  wall: blocking LLM.generate, median of --repeats requests, no profiler
  GPU:  one more request with vLLM's CUDA profiler on, run under nsys with
        --capture-range=cudaProfilerApi --capture-range-end=repeat; GPU time = union of the kernel
        intervals on rank 0's GPU (streams overlap, so a plain sum would double count)
Predicted GPU per step at context k*c, per layer type (per GPU, cold-L2 tables):
  12 x [fa(c, kc) + dense_attn(c) + 2 ar(c)] + 36 x [gdn(c) + gdn_conv(c) + dense_gdn(c) + 2 ar(c)]
  + 48 x moe(c)        moe: KT1 moe_mixed (real routing) at the smallest decode batch, or the
                       random-router table (--moe random)
Run it under nsys (the script re-launches itself):
  CUDA_VISIBLE_DEVICES=6,7 taskset -c 72-95 python bench/validate_forward_moe.py
Output: results/step01/<date>_validate_forward_Qwen3-Next-80B-A3B-Instruct_tp2/{config.json, summary.csv}
"""
import argparse, datetime, glob, json, os, pathlib, sqlite3, statistics as st, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL_TAG = "Qwen3-Next-80B-A3B-Instruct_tp2"


def engine_run(a):
    """Child: runs under nsys. One engine per c; wall timings to stdout as JSON lines."""
    import numpy as np
    from vllm import LLM, SamplingParams
    rng = np.random.default_rng(0)
    for c in map(int, a.c.split(",")):
        ts = [t for t in map(int, a.t.split(",")) if t % c == 0]
        # capture CUDA graphs up to c: a step with more tokens than the largest captured size runs fully
        # eager (~1860 kernel launches, ~85-90 ms of CPU per step on Qwen3-Next TP2; the first run of this
        # script used max_num_seqs=16, which capped the captured sizes at 32 and made every step eager)
        sizes = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, c})
        llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_num_batched_tokens=c, max_model_len=max(ts) + c + 64,
                  enable_prefix_caching=False, profiler_config={"profiler": "cuda"},
                  compilation_config={"cudagraph_capture_sizes": sizes})
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        p = lambda n: [{"prompt_token_ids": rng.integers(1000, 100000, n).tolist()}]
        for t in ts:
            llm.generate(p(t + c), sp, use_tqdm=False)  # warmup
            walls = []
            for _ in range(a.repeats):
                q = p(t + c); t0 = time.perf_counter(); llm.generate(q, sp, use_tqdm=False)
                walls.append((time.perf_counter() - t0) * 1e3)
            llm.start_profile(); llm.generate(p(t + c), sp, use_tqdm=False); llm.stop_profile()
            print("ROW " + json.dumps(dict(c=c, t=t, wall_ms=st.median(walls), wall_min_ms=min(walls), wall_max_ms=max(walls))), flush=True)
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
        time.sleep(5)


def gpu_union_ms(rep):
    """Union of kernel intervals on the lowest device id of an nsys report (rank 0), and step count."""
    db = rep.with_suffix(".sqlite")
    if not db.exists():
        subprocess.run(["nsys", "export", "-t", "sqlite", "-o", str(db), str(rep)], check=True, capture_output=True)
    con = sqlite3.connect(db)
    dev = con.execute("select min(deviceId) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    iv = sorted(con.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (dev,)).fetchall())
    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            iv += con.execute(f"select start, end from {tbl} where deviceId = ?", (dev,)).fetchall()
        except sqlite3.OperationalError:
            pass
    iv.sort()
    union, steps, (cs, ce) = 0, 1, iv[0]
    for s, e in iv[1:]:
        if s > ce:
            union += ce - cs
            steps += (s - ce) > 2_000_000  # > 2 ms idle: next engine step
            cs, ce = s, e
        else:
            ce = max(ce, e)
    union += ce - cs
    return union / 1e6, steps, (iv[-1][1] - iv[0][0]) / 1e6


def predicted_ms(c, t, moe_src):
    import numpy as np, pandas as pd
    sys.path.insert(0, str(REPO / "bench"))
    from oracle import Costs, layers
    tab = pd.concat([pd.read_csv(REPO / f"results/step0{i}/cost_tables/{MODEL_TAG}.csv") for i in (1, 2)]).drop_duplicates(["op", "c", "t"])
    cost = Costs(tab)
    ar = pd.read_csv(glob.glob(str(REPO / "results/step01/*_allreduce_Qwen3-Next-80B-A3B-Instruct_tp2/summary.csv"))[0])
    allreduce = lambda n: float(np.interp(n, ar.n, ar.p50_ms))
    cfg = json.load(open(REPO / "results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json")); cfg = cfg.get("text_config", cfg)
    n_attn, n_gdn, n = layers(cfg)
    if moe_src == "random":
        moe = lambda m: float(cost("moe", m))
    else:  # KT1 moe_mixed with real routing at the smallest decode batch (prefill + bd decode tokens)
        k = pd.concat([pd.read_csv(f) for f in glob.glob(str(REPO / "results/step02/*_kt1_draw*/summary.csv"))])
        k = k[(k.op == "moe_mixed") & (k.t == k.t.min())].groupby("c").p50_ms.mean()
        moe = lambda m: float(np.interp(m, k.index, k.values))
    g = lambda op, m, tt=None: float(cost(op, m, tt))
    total = 0.0
    for j in range((t + c) // c):
        ctx = j * c
        total += (n_attn * (g("fa", c, ctx) + g("dense_attn", c) + 2 * allreduce(c))
                  + n_gdn * (g("gdn", c) + g("gdn_conv", c) + g("dense_gdn", c) + 2 * allreduce(c)) + n * moe(c))
    return total, (t + c) // c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--c", default="512,2048,8192")
    ap.add_argument("--t", default="0,16384,65536")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--analyze", default=None, help="re-analyze an existing run directory")
    a = ap.parse_args()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
    a.model = a.model or sorted(glob.glob(f"{hf}/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/*/"))[0]
    if a.child:
        return engine_run(a)
    name = f"{datetime.date.today()}_validate_forward_{MODEL_TAG}"
    out = REPO / "results/step01" / name
    i = 2
    while out.exists():
        out = out.with_name(f"{name}_run{i}"); i += 1
    raw = pathlib.Path.home() / "hyprefill_data/step01" / out.name
    raw.mkdir(parents=True, exist_ok=True)
    cmd = ["nsys", "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace", "node",
           "--capture-range=cudaProfilerApi", "--capture-range-end=repeat", "-o", str(raw / "prof"), "--force-overwrite", "true",
           sys.executable, "-u", __file__, "--child", "--model", a.model, "--tp", str(a.tp), "--c", a.c, "--t", a.t,
           "--repeats", str(a.repeats)]
    log = raw / "child.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    rows = [json.loads(l[4:]) for l in log.read_text().splitlines() if l.startswith("ROW ")]
    reps = sorted(raw.glob("prof*.nsys-rep"), key=lambda p: p.stat().st_mtime)
    assert len(reps) == len(rows), f"{len(reps)} nsys reports for {len(rows)} rows (rc {rc}), see {log}"
    import csv
    out.mkdir(parents=True)
    res = []
    for r, rep in zip(rows, reps):
        gpu, steps_seen, span = gpu_union_ms(rep)
        pred_real, steps = predicted_ms(r["c"], r["t"], "kt1")
        pred_rand, _ = predicted_ms(r["c"], r["t"], "random")
        res.append(dict(**r, steps=steps, gpu_steps_seen=steps_seen, gpu_union_ms=gpu, gpu_span_ms=span,
                        pred_gpu_ms=pred_real, pred_gpu_random_moe_ms=pred_rand,
                        pred_err_pct=100 * (pred_real / gpu - 1), gpu_frac_of_wall=gpu / r["wall_ms"],
                        cpu_idle_per_step_ms=(r["wall_ms"] - gpu) / steps))
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    import vllm
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    (out / "config.json").write_text(json.dumps(dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"), model=a.model, tp=a.tp, c=a.c, t=a.t, repeats=a.repeats,
        vllm=vllm.__version__, prefix_caching=False, lock_mhz=os.environ.get("LOCK_MHZ"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), repo_commit=git("rev-parse", "HEAD"),
        repo_dirty=bool(git("status", "--porcelain")), cmd=" ".join(sys.argv), raw=str(raw),
        note="GPU = union of kernel/memcpy/memset intervals on rank 0 (nsys, one profiled request); wall = median of "
             "unprofiled requests; prediction from step 1/2 tables + all-reduce + KT1 moe_mixed"), indent=2))
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    main()
