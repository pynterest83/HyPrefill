#!/usr/bin/env python3
"""Gate G1c on the real model (plan/02 Việc 1): does a larger prefill chunk cost KV-cache capacity
on Qwen3.8-Flash-Next? vLLM sizes the KV cache after a profiling run at max_num_batched_tokens,
so any activation / indexer workspace that grows with the chunk shows up as fewer KV tokens.

One engine start per chunk size (max_num_batched_tokens = c), same max_model_len and memory
utilization; the KV capacity is read from vLLM's startup log ("GPU KV cache size", "Available KV
cache memory"), then one long prefill is run as a smoke test.

  CUDA_VISIBLE_DEVICES=6,7 taskset -c 72-95 python bench/g1c_kv_capacity.py --tp 2
Output: results/step02/<date>_g1c_kv_capacity/{config.json, summary.csv}, logs in ~/hyprefill_data/step02/
"""
import argparse, csv, datetime, glob, json, os, pathlib, re, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CHILD = r'''
import os, sys, time
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], tensor_parallel_size=int(sys.argv[2]), max_model_len=int(sys.argv[3]),
          max_num_batched_tokens=int(sys.argv[4]), gpu_memory_utilization=float(sys.argv[5]),
          enable_prefix_caching=False, max_num_seqs=64)
import random
random.seed(0)
n = int(sys.argv[6])
t0 = time.perf_counter()
llm.generate([{"prompt_token_ids": [random.randrange(1000, 100000) for _ in range(n)]}],
             SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
print(f"SMOKE prefill {n} tokens in {time.perf_counter() - t0:.2f} s", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--c", default="2048,8192,32768")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--smoke-tokens", type=int, default=131072)
    a = ap.parse_args()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
    model = a.model or sorted(glob.glob(f"{hf}/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/*/"))[0]
    out = REPO / "results/step02" / f"{datetime.date.today()}_g1c_kv_capacity"
    i = 2
    while out.exists():
        out = out.with_name(f"{datetime.date.today()}_g1c_kv_capacity_run{i}"); i += 1
    out.mkdir(parents=True)
    logs = pathlib.Path.home() / "hyprefill_data/step02" / out.name
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in map(int, a.c.split(",")):
        log = logs / f"c{c}.log"
        with open(log, "w") as fh:
            rc = subprocess.run([sys.executable, "-c", CHILD, model, str(a.tp), str(a.max_model_len), str(c),
                                 str(a.gpu_mem), str(a.smoke_tokens)], stdout=fh, stderr=subprocess.STDOUT).returncode
        txt = log.read_text()
        num = lambda pat: (lambda m: float(m.group(1).replace(",", "")) if m else None)(re.search(pat, txt))
        rows.append(dict(c=c, returncode=rc,
                         kv_cache_tokens=num(r"GPU KV cache size:\s*([\d,]+) tokens"),
                         available_kv_gib=num(r"Available KV cache memory:\s*([\d.]+) GiB"),
                         max_concurrency=num(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x"),
                         smoke_prefill_s=num(r"SMOKE prefill \d+ tokens in ([\d.]+) s")))
        print(rows[-1], flush=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import vllm
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    (out / "config.json").write_text(json.dumps(dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"), model=model, tp=a.tp, max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem, smoke_tokens=a.smoke_tokens, vllm=vllm.__version__,
        indexer_logits_cap_mb=os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "default (512)"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), repo_commit=git("rev-parse", "HEAD"),
        repo_dirty=bool(git("status", "--porcelain")), cmd=" ".join(sys.argv), logs=str(logs)), indent=2))
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    main()
