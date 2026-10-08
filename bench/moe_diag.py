#!/usr/bin/env python3
"""Diagnose what the MoE cost of a mixed batch is made of (KT1 follow-up, 2026-10-08).

KT1 found inc(D, 2048) ~ 0.45 x 4 inc(D, 512) with real routing, while a 2048 chunk touches
only ~20% more experts than a 512 chunk. This sweeps the fused MoE kernel with a controlled
routing that decouples the two drivers of its cost:
  E_t  distinct experts touched (weight bytes read: E_t x 3 x inter x hidden x 2 B per GPU)
  T    tokens, each routed to top-k of those E_t experts, spread evenly (T k / E_t per expert)
Only `fused_experts` is timed (no router GEMM, no shared expert), cold L2, CUDA graph method
of op_cost.py. A fit time = a + b E_t + c T k (+ per-expert tile term) shows how much of the
KT1 increment is weight reads versus per-token work.

  CUDA_VISIBLE_DEVICES=4 taskset -c 48-71 python bench/moe_diag.py --tp 2 --tag moe_diag
Output: results/step02/<date>_<tag>/{config.json, summary.csv}
"""
import argparse, csv, datetime, json, os, pathlib, shlex, subprocess, sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cgroup_cpu  # noqa: E402
from op_cost import (ClockSampler, gpu_state, gpu_time, load_shapes, stats, versions)  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def make(shape, n_exp, tokens, dev, dtype, seed=0):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    E, k, inter, h = shape["experts"], shape["topk"], shape["inter"], shape["hidden"]
    assert k <= n_exp <= E
    g = torch.Generator(device="cpu").manual_seed(seed)
    chosen = torch.randperm(E, generator=g)[:n_exp]
    # token i takes k consecutive slots of a round-robin over the chosen experts: every expert
    # gets T k / E_t tokens (+-1) and the k experts of one token are distinct
    slots = (torch.arange(tokens)[:, None] * k + torch.arange(k)[None, :]) % n_exp
    ti = chosen[slots].to(dev, torch.int32)
    tw = torch.softmax(torch.randn(tokens, k, device=dev), -1)
    x = torch.randn(tokens, h, device=dev, dtype=dtype)
    w1 = torch.randn(E, 2 * inter, h, device=dev, dtype=dtype) * 0.02
    w2 = torch.randn(E, h, inter, device=dev, dtype=dtype) * 0.02
    return lambda: fused_experts(x, w1, w2, tw, ti), int(torch.unique(ti).numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--experts", default="16,32,64,128,192,256,384,512")
    ap.add_argument("--tokens", default="16,32,64,128,256,320,384,448,512,640,768,1024,1536,2048,3072,4096,8192")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--clock-locked", action="store_true")
    ap.add_argument("--tag", default="moe_diag")
    a = ap.parse_args()
    assert a.warmup >= 10 and a.iters >= 30, "docs/03_MEASUREMENT.md: warmup >= 10, iters >= 30"
    dev, dtype = "cuda:0", torch.bfloat16
    shape = load_shapes(a.config, a.tp)["moe"]
    torch.zeros(1, device=dev)
    before = gpu_state()
    busy = [g for g in before if int(g["utilization.gpu"]) > 5 or int(g["memory.used"]) * 10 > int(g["memory.total"])]
    if busy:
        sys.exit(f"GPU busy: {busy}")

    base = REPO / "results" / "step02"
    d, i = base / f"{datetime.date.today()}_{a.tag}", 2
    while True:  # never overwrite
        try:
            d.mkdir(parents=True); break
        except FileExistsError:
            d = base / f"{datetime.date.today()}_{a.tag}_run{i}"; i += 1
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    bytes_per_expert = 3 * shape["inter"] * shape["hidden"] * 2
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "model_config": a.config, "tp": a.tp,
            "shape": shape, "bytes_per_expert": bytes_per_expert, "gpu": before,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **versions(),
            "repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "--porcelain")),
            "cmd": " ".join(shlex.quote(x) for x in sys.argv), "clock_locked": a.clock_locked,
            "warmup": a.warmup, "iters": a.iters, "timing": "graph [flush L2, fused_experts] x k minus flush"}
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    f = open(d / "summary.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["experts_touched", "tokens", "tokens_per_expert", "p50_ms", "p99_ms", "min_ms", "max_ms", "n",
                "weight_mib", "eff_weight_gbps", "sm_clock_mean_mhz", "sm_clock_min_mhz", "power_max_w"])
    sampler = ClockSampler()
    cg0 = cgroup_cpu.snapshot()
    for e in map(int, a.experts.split(",")):
        for t in map(int, a.tokens.split(",")):
            if e < shape["topk"]:
                continue
            torch.cuda.empty_cache()
            fn, e_real = make(shape, e, t, dev, dtype)  # < e when t k < e
            with sampler:
                samples, _, _ = gpu_time(fn, a.warmup, a.iters)
            clk_mean, clk_min, pw = sampler.summary()
            s = stats(samples)
            mib = e_real * bytes_per_expert / 2**20
            w.writerow([e_real, t, f"{t * shape['topk'] / e_real:.1f}", f"{s['p50']:.5f}", f"{s['p99']:.5f}", f"{s['min']:.5f}",
                        f"{s['max']:.5f}", s["n"], f"{mib:.1f}", f"{mib * 2**20 / (s['p50'] * 1e-3) / 1e9:.0f}",
                        f"{clk_mean:.0f}", clk_min, f"{pw:.0f}"])
            f.flush()
            print(f"E_t={e_real:3d} T={t:5d}  {s['p50']:.4f} ms  weights {mib:6.0f} MiB  clk {clk_mean:.0f}/{clk_min}", flush=True)
            del fn
    meta["cgroup_cpu"] = cgroup_cpu.delta(cg0); cgroup_cpu.warn(meta["cgroup_cpu"])
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()
