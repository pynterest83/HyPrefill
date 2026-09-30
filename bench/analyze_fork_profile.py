#!/usr/bin/env python3
"""Kernel time by category per ModelRunner.run call in the fork's profiled traces, chunked vs
layered (bench/step00_layered_profile.sh). The question: on H200, how much of a step is MoE
(expert weight reads), the cost Layered Prefill is designed to cut?"""
import glob, gzip, json, os, re, sys
from collections import Counter

CATS = [("moe", r"moe|expert|topk_|fused_moe|align_block"), ("attention", r"flash|attn|FlashAttn|flash::"),
        ("gemm", r"gemm|nvjet|sm90_xmma|cutlass|cublas|Kernel2"), ("allreduce", r"allreduce|all_reduce|nccl|cross_device"),
        ("norm", r"rms_norm|rmsnorm|layernorm|layer_norm"), ("other", r".")]
import csv
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*/"))):
    kern = sorted(glob.glob(os.path.join(d, "kern*_cuda_gpu_kern_sum.csv")))
    if kern:  # nsys: kernel summary over the capture window (both ranks) + NVTX ranges
        by = Counter()
        rows = list(csv.DictReader(open(kern[0])))
        tcol = next(c for c in rows[0] if c.startswith("Total Time"))
        for r in rows:
            by[next(c for c, p in CATS if re.search(p, r["Name"]))] += float(r[tcol])
        tot = sum(by.values())
        nv = sorted(glob.glob(os.path.join(d, "nvtx*_nvtx_sum.csv")))
        runs = ""
        if nv:
            for r in csv.DictReader(open(nv[0])):
                if "ModelRunner::run" in r.get("Range", "") and "model" not in r.get("Range", "").lower():
                    runs = f"ModelRunner::run: {r.get('Instances')} calls, median {float(r.get('Med (ns)', 0)) / 1e6:.2f} ms"
        print(f"{os.path.basename(d.rstrip('/')):8s} (nsys, both ranks): GPU kernel time {tot / 1e9:.2f} s | "
              + " ".join(f"{c} {100 * v / tot:.0f}%" for c, v in by.most_common()) + (f" | {runs}" if runs else ""))
        top = sorted(((float(r[tcol]), r["Name"][:70]) for r in rows if re.search(CATS[0][1], r["Name"])), reverse=True)[:3]
        print("   top MoE kernels (s):", [(n, round(t / 1e9, 2)) for t, n in top])
        continue
    f = sorted(glob.glob(os.path.join(d, "*.json")))
    if not f:
        print(os.path.basename(d.rstrip("/")), "no trace"); continue
    ev = json.load(open(f[0]))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    by = Counter()
    for e in k:
        by[next(c for c, p in CATS if re.search(p, e["name"]))] += e["dur"]
    tot = sum(by.values())
    calls = int(re.search(r"calls(\d+)-(\d+)", f[0]).group(2)) - int(re.search(r"calls(\d+)-", f[0]).group(1))
    span = (max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k)) / 1e3
    print(f"{os.path.basename(d.rstrip('/')):8s}: {calls} calls, GPU busy {tot / 1e3:.0f} ms of {span:.0f} ms span "
          f"({100 * tot / 1e3 / span:.0f}%), per call {tot / 1e3 / calls:.2f} ms | " +
          " ".join(f"{c} {100 * v / tot:.0f}%" for c, v in by.most_common()))
    top = Counter()
    for e in k:
        if re.search(CATS[0][1], e["name"]):
            top[e["name"][:70]] += e["dur"]
    print("   top MoE kernels:", [(n, round(v / 1e3, 1)) for n, v in top.most_common(3)])
