#!/usr/bin/env python3
"""Does FA3 over vLLM's paged KV cost more than the contiguous KV of the step 1 tables?

plan/01 §2.5 left a 5-10% gap at c >= 2048 between the cost table and a real vLLM prefill.
vLLM reads KV through a block table; for Qwen3.8-27B it sets the attention block size to 784
tokens (to match the mamba page size). Time prefill attention (q = c, kv = t + c) both ways
with the same cold-L2 CUDA-graph method as bench/op_cost.py.

  CUDA_VISIBLE_DEVICES=7 python bench/fa_paged_check.py --block 784
"""
import argparse, json, pathlib, sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import op_cost as oc  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/step00/configs/Qwen_Qwen3.8-27B.json")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--block", type=int, default=784)
    ap.add_argument("--c", default="512,2048,8192")
    ap.add_argument("--t", default="16384,65536")
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    sh = oc.load_shapes(REPO / a.config, a.tp)["attn"]
    h, hk, d = sh["heads"], sh["kv_heads"], sh["head_dim"]
    dev, dt = "cuda:0", torch.bfloat16
    for t in [int(x) for x in a.t.split(",")]:
        for c in [int(x) for x in a.c.split(",")]:
            L = t + c
            nb = -(-L // a.block)
            q = torch.randn(c, h, d, device=dev, dtype=dt)
            kc = torch.randn(nb, a.block, hk, d, device=dev, dtype=dt)
            vc = torch.randn_like(kc)
            bt = torch.randperm(nb, device=dev, dtype=torch.int32).unsqueeze(0)  # scattered blocks, as after churn
            cu_q = torch.tensor([0, c], device=dev, dtype=torch.int32)
            used = torch.tensor([L], device=dev, dtype=torch.int32)
            paged = lambda: flash_attn_varlen_func(q, kc, vc, max_seqlen_q=c, cu_seqlens_q=cu_q, max_seqlen_k=L,
                                                   seqused_k=used, block_table=bt, causal=True, fa_version=3)
            contig = oc.make_fa(sh, c, t, dev, dt, 3)
            p = oc.stats(oc.gpu_time(paged, 10, 30)[0])["p50"]
            s = oc.stats(oc.gpu_time(contig, 10, 30)[0])["p50"]
            print(f"t={t:6d} c={c:5d}: contiguous {s:8.4f} ms  paged(block {a.block}) {p:8.4f} ms  ({100 * (p - s) / s:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
