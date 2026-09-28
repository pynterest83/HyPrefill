#!/usr/bin/env python3
"""Quick sanity check of the HyPrefill environment on the GPU server.

Prints library versions, GPUs, and runs two tiny timed kernels
(flash-attention and Gated DeltaNet) so you know week 1 will work.
"""
import os, sys, time

def ok(msg):  print(f"  OK    {msg}")
def bad(msg): print(f"  FAIL  {msg}")

print("Python", sys.version.split()[0], "| HF_HOME =", os.environ.get("HF_HOME", "(unset — defaults to ~/.cache)"))

try:
    import torch
    ok(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"        [{i}] {p.name}  {p.total_memory/2**30:.0f} GiB  SMs={p.multi_processor_count}")
except Exception as e:
    bad(f"torch: {e}"); sys.exit(1)

def timed(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

dev = "cuda:0"
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 1024, 16, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 8192, 16, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ms = timed(lambda: flash_attn_func(q, k, v, causal=False))
    ok(f"flash_attn: q=1024, kv=8192 -> {ms:.3f} ms")
except Exception as e:
    bad(f"flash_attn: {e}")

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, H, D = 1, 2048, 16, 128
    q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16), dim=-1)
    v = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
    h0 = torch.zeros(B, H, D, D, device=dev, dtype=torch.float32)
    ms = timed(lambda: chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0, output_final_state=True))
    ok(f"fla chunk_gated_delta_rule: T=2048 with initial_state -> {ms:.3f} ms")
except Exception as e:
    bad(f"fla gated_delta_rule: {e}  (check the argument order for your fla version)")

try:
    import transformers; ok(f"transformers {transformers.__version__}")
except Exception as e:
    bad(f"transformers: {e}")
