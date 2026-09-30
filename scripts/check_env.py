#!/usr/bin/env python3
"""Quick sanity check of the HyPrefill environment on the GPU server.

Prints library versions, GPUs, and runs two tiny timed kernels -- the ones vLLM
serves with on H200: FA3 (vllm_flash_attn) and FlashInfer GDN -- so you know
step 1 will work.
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
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    q = torch.randn(1024, 16, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(8192, 8, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1024], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, 8192], device=dev, dtype=torch.int32)
    ms = timed(lambda: flash_attn_varlen_func(q, k, v, max_seqlen_q=1024, cu_seqlens_q=cu_q,
                                              max_seqlen_k=8192, cu_seqlens_k=cu_k,
                                              causal=True, fa_version=3))
    ok(f"vllm_flash_attn FA3: q=1024, kv=8192 -> {ms:.3f} ms")
except Exception as e:
    bad(f"vllm_flash_attn FA3: {e}")

try:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule
    T, HK, HV, D = 2048, 16, 32, 128
    q = torch.randn(1, T, HK, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, HK, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, D, device=dev, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, HV, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, HV, device=dev, dtype=torch.bfloat16).sigmoid()
    h0 = torch.zeros(1, HV, D, D, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    ms = timed(lambda: fi_chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
                                                 output_final_state=True, cu_seqlens=cu))
    ok(f"FlashInfer GDN (vLLM wrapper): T=2048 with initial_state -> {ms:.3f} ms")
except Exception as e:
    bad(f"FlashInfer GDN: {e}")

try:
    import vllm, flashinfer
    ok(f"vllm {vllm.__version__}, flashinfer {flashinfer.__version__}")
except Exception as e:
    bad(f"vllm/flashinfer: {e}")

try:
    import transformers; ok(f"transformers {transformers.__version__}")
except Exception as e:
    bad(f"transformers: {e}")
