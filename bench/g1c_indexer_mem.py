#!/usr/bin/env python3
"""Gate G1c (plan/02 Việc 1): does the QSA indexer of Qwen3.8-Flash-Next allocate a c x t
logits buffer per call during prefill, or a bounded workspace?

Calls vLLM 0.30's qsa_select_paged_prefill directly on synthetic inputs with the model's
indexer shapes (config: 4 heads, head_dim 128, compress ratio 4, budget 2048; fp8 e4m3 Q and
compressed-K cache as the model runs them), one request of c new tokens at context t.
Records the peak memory added by the call (max_memory_allocated / reserved above the inputs)
and its GPU time. Reading the code says the logits are chunked over queries to stay under
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (default 512); run with a smaller cap as well to see it.

  CUDA_VISIBLE_DEVICES=7 python bench/g1c_indexer_mem.py
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 CUDA_VISIBLE_DEVICES=7 python bench/g1c_indexer_mem.py --tag cap64
"""
import argparse, datetime, json, os, pathlib, subprocess, sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/step00/configs/Qwen_Qwen3.8-Flash-Next.json")
    ap.add_argument("--c", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--t", default="32768,131072,262144")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = json.load(open(REPO / a.config)); cfg = cfg.get("text_config", cfg)
    H, D, r, topk = cfg["indexer_n_heads"], cfg["indexer_head_dim"], cfg["indexer_compress_ratio"], cfg["indexer_budget"]
    from vllm import envs
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import qsa_select_paged_prefill
    cap_mb = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    dev, f8 = "cuda:0", torch.float8_e4m3fn
    rows = []
    for t in [int(x) for x in a.t.split(",")]:
        for c in [int(x) for x in a.c.split(",")]:
            torch.cuda.empty_cache()
            L = t + c
            n_rows = -(-L // r)
            n_pages = -(-n_rows // a.page_size)
            q = torch.randn(c, H, D, device=dev).to(f8)
            k_cache = torch.randn(n_pages, a.page_size, 1, D, device=dev).to(f8)
            page_table = torch.arange(n_pages, device=dev, dtype=torch.int32).unsqueeze(0)
            qsl = torch.tensor([0, c], device=dev, dtype=torch.int32)
            pos = torch.arange(t, t + c, device=dev, dtype=torch.int32)
            visible = torch.div(pos + 1, r, rounding_mode="floor").to(torch.int32)  # causal: rows ≤ own position
            out = torch.empty(c, topk // r, device=dev, dtype=torch.int32)
            call = lambda: qsa_select_paged_prefill(q, k_cache, page_table, qsl, visible, topk, r, c, out, L)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            call(); torch.cuda.synchronize()
            peak_a = torch.cuda.max_memory_allocated() - base_a
            peak_r = torch.cuda.max_memory_reserved() - base_r
            ts = []
            for _ in range(a.iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); call(); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            ts.sort()
            naive = c * (-(-n_rows // 64) * 64) * 4  # one fp32 logits buffer for all c rows
            rows.append(dict(c=c, t=t, peak_alloc_mib=peak_a / 2**20, peak_reserved_mib=peak_r / 2**20,
                             naive_ct_buffer_mib=naive / 2**20, p50_ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1]))
            print(f"t={t:6d} c={c:5d}: peak +{peak_a / 2**20:8.1f} MiB alloc (+{peak_r / 2**20:7.1f} reserved), "
                  f"naive c*t buffer {naive / 2**20:8.1f} MiB, time p50 {ts[len(ts) // 2]:.3f} ms", flush=True)
            del q, k_cache, out
    out_dir = REPO / "results/step02" / f"{datetime.date.today()}_g1c_indexer_mem{('_' + a.tag) if a.tag else ''}"
    i = 2
    while out_dir.exists():
        out_dir = out_dir.with_name(out_dir.name + f"_run{i}"); i += 1
    out_dir.mkdir(parents=True)
    import csv
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import vllm
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out_dir / "config.json").write_text(json.dumps(dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"), H=H, D=D, compress_ratio=r, topk=topk,
        page_size=a.page_size, logits_cap_mb=cap_mb, vllm=vllm.__version__, torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), repo_commit=commit, cmd=" ".join(sys.argv),
        timing="CUDA events around one eager call (host loop over query chunks inside); memory is the point here"),
        indent=2))
    print(f"wrote {out_dir.relative_to(REPO)}/  (logits cap {cap_mb} MB)")


if __name__ == "__main__":
    main()
