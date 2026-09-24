# Danh sách bài đọc HyPrefill — link PDF để in

Thứ tự là thứ tự đọc. Mỗi bài ghi 10 dòng: vấn đề, cơ chế, kết quả, baseline, điểm yếu, khác HyPrefill ở đâu.

## Nhóm 1 — Đọc kỹ, bắt buộc (7 bài, ~110 trang)

| # | Bài | PDF | Tìm gì khi đọc |
|---|---|---|---|
| 1 | Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference (OSDI 2024) | https://arxiv.org/pdf/2403.02310 | Định nghĩa stall-free batching; cách chọn chunk theo TBT; hình TTFT–TBT Pareto. Baseline số 1. |
| 2 | From Tokens to Layers: Layered Prefill (MLSys 2026 Oral) | https://arxiv.org/pdf/2510.08055 | Vì sao chunk theo token gây đọc lại expert 39%; công thức nhóm layer G(L); vì sao chỉ test attention+MoE. Tiền bối trực tiếp. |
| 3 | SLOWeave: Deadline-Aware Adaptive Prefill Chunking (09/2026) | https://arxiv.org/pdf/2609.07883 | Cost model "monotone iteration-cost"; tìm nhị phân chunk theo deadline. Baseline adaptive mạnh nhất, đồng nhất theo layer. |
| 4 | COREY: Entropy-Guided Runtime Chunk Scheduling for Selective Scan Kernels (04/2026) | https://arxiv.org/pdf/2604.10597 | Overhead ước lượng 0.7–4.6% và vì sao "best static chunk beats all adaptive". Phản ví dụ phải né. |
| 5 | Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR 2025) | https://arxiv.org/pdf/2412.06464 | Công thức cập nhật state; vì sao state cố định; thuật toán chunkwise. |
| 6 | Gated Linear Attention Transformers with Hardware-Efficient Training (ICML 2024) | https://arxiv.org/pdf/2312.06635 | Thuật toán chunkwise của Flash Linear Attention: phần intra-chunk là matmul, phần inter-chunk truyền state; chunk kernel 64. Đọc §4. |
| 7 | Marconi: Prefix Caching for the Era of Hybrid LLMs (MLSys 2025) | https://arxiv.org/pdf/2411.19379 | Cách một paper systems về hybrid đặt vấn đề và đánh giá; state SSM không cắt được ở prefix bất kỳ. |

## Nhóm 2 — Đọc để hiểu sparse attention (2 bài + 2 model card)

| # | Bài | PDF | Tìm gì |
|---|---|---|---|
| 8 | DeepSeek-V3.2: Pushing the Frontier of Open LLMs (DSA, lightning indexer) | https://arxiv.org/pdf/2512.02556 | Indexer chấm điểm thế nào, key indexer 128 chiều FP8, top-k; phần nào còn tăng theo t. |
| 9 | DistServe: Disaggregating Prefill and Decoding (OSDI 2024) | https://arxiv.org/pdf/2401.09670 | Lập luận interference giữa hai pha; vì sao PD disaggregation; giới hạn khi chuyển KV. |
| 10 | Model card Qwen3.8-Flash-Next (GDN + QSA + MoE) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Layer pattern 12×(3 GDN + 1 QSA); QSA micro-block, budget 512 block / 2048 token. |
| 11 | Model card Qwen3-Next-80B-A3B | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | Layer pattern 12×(3 GDN + 1 gated attention); 512 expert, 10+1 active. |

## Nhóm 3 — Đọc lướt, 3 dòng mỗi bài (họ làm gì, khác HyPrefill ở đâu)

| # | Bài | PDF |
|---|---|---|
| 12 | Medha: Efficiently Serving Multi-Million Context Length LLM Inference | https://arxiv.org/pdf/2409.17264 |
| 13 | Niyama: Breaking the Silos of LLM Inference Serving | https://arxiv.org/pdf/2503.22562 |
| 14 | POD-Attention: Unlocking Full Prefill-Decode Overlap (ASPLOS 2025) | https://arxiv.org/pdf/2410.18038 |
| 15 | DUET: Disaggregated Hybrid Mamba-Transformer LLMs with Prefill/Decode-Specific Packages | https://arxiv.org/pdf/2603.15530 |
| 16 | FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity (operator-level preemption) | https://arxiv.org/pdf/2602.16603 |
| 17 | PrefillOnly (đọc §4.2 "hybrid prefilling": chunk FFN, attention nguyên khối) | https://arxiv.org/pdf/2505.07203 |
| 18 | Not All Prefills Are Equal: PPD disaggregation cho multi-turn (append-prefill trên node decode) | https://arxiv.org/pdf/2603.13358 |
| 19 | Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput | https://arxiv.org/pdf/2508.01989 |
| 20 | Jenga: Effective Memory Management for Serving LLM with Heterogeneity | https://arxiv.org/pdf/2503.18292 |
| 21 | Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer Inference | https://arxiv.org/pdf/2605.22416 |

## Không in, đọc trên màn hình (issue/PR/blog)

- vLLM Hybrid KV Cache Manager design doc — https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html
- vLLM issue #54775 (buffer chunked-scan GDN/KDA scale theo max-num-batched-tokens) — https://github.com/vllm-project/vllm/issues/54775
- vLLM issue #56457 (buffer logits QSA indexer phình theo context, OOM) — https://github.com/vllm-project/vllm/issues/56457
- vLLM RFC #55524 (Mamba2 chunk size phải giống nhau giữa prefill và decode) — https://github.com/vllm-project/vllm/issues/55524
- vLLM PR #54076 (align chunk theo block Mamba) — https://github.com/vllm-project/vllm/pull/54076
- vLLM PR #49827 (mixed decode+prefill cho Qwen GDN, FLA_CHUNK_SIZE=64) — https://github.com/vllm-project/vllm/pull/49827
- SGLang issue #39342 (mixed-chunk hỏng mamba radix cache) — https://github.com/sgl-project/sglang/issues/39342
- Blog vLLM Qwen3-Next — https://vllm.ai/blog/2025-09-11-qwen3-next
- Code Layered Prefill — https://github.com/scale-snu/layered-prefill

## Tải hết PDF nhóm 1–3 bằng một lệnh

```bash
mkdir -p papers && cd papers
for id in 2403.02310 2510.08055 2609.07883 2604.10597 2412.06464 2312.06635 2411.19379 \
          2512.02556 2401.09670 \
          2409.17264 2503.22562 2410.18038 2603.15530 2602.16603 2505.07203 2603.13358 2508.01989 2503.18292 2605.22416; do
  curl -sL -o "$id.pdf" "https://arxiv.org/pdf/$id"; sleep 3
done
ls -la
```
