# Related Work — số liệu đã xác minh

Mọi con số dưới đây lấy trực tiếp từ bản HTML của paper trên arXiv hoặc từ PR/issue gốc, kiểm tra ngày 17–18/09/2026. Đây là nguyên liệu cho phần Related Work của bài; giữ nguyên số khi trích.

---

## 1. Chunked prefill và scheduling

### Sarathi-Serve — arXiv 2403.02310, OSDI 2024
- Generation stall: Falcon-180B, prompt 4K token mất ~1150 ms, iteration decode batch 32 mất ~200 ms → bubble tới ~950 ms.
- Chỉ ~512 token đã bão hoà compute GPU; prompt thực tế median 1730–7059 token.
- Token budget chọn bằng profiling offline: "maximum number of tokens that can be packed in a batch without violating TBT SLO".
- **Tile quantization**: chunk 257 chậm hơn chunk 256 tới 32%.
- Capacity dưới SLO strict: Mistral-7B 2.6×, Yi-34B 3.7×, LLaMA2-70B 4.3×, Falcon-180B 3.6× so với vLLM; so với Orca tới 6.3×.
- Ablation Yi-34B: hybrid batching only 0.53 s TTFT / 0.68 s P99 TBT; chunked prefill only 1.04 / 0.17; cả hai 0.76 / 0.14.
- Overhead chunking: tối đa ~25% ở chunk 512, gần bằng 0 ở chunk 2048.
- **Giới hạn**: một token budget toàn cục, tune offline; cost model giả định mọi layer giống nhau; không SSM/GDN/MoE.

### Layered Prefill — arXiv 2510.08055, MLSys 2026 Oral
- "Sparsity erosion": "chunking inflates expert coverage while suppressing reuse".
- Bảng 1 (Qwen3-30B-A3B, ShareGPT): expert coverage < 50% ở batch 16, < 70% ở batch 64, **86.3% ở batch 128**.
- Bảng 7: 100 request arXiv (prompt trung bình 9194 token), traffic đọc expert **35.6 TB → 21.7 TB (−39%)**. ShareGPT (2340 token) chỉ **−12%** (28.5 → 25.1 TB).
- Bảng 2: chunk 512 → 2048 giảm expert load 955 → 304 GB/request (−68%) và energy 60.2 → 32.4 mJ/token (−46%), nhưng **P99 TBT 48.4 → 129 ms**.
- Công thức nhóm layer: `N_lg(L) = max(1, ⌈L/512⌉)`. Mỗi iteration, đúng một nhóm layer chạy prefill + decode, các nhóm khác chỉ decode.
- Bảng 6 (Qwen/arXiv, 1.3 req/s): mean TTFT **2.80 → 1.24 s (−56%)**, P99 TTFT −53%, mean TBT −35%, P99 TBT −27%. Energy/token −20…−22%.
- Bảng 9: GPT-OSS-120B trên 2×H100 TTFT −65%; Qwen3-235B trên 8×H100 −47%.
- Bảng 11: N_lg = 2 cho TTFT thấp nhất nhưng P99 TBT 138 ms; N_lg = 16 cho P99 TBT 35.7 ms nhưng TTFT 1.27 s.
- **Ablation quyết định**: trên model dense Qwen3-8B, layered prefill **thua** chunked prefill (TTFT 1.45 s so với 0.955 s).
- **Giới hạn**: chỉ MoE + full attention (Qwen3-30B-A3B, GPT-OSS-20B trên 2×H100). **Không một chữ nào về SSM, Mamba, GDN, linear attention.** Quản lý activation trung gian giữa các nhóm layer không đặc tả. N_lg(L) tĩnh.
- Code: `github.com/scale-snu/layered-prefill` (nanovllm-based, MIT, ~19 sao). **Không có PR/issue nào trong vLLM hay SGLang nhắc đến nó.**

### SLOWeave — arXiv 2609.07883 (09/2026)
- `C* = C*(n, L, D, T, λ)`: chunk tối ưu phụ thuộc batch size, độ dài prompt, target TPOT, cost function, arrival rate. Chunk cố định "strands slack in every window".
- Cost model chỉ cần **đơn điệu theo c**: "can be a lookup table..., a fitted regressor, or a conservative analytical model. SLOWeave requires only monotonicity in c".
- Mô hình synthetic mặc định: `T(n,c) = 0.35 + 𝟙[n>0]·(0.90+0.055n) + 𝟙[c>0]·(0.40+0.006c)` ms.
- `B_t = max(0, min_{i∈A_t}(d_i − s_t))`; `c*_t = max{c ≤ min(C_max, L_t) : T(|A_t|, c) ≤ B_t}`, binary search O(log C_max).
- Mệnh đề 1: chunk chọn được vừa an toàn deadline vừa lớn nhất trong lớp an toàn. Với cost model học, kiểm tra `T̂ + δ ≤ B_t`.
- 8×A100 và 8×H100, model dense 8B và 70B. A100/8B mixed: P99 TTFT 1780 → 1190 ms, goodput 38.4 → 67.1 req/s. H100/70B-TP8 long-context: P99 TTFT 5940 → 3960 ms, goodput 22.1 → 40.8.
- TPOT 25 ms: SLO attainment mixed 79.4% (SLOWeave) so với 59.4% (fixed-1024), full prefill 6.9%. TPOT 10 ms: cải thiện tới 3.35×. TPOT 50 ms: gần bằng full prefill.
- **Giới hạn**: một chunk `c*_t` cho toàn bộ model, không có khái niệm chunk theo layer hay operator. Không MoE, không SSM, không hybrid. Không mô hình preemption hay KV transfer.

### COREY — arXiv 2604.10597 (04/2026) — KẾT QUẢ ÂM
- Chunk 512 thay vì 64 nhanh hơn **4.41×** trên RTX 3070, 3.90–4.04× trên datacenter accelerator.
- `Ĥ(Z) = −Σ p_k log(p_k + ε)`; `H̃ = Ĥ / log K` với `H_ref = log K` (K = 256 → 5.55 nats).
- `C = clip(2^round(log₂(C_min + r·(C_max − C_min))), 32, 512)`.
- **Bảng 5, end-to-end trên H800, prompt 976 token**: static-512 (oracle) **891.51 ± 10.17 ms**; full-histogram COREY 970.26 ± 27.36 ms (**chậm hơn 8.8%**); sampled 932.69 (+4.6%); guarded 903.03 (+1.3%); learned seq-len table 897.63 (+0.7%).
- Kết luận của tác giả: **"no entropy-guided variant beats the best static chunk"**.
- Bảng 6–7: static-512 toàn cục 317.1 ms so với per-regime oracle 316.7 ms — oracle chỉ hơn 0.14%.
- Bảng 14: 80 prompt LongBench có entropy tập trung hẹp 3.87–4.14 nats, tất cả rơi vào cùng bucket chunk 256.
- Overhead: đo entropy +8.3% nếu instrument 48 layer, ≈2.1% nếu sample mỗi 4 layer; +2.3% trên datacenter. Mỗi lần gọi scheduler 1.10 ± 0.16 ms.
- **Nguyên nhân thất bại**: không phải chọn sai chunk (decision quality cao) mà **decision cost** — dựng histogram, synchronization boundary, dispatch overhead. Phải trả cả khi kết quả trùng static-512.
- **Bài học cho HyPrefill**: tín hiệu điều khiển phải là cấu trúc + SLO, không phải nội dung activation. Bắt buộc có bảng "decision quality vs decision cost".

### Medha — arXiv 2409.17264; Niyama — arXiv 2503.22562; SlidingServe — arXiv 2606.05933
Adaptive chunk size theo slack/deadline, vẫn một chunk toàn cục cho mọi layer.

### FlowPrefill — arXiv 2602.16603 (02/2026)
- "Operator-Level Preemption": đặt điểm preempt ở biên các operator (`qkv_proj, attn, o_proj, gate_up_proj, down_proj`) để ngắt prefill mà không cần chunk nhỏ.
- **Cùng số token qua mọi operator**, không buffer activation. Cost model là polynomial theo token count toàn request.
- Model: Llama3-8B, Qwen2.5-14B, Llama3-70B, Qwen3-30B-A3B. Không hybrid. Trên vLLM 0.11.2.
- Khác HyPrefill: giải head-of-line blocking (khi nào ngắt), không phải chunk-size heterogeneity (mỗi operator bao nhiêu token).

### PrefillOnly §4.2 — arXiv 2505.07203 (05/2025, preprint)
- "Hybrid prefilling": "we prefill the non-attention layers chunk-by-chunk and prefill the attention layers normally".
- **Analog cấu trúc gần nhất** với HyPrefill: granularity khác nhau theo operator trong cùng forward pass.
- Nhưng: mục tiêu peak memory (intermediate tensor lớn gấp 14× KV của một layer), workload prefill-only một token output, **không có decode**, không TBT SLO, dense transformer, không cost model chọn chunk.

---

## 2. Prefill/decode disaggregation

### DistServe — arXiv 2401.09670, OSDI 2024
- Model 13B, 1 GPU: colocated ~1.6 req/s ở 90% SLO; prefill-only 5.6; decode-only 10; disaggregated (2+1 GPU) trung bình 3.3 req/s/GPU = **2.1×**.
- Về chunked prefill: "alleviates the slowdown... but it does not eliminate it"; chia N chunk làm chi phí đọc lại KV cache thành **O(N²)**.
- Goodput = "maximum request rate that can be served adhering to the SLO attainment goal... for each GPU provisioned".
- Placement: liệt kê cấu hình song song → simulator (sai số < 2%) → binary search rate. Chạy < 1.3 phút.
- KV transfer pull-based. OPT-66B, 512 token, 10 req/s → 11.3 GB/s ≈ 90 Gbps; > 95% request truyền < 30 ms.
- 32 GPU A100-80GB. Chatbot OPT-13B 2.0–4.6× so với vLLM; OPT-175B **7.4×** so với DeepSpeed-MII; Summarization chịu SLO chặt hơn **12.6×**.
- vLLM++ (parallelism tối ưu) không cải thiện → vấn đề là colocated, không phải cấu hình song song.
- Tác giả thừa nhận: với ít GPU, "simpler architectural choices like non-disaggregated systems may reduce deployment complexity".

### Not All Prefills Are Equal (PPD) — arXiv 2603.13358, ICML 2026
Append-prefill của lượt sau chạy ngay trên node decode thay vì chuyển KV đi về; giảm 68% TTFT từ lượt hai. **Cơ sở cho phạm vi (2) của HyPrefill.**

### Prefill-Decode Aggregation or Disaggregation — arXiv 2508.01989
Mỗi chế độ thắng ở một vùng tải. Cơ sở cho phạm vi (1).

---

## 3. Linear attention và kernel

### Gated DeltaNet — arXiv 2412.06464, ICLR 2025
```
Linear attention:  S_t = S_{t−1} + k_t v_tᵀ,   o_t = S_tᵀ q_t
Mamba2:            S_t = α_t S_{t−1} + k_t v_tᵀ
DeltaNet:          S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ
Gated DeltaNet:    S_t = S_{t−1} · α_t (I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ
```
- WY representation: `T = [I + strictLower(diag(β) K Kᵀ)]⁻¹ diag(β) ∈ ℝ^{C×C}`, `W = TK`, `Ũ = TV`.
- Intra-chunk O(C²d), inter-chunk O(Cd²). State `d_k × d_v` mỗi head, **không đổi theo t**.
- 400M và 1.3B, 100B token. WikiText ppl GDN 16.42 (Mamba2 16.56, DeltaNet 17.71). Recall 2K: 30.6% (hybrid GDN-H2 40.1%). S-NIAH-3: 27.6% so với Mamba2 4.6%.
- Paper không tự công bố C hay FLOP kernel; dẫn chiếu Flash Linear Attention.

### Gated Linear Attention — arXiv 2312.06635, ICML 2024
- `S_t = Diag(α_t) S_{t−1} + k_tᵀ v_t`, α_t là vector cổng theo kênh; gate hạng thấp `α_t = σ(x_t W¹ W² + b)^{1/τ}`, W¹ ∈ ℝ^{d×16}, τ = 16.
- **FlashLinearAttention hai pha**: Pha 1 tuần tự trong SRAM tích luỹ S và ghi S[n] ra HBM; Pha 2 song song giữa các chunk. Đổi 10–20% bộ nhớ lấy song song cấp chuỗi.
- **C = 64**, bội của 16 để khớp tile tensor core. Đây là nguồn gốc của `FLA_CHUNK_SIZE=64`.
- Tổng chi phí O(LCd + Ld²) so với O(L²d) và O(Ld²).
- 340M và 1.3B, SlimPajama. WikiText 1.3B: GLA 17.22, Transformer++ 16.85, RetNet 18.64. Kernel trên H100 nhanh hơn FlashAttention-2 trên dải 512–32K.

### Marconi — arXiv 2411.19379, MLSys 2025
- "SSM states are updated in place, so a sequence's states cannot be rolled back to represent its prefixes".
- Block 32: 25.0% block KV được tái dùng nhưng chỉ **0.4%** state SSM — lệch **65.3×**. Model 7B, chuỗi 10K token tốn 17.4 GB lưu state, gấp 3.3× Transformer.
- Admission bằng speculative insertion (chỉ checkpoint tại điểm phân nhánh). Eviction `S(n) = recency(n) + α·flop_efficiency(n)`, α tự hiệu chỉnh bằng grid search sau bootstrap.
- Hybrid 7B và Jamba-1.5-Mini trên 4×A100-40GB; LMSys/ShareGPT/SWE-Bench. Hit rate hơn vLLM+ 4.5×/7.3×/**34.4×**; P95 TTFT −36.1%/−71.1%/−46.8%. So với SGLang+ hit rate hơn 19.0–219.7%.
- SWE-Bench theo độ dài: chuỗi < 7K thua tới 3.0%, chuỗi > 7K thắng tới 25.5%.
- **Giá trị cho HyPrefill**: mạch viết paper (định lượng pathology → vì sao one-size-fits-all thất bại → cơ chế tách theo loại operator → engine thật → baseline mạnh + phân tích theo bucket).

---

## 4. Sparse attention

### DeepSeek-V3.2 / DSA — arXiv 2512.02556
```
I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
u_t = Attn(h_t, { c_s : I_{t,s} ∈ Top-k(I_{t,:}) }),  k = 2048
```
- Paper nói rõ: **"the lightning indexer still has a complexity of O(L²)"**. Attention chính O(L²) → O(L·k).
- Indexer ít head, chiều nhỏ, FP8, ReLU thay softmax. `H^I` và `d^I` không công bố tường minh trong văn bản — lấy từ `config.json`.
- Huấn luyện hai giai đoạn: dense warm-up (KL, lr 1e-3, 1000 bước, 2.1B token) rồi sparse (lr 7.3e-6, 15000 bước, 943.7B token, tách indexer khỏi đồ thị chính).
- Suy ra cho chunked prefill: `cost_indexer(c,t) = O(c·t·H^I·d^I)` tăng theo t; `cost_sparse(c) = O(c·k·d)` không phụ thuộc t; `mem_indexer(c,t) = c·t` phần tử mỗi layer.

### vLLM issue #56457 — bằng chứng thực nghiệm cho ràng buộc bộ nhớ
> "chunk i of a long prompt now allocates 3200 × (800·i) × 4B = 10.24 MB × i, a strictly increasing size per chunk"

Allocator không tái dùng được block cũ, tích luỹ ~14 GB mỗi rank, OOM ở 254K token (treo tại 166,400 token). Vá bằng `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 → 64). **Khớp chính xác công thức `mem_indexer ∝ c·t`.**

### You Only Index Once — arXiv 2606.06467; Cross-Layer Attention — arXiv 2405.12981
Chia sẻ Top-K / KV qua layer, **static**. Nền cho CSA2 của DeepSeek V4.1.

---

### HySparse2 — arXiv 2609.26368 (22/09/2026), Xiaomi LLM-Core
Kiến trúc model, không phải paper lập lịch. **Không scoop HyPrefill.** Không có weights hay code công bố.

- Model 80B-A3B MoE, 49 layer, hidden 2048, MQA (64 query head / 1 KV head), head dim 256.
- Chia kiểu YOCO: **self-decoder** (~25 layer) dùng sliding-window attention cửa sổ 128 token cộng 5 layer full attention; **cross-decoder** dùng full attention cộng sparse attention (1024 token chọn ở mức token). KV của cross-decoder dựng từ hidden state của self-decoder ("KV Bridging").
- **"Prefill can therefore exit after the self-decoder, skipping all cross-decoder layers."** Trong PD disaggregation, node prefill chỉ cần ~25 layer đầu, gần nửa bộ nhớ.
- Ở 1M token: FLOP prefill giảm **2.92×** so với HySparse và **5.02×** so với Hybrid SWA. KV cache 2.69 GB so với 6.72 GB (HySparse) và 12.09 GB (Hybrid SWA), FP8.
- Không có gì về chunked prefill, lập lịch, TTFT/TBT hay engine.

**Dùng trong paper HyPrefill:**
1. *Intro / Motivation* — trích nguyên văn: "Across interaction rounds, a short generated action or tool call can return a much longer search result, execution trace, or document that requires prefill before decoding resumes." Một lab lớn thiết kế cả kiến trúc quanh đúng workload append-prefill.
2. *Limitations* — HyPrefill áp dụng cho hybrid kiểu stack thường. Ở kiến trúc thoát prefill sớm (YOCO như HySparse2, CED như DeepSeek V4.1), sparse attention nằm ở phần bị bỏ qua khi prefill, nên **chế độ indexer bị giới hạn bộ nhớ — nơi chunk theo operator có giá trị nhất — không xảy ra trong prefill**.
3. *Bối cảnh xu hướng* — lần thứ ba quan sát các lab giải chi phí prefill bằng kiến trúc thay vì lập lịch: sparse attention (DSA/QSA), CED (DeepSeek V4.1), YOCO (HySparse2).

## 5. Trạng thái engine (kiểm tra 14–18/09/2026)

| Bằng chứng | Link |
|---|---|
| vLLM: chunk size là một tham số toàn cục, không hướng dẫn theo model/layer type, không adaptive | docs.vllm.ai/en/latest/configuration/optimization.html |
| vLLM Hybrid KV Cache Manager: nhóm layer theo *KV cache group* để cấp phát **bộ nhớ**, không phải chunk size compute | docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html |
| vLLM PR #54076: chunk boundary phải align theo block size của Mamba group (1648 token) thay vì attention group (816) — vì **correctness**, không vì performance | github.com/vllm-project/vllm/pull/54076 |
| vLLM issue #54775: buffer chunked-scan GDN/KDA scale tuyến tính theo `max-num-batched-tokens`; chunk kernel hard-code 64; đề xuất "make chunk size configurable" | github.com/vllm-project/vllm/issues/54775 |
| vLLM RFC #55524: "The chunk size must be identical for prefill and decode" (Mamba2, bit-identical) | github.com/vllm-project/vllm/issues/55524 |
| vLLM PR #49827: mixed decode+prefill cho Qwen GDN, `FLA_CHUNK_SIZE=64`, +6.7% throughput, TP1 only | github.com/vllm-project/vllm/pull/49827 |
| SGLang issue #39342: `--enable-mixed-chunk` làm hỏng mamba radix cache checkpoint trên hybrid GDN (accuracy 0.91 → 0.85) | github.com/sgl-project/sglang/issues/39342 |
| SGLang issue #37904: PDMux deadlock khi mixed decode + split-prefill sau prompt ≥ 4K trên hybrid GDN | github.com/sgl-project/sglang/issues/37904 |
| SGLang Qwen3-Next cookbook: chỉ có `--max-mamba-cache-size`, `--mamba-full-memory-ratio`, radix cache V1/V2; không có gì về chunk size | docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next.md |
| vLLM blog FP8 KV cache: `--kv-cache-dtype-skip-layers sliding_window` — "keeping those layers in BF16 is actually faster than quantizing them" | vllm.ai/blog/2026-04-22-fp8-kvcache |
| GitHub search "layered prefill" / arXiv 2510.08055 trong vLLM và SGLang: **0 kết quả** | — |
| MLX / MLC-LLM: **chưa hỗ trợ** chunked prefill cho hybrid Mamba2 | — |

Điểm cuối cùng đáng chú ý: vLLM đã bắt đầu đối xử khác nhau **theo layer type** trên trục quantization (skip sliding-window layer khi lượng tử hoá KV). Dùng làm một câu trong Motivation: engine đã chấp nhận nguyên tắc "operator khác nhau cần chính sách khác nhau", chỉ chưa áp dụng cho chunk scheduling.

---

## 6. Ba câu hỏi reviewer chắc chắn hỏi

1. **So với Layered Prefill**: activation trung gian giữa các operator group quản lý ra sao (chính là chỗ họ bỏ ngỏ)? Và trên model chỉ có attention + MoE, HyPrefill có suy biến về Layered Prefill và có thắng nó trên chính benchmark của họ không?
2. **So với SLOWeave**: vì sao không chạy SLOWeave độc lập cho từng operator group? *Trả lời*: các group chia chung một deadline B_t và phụ thuộc nhau qua activation buffer, nên là bài toán tối ưu đa biến ràng buộc chung, không tách được thành N lần SLOWeave.
3. **So với COREY**: overhead của cơ chế chọn chunk theo group được đo và kiểm soát ra sao để không lặp lại kết quả âm? *Trả lời*: cost model offline, tra bảng, không ước lượng runtime; kèm bảng decision-cost.

---

## 6. Bổ sung 06/10/2026 (quét lại; mức xác minh thấp hơn các mục trên)

Các mục dưới đây đọc từ method section (CascadeEP), thân PR/issue, hoặc abstract. Chưa phải "số liệu đã xác minh" như các mục 1–5. Kiểm lại trước khi trích. Tổng hợp đầy đủ ở `docs/09_RESCAN_2026-10-06.md`.

### CascadeEP / AsyncEP — arXiv 2609.33252
- streamFFN gom token expert đã sẵn sàng tới một ngưỡng launch rồi mới chạy GEMM FFN. Đây là cùng ý khấu hao với nhóm MoE của HyPrefill.
- Động cơ: lệch tải attention giữa các replica data-parallel trong expert parallel đồng bộ. Không có decode TBT, không linear attention, attention không chạy chunk nhỏ, không so le giữa các iteration. Ngưỡng lấy từ điểm bão hoà throughput GEMM.
- **Phải trích và phân biệt.**

### SGLang PDMux — PR #42411 (tracking #41861), mở 03/10/2026
- "Layerwise prefill + decode overlap" cho GLM-5.3-Flash (KDA + DSA + MoE), có giới hạn số layer; bản cho DeepSeek-V4.1-Flash ở #42507.
- Đây là pipeline theo chiều sâu (ý Layered Prefill) cho hybrid, **đang vào upstream**. Mọi operator vẫn cùng số token. Chưa đọc code.
- Dùng làm baseline Layered thứ hai bên cạnh bản cài lại trong vLLM.

### vLLM PR #54145 — gom chunk prefill nhỏ khi đang có decode
- `--min-prefill-chunk-tokens`, `--max-prefill-chunk-delay-steps`.
- Tác giả báo: context token/step 1669 → 4785, GPU time prefill 15.1 → 8.8 µs/token (−42%), nhưng QPH chỉ +1.9%.
- Mở, conflict từ 31/08, chưa có review của maintainer.
- Là bằng chứng độc lập rằng chi phí cố định mỗi iteration có thật, và một bản thô của việc gom k chunk chỉ cho gain một chữ số thấp.

### vLLM RFC #52906 — P-PAS (arXiv 2608.15171), prefill budget thích nghi
- E2E −8.5% so với chunk cố định 2K. Chỉ điều chỉnh một budget toàn cục.

### vLLM #56457 → PR #57105 (merge 27/09/2026)
- Đặt trước workspace logits indexer cho trường hợp xấu nhất. Bộ nhớ indexer thành hằng số (≤ `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`); phần vượt cap chỉ thành thêm lần launch.
- Hệ quả cho G1c: xem `docs/09` §3.

### NVIDIA Dynamo — conditional disaggregation
- Prefill tại node decode chỉ khi ISL hiệu dụng < `eff_isl_threshold` (mặc định 2048) và tỉ lệ ≤ 0.7 (`eff_isl_ratio_threshold`). Hỗ trợ vLLM; SGLang từ chối annotation bypass (issue ai-dynamo/dynamo#11514).
- Nguồn: docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation.

### PPD — arXiv 2603.13358 (bổ sung số)
- Full prefill làm decode đồng thời chậm khoảng 48%; append-prefill chỉ khoảng 2%. Con số 2% nghĩa là budget P ít chặt hơn ví dụ ở PROPOSAL §2.5.3.

### IndexCache / IndexShare — arXiv 2603.12201
- Bỏ 75% compute indexer, prefill nhanh 1.82× trên model DSA 30B; GLM-5.2 dùng chung một indexer cho 4 layer DSA.

