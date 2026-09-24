# Phản biện: Adaptive FP4 Execution on Hopper — 2026-09-17

File idea: `../adaptive_fp4_execution_on_hopper.md` (tự chấm 8–8.5/10).
Phương pháp: kiểm tra trực tiếp vLLM/SGLang/TensorRT-LLM (PR, issue, docs), CUTLASS, PyPI, và paper gốc. WebSearch hết quota nên chưa quét arXiv rộng; phần "paper" dựa trên kiến thức đã có và các báo cáo trước.

## 1. Bước 1 của idea ("verify the software gap") — kết quả: gap đã bị lấp phần lớn

| Đường chạy FP4 trên Hopper | Trạng thái upstream | Link | Ý nghĩa |
|---|---|---|---|
| NVFP4 → FP8 block-scaled, convert lúc load, +0.8% bộ nhớ; +62% throughput, TTFT −43%, P99 ITL −46% trên Mistral-Small-4-119B | vLLM PR #38728 (04/2026), **chưa merge**, conflict | https://github.com/vllm-project/vllm/pull/38728 | Đúng "FP4 storage → FP8 compute" của idea, đã có code và số |
| MXFP4 → block-FP8 **lossless** (scale E8M0 là luỹ thừa 2, 6.0×2⁶ = 384 < 448), materialize FP8, expert tốn 2× bộ nhớ, KV cache 83.9 → 53.7 GiB; prefill 1.38–1.53×, decode 1.12× (H20, TP4) | vLLM PR #53709 (08/2026), opt-in `VLLM_DSV4_FP4_DEQUANT=1` | https://github.com/vllm-project/vllm/pull/53709 | Chính là "W4A8 cho prefill"; lựa chọn **tĩnh toàn cục**, không theo M hay theo expert |
| MXFP4 weight × FP8 activation fused MoE trên SM90 (FlashInfer CUTLASS "humming"), đọc 4-bit on-the-fly, không materialize | vLLM PR #54032 (08/2026), opt-in `--moe-backend flashinfer_cutlass_humming` | https://github.com/vllm-project/vllm/pull/54032 | Đúng kernel C2 của idea. Số của chính PR: "behind at low concurrency and ahead at high" (−2.6% ở 16, +5.6% ở 64, +1.8% ở 128 so với Humming) |
| Humming (humming-kernels, JIT GEMM/MoE, FP8 activation × INT/FP weight ≤ 8 bit, SM75+) làm backend mặc định cho MXFP4 MoE trên SM90, +19.9% throughput so với Marlin | vLLM PR #52318, #53848 (GLM W4AFP8), revert #53805 sau lỗi H100 | https://github.com/vllm-project/vllm/pull/52318 | W4AFP8 trên Hopper đã là mặc định, còn bug |
| Marlin nhận input FP8 (`VLLM_MARLIN_INPUT_DTYPE=fp8`) | vLLM, có bug trên sm_121a (issue #49546) | https://github.com/vllm-project/vllm/issues/49546 | Thêm một W4A8-FP8 nữa |
| TensorRT-LLM: NVFP4/MXFP4 **không** hỗ trợ Hopper; W4A8 AWQ có | Docs support matrix | https://nvidia.github.io/TensorRT-LLM/features/quantization.html | Gap thật ở TRT-LLM, nhưng là "framework feature" |
| SGLang: MegaMoE MXFP4/NVFP4 chỉ SM100; Hopper dùng đường khác; MXFP4 KV cache trên Hopper | SGLang issue #35557, PR #35558, #37138, #38080 | https://github.com/sgl-project/sglang/issues/35557 | Hopper là hạng hai với model FP4, nhu cầu có thật |
| CUTLASS mixed-input SM90: {fp16,bf16}×{int8,int4,int2} và {fp8}×{int4}; group size bội threadblock-K | CUTLASS example 55 | https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm | Nền kernel W4A8 có sẵn; FP4 E2M1 → FP8 là bit-shift rẻ |

**Kết luận mục 1:** ba trong năm contribution của idea (C2 kernel FP4→FP8 không materialize, và phần lớn C1 characterization prefill-vs-decode) đã tồn tại trong vLLM tính đến 08/2026, do chính nhu cầu chạy DeepSeek-V4/GLM/MiniMax-M3 FP4 trên H100/H200. Idea được viết như thể đường này chưa có; nó có, và đã có số.

## 2. Cái gì còn mở

1. **Chọn đường theo regime (C3, C4, C5).** Mọi PR trên đều là flag tĩnh, opt-in toàn cục. PR #54032 tự ghi nhận regime dependence ("behind at low concurrency, ahead at high") mà không khai thác. Chưa PR nào chọn W4A16/W4A8 theo M hay theo expert. Tiền lệ đổi đường theo M đã có trong vLLM (heuristic dequant + matmul cho AWQ khi M lớn; không xác minh được dòng code trong phiên này vì raw file 404), nên về kỹ thuật đây là một heuristic vài chục dòng, không phải một cơ chế mới.
2. **NVFP4 (scale E4M3, không phải luỹ thừa 2) → FP8 giữ ngữ nghĩa scale.** MXFP4 → FP8 là lossless; NVFP4 → FP8 gộp scale vào giá trị sẽ mất ~3 bit mantissa. PR #38728 giữ scale riêng dạng block-FP8 (`weight_scale_inv`), chưa rõ có lossy không. Đây là điểm kỹ thuật thật nhưng nhỏ.
3. **Điểm gãy M\*** giữa W4A16 và W4A8 trên Hopper: về nguyên lý đã biết từ Marlin (4× ở batch 16–32, giảm dần ở 64–128) và QServe. Số cụ thể cho FP4 MoE trên H200 chưa ai công bố có hệ thống, nhưng đây là bảng đo, không phải phát hiện.
4. **Bộ nhớ.** Materialize FP8 tốn 2× expert và mất 36% KV cache (PR #53709). Kernel on-the-fly (Humming, FlashInfer SM90) tránh được. Vấn đề "đừng materialize BF16" của idea đã được giải bằng "đừng materialize gì cả".

## 3. Kiểm tra độ bền

- **Frontier-model test:** đúng chiều thuận. Model mở hàng đầu giờ ship FP4-first (DeepSeek-V4 MXFP4 expert, Qwen3.8-Flash-Next W4A4 MXFP4, MiniMax-M3 NVFP4, Ling-3.0 FP4, GLM W4AFP8) và Hopper còn nhiều năm. Nhu cầu thật, có issue lỗi liên tục (#49070 MiniMax-M3 garbage trên sm90, #35557 DSV4 crash, #53805 revert Humming).
- **OSS test: rớt.** Phần còn mở (heuristic theo M, per-expert dispatch) là thứ vLLM có thể thêm bằng một PR ngắn, và họ đang hoạt động đúng vùng này hằng tuần. Kernel thì Humming/FlashInfer/CUTLASS đã có.
- **Abstraction test:** "storage precision ≠ compute precision, chọn theo regime" là câu hay, nhưng cùng luận điểm với Mix-Quant (precision theo pha trên Blackwell, xem `cluster_reports/review_cluster_G_deepseek.md`) và với 15 năm literature mixed-precision GEMM. Không đủ làm trụ cho paper ASPLOS/OSDI.
- **Cheap falsification: đạt.** Sweep GEMM M ∈ {1…8192} với Marlin FP4, Humming, FlashInfer SM90, FP8 materialized trên H200: 3–5 ngày, 1 GPU.
- **Evidence test:** dễ, vì có số upstream sẵn.

## 4. Chấm điểm

| Tiêu chí | Đánh giá |
|---|---|
| Novelty risk | **Cao.** Kernel và characterization cơ bản đã có trong vLLM; phần còn lại là heuristic dispatch. |
| Feasibility | **Rất cao.** 8 H200 là đúng phần cứng; kernel có sẵn để so; 2 tuần ra số. |
| Thời gian đến kết quả đầu | 1 tuần (sweep), 3–4 tuần (per-expert dispatch prototype trên vLLM). |
| Venue | Workshop MLSys/ASPLOS hoặc short paper; full paper MLSys chỉ nếu mở rộng thành "chạy model FP4-native trên Hopper: characterization + runtime" với ≥ 4 model và số end-to-end. OSDI/ASPLOS không thực tế. |
| **Điểm** | **5/10** làm paper độc lập (idea tự chấm 8–8.5 vì viết trước khi kiểm tra upstream). **7/10** nếu dùng làm một section/knob trong HyPrefill. |

## 5. Khuyến nghị (cập nhật 18/09/2026)

**Không làm thành paper riêng — nhưng cũng không bỏ.** Phần đo lường của idea này trùng ~70% với kill test tuần 1–2 của HyPrefill: dựng `cost_MoE(c)` trên H200 buộc phải chọn đường kernel (Marlin W4A16 vs Humming W4AFP8), nên thêm chiều precision vào sweep chỉ tốn ~2 ngày.

### 5.1 Việc phải làm trong tuần 1–2 của HyPrefill (2 ngày)
- Đo `cost_MoE(c, path)` với path ∈ {Marlin W4A16, Humming W4AFP8, FlashInfer SM90 MXFP4×FP8, FP8 materialized}.
- Dump phân bố M_e theo expert ở vài mức tải trên DeepSeek-V4-Flash hoặc GLM-5.3-Flash.
- Kết quả vào thẳng cost model của HyPrefill: scheduler chọn đồng thời (c_MoE, precision path).

### 5.2 Kiểm tra arXiv (18/09/2026)
- "FP4 checkpoint Hopper FP8 expansion inference": **0 kết quả**.
- "weight precision selection runtime batch size GEMM W4A8 W4A16": **0 kết quả**.
- "quantization format choice KV cache capacity tradeoff serving": **0 kết quả**.
- "MXFP4 inference serving": 5 kết quả, không bài nào về chọn precision path theo regime trên Hopper.

Đọc hai chiều: chưa ai claim trên giấy, nhưng arXiv trống trong khi upstream đầy PR là dấu hiệu cộng đồng coi đây là engineering. Bốn PR vLLM liên quan đều không kèm paper.

### 5.3 Bản sharpened nếu vẫn muốn theo đuổi độc lập — "Precision residency"
Framing kernel + tìm M\* đã bị vLLM lấp. Framing còn mở là **bài toán phân bổ bộ nhớ**:

> Cho ngân sách bộ nhớ B, chọn tập expert nào thăng hạng lên FP8 (nhanh hơn, tốn 2× bộ nhớ) và tập nào giữ FP4, để tối đa throughput dưới ràng buộc dung lượng KV cache.

Cơ sở: PR #53709 đo bật FP8 toàn cục làm KV cache rơi 83.9 → 53.7 GiB mỗi GPU (−36%), đổi lấy prefill 1.38–1.53× và decode 1.12×; PR để đó dưới dạng biến môi trường, không trả lời "khi nào nên bật". Phân bố token theo expert lệch mạnh nên expert nóng (M_e lớn, compute-bound) hưởng lợi từ FP8 tensor core, còn expert lạnh (M_e nhỏ, memory-bound) giữ FP4 là đủ.

Bền hơn trước OSS test: vLLM thêm heuristic dispatch theo M chỉ mất một PR ngắn, nhưng sẽ không tự làm precision residency planning có ràng buộc bộ nhớ.

Kill test 1 tuần: nếu thăng hạng 20% expert đạt ≥ 80% lợi ích với ~20% chi phí bộ nhớ → đáng viết tiếp. Nếu đường cong phẳng → bỏ.

### 5.4 Vai trò trong danh mục
Đây là **phương án lui nhanh nhất** nếu HyPrefill trượt cổng G1 ở tuần 2, vì lúc đó số liệu đã có sẵn và không cần hạ tầng mới. Xem bảng phương án lui trong `01_HYPREFILL_PLAN.md` §0.2b.

## 6. Lỗi thực tế trong file idea cần sửa

- "Hopper deployment of FP4 usually falls back to BF16 compute" — đúng tới đầu 2026, sai từ 08/2026: mặc định MXFP4 MoE trên SM90 trong vLLM là Humming (W4AFP8).
- "Do not materialize the full BF16 weight tensor" là yêu cầu đã được đáp ứng; vấn đề thực tế bây giờ là materialize FP8 (2× expert memory) hay on-the-fly.
- Baseline "Humming" được liệt kê như thứ cần so sánh, nhưng nó chính là lời giải của C2.
- MXFP4 → FP8 là lossless (E8M0), chỉ NVFP4 mới có vấn đề ngữ nghĩa scale; file gộp hai trường hợp làm một.

---

## 7. Cập nhật 24/09/2026 — phân biệt NVFP4 và MXFP4

Báo cáo gốc gộp hai định dạng. Kiểm lại cho thấy chúng ở hai trạng thái rất khác nhau trên Hopper.

| Định dạng | Trạng thái | Bằng chứng |
|---|---|---|
| **MXFP4** | **Đã ship, đông** | Humming là backend mặc định cho MoE trên SM90 (PR #52318); PR #53709 cho đường block-FP8 lossless opt-in; PR #54032 FlashInfer SM90 MXFP4×FP8 đọc 4-bit on-the-fly |
| **NVFP4** | **Chưa giải quyết** | PR #38728 vẫn **mở, kẹt merge conflict từ 05/2026**, maintainer chưa duyệt, không có số đo độ chính xác ("perplexity to be added"). Issue #49070 vẫn **mở**: MiniMax-M3 NVFP4 ra rác ở concurrency 1 và crash CUDA ở concurrency 8 trên sm90, do **ba lỗi độc lập** (thiếu tham số activation swigluoai; lỗi căn chỉnh bộ nhớ khi `moe_intermediate_size` không align; lỗi indexing top-k trong Marlin epilogue), cần hai PR #48929 và #48624 áp cùng lúc |

Kết luận: gap **NVFP4 trên Hopper mở hơn** báo cáo gốc đánh giá. arXiv vẫn trống.

### 7.1 Câu hỏi của maintainer — phải trả lời được trước khi làm

Trong PR #38728, maintainer hỏi:

> "can you explain why you wouldn't just use a FP8 checkpoint instead? To me it seems you are losing accuracy for no reason if you are starting from a lower precision checkpoint"

Tác giả không trả lời chính thức. Đây là câu hỏi đúng và giết framing gốc: nếu vật chất hoá FP8 trong HBM thì tốn đúng bằng checkpoint FP8, mà checkpoint FP8 chính xác hơn — vậy chuyển FP4→FP8 lúc load là mất chất lượng không đổi lấy gì.

Chỉ một câu trả lời đứng vững: **giữ FP4 trong HBM, chỉ bung FP8 trong register/SMEM** (đường on-the-fly). Nhưng đường đó đã ship cho MXFP4; làm nó chạy với NVFP4 là kỹ thuật kernel, không phải nghiên cứu.

### 7.2 Hệ quả: framing đúng là phân bổ bộ nhớ, không phải kernel

Câu hỏi của maintainer không giết idea mà chỉ ra chỗ giá trị thật nằm:

> Cho ngân sách HBM cố định, mỗi tensor trọng số nên nằm ở precision nào, và phép tính diễn ra ở precision nào? Hai thứ tách rời được trên phần cứng không hỗ trợ native.

PR #53709 cho số cụ thể: bật FP8 toàn cục → KV cache 83.9 → 53.7 GiB mỗi GPU (−36%), đổi lấy prefill 1.38–1.53×. **Không ai trả lời khi nào nên đánh đổi, và không ai làm theo từng expert.** Đó là §5.3 "precision residency".

### 7.3 Điểm sửa lại

**5/10 → 6/10** cho bản NVFP4 + precision residency, vì gap NVFP4 thật hơn đánh giá ban đầu.

Vẫn dưới HyPrefill và vẫn là **phương án lui số 1**, không thay thế. Lý do: sản phẩm cuối là một kernel cộng một heuristic phân bổ; vLLM sẽ ship phần kernel. Phần nghiên cứu là measurement study cộng allocation policy — đủ cho workshop hoặc systems venue hạng trung, không đủ cho ICML/SOSP.

### 7.4 Nếu muốn kiểm trong 1 tuần

1. Xác nhận NVFP4 trên H200 còn hỏng thật: chạy MiniMax-M3 NVFP4, xem có tái hiện issue #49070 không.
2. Đo `cost_MoE(c, path)` cho 4 đường kernel (đã nằm trong tuần 1–2 của HyPrefill, +2 ngày).
3. Dump phân bố `M_e` theo expert ở vài mức tải.
4. Tính oracle: thăng hạng 20% expert nóng lên FP8 đạt bao nhiêu % lợi ích với bao nhiêu % chi phí bộ nhớ? Nếu ≥80% lợi ích với ~20% bộ nhớ → đáng viết tiếp. Đường cong phẳng → bỏ.
