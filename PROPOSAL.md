# HyPrefill — Research Proposal

**Tên đầy đủ:** Operator-Decoupled Chunked Prefill for Hybrid LLM Serving
**Mục tiêu nộp:** ICML 2027 · dự phòng SIGMETRICS 2027 · NeurIPS 2027 (deadline ở §8)
**Phần cứng:** 1–8 × NVIDIA H200 (141 GB)

---

## Abstract (bản nộp)

> **HyPrefill: Operator-Decoupled Chunked Prefill for Hybrid LLM Serving**
>
> Hybrid large language models that interleave full or sparse attention, linear attention (e.g. Gated DeltaNet), and Mixture-of-Experts layers have become the default architecture for long-context serving. Serving systems process long prompts with chunked prefill, splitting the prompt into fixed-size chunks that are co-scheduled with decode requests to bound time-between-tokens (TBT). Today, a single chunk size governs every layer of the model. We show that this uniform choice is fundamentally mismatched to hybrid architectures: different operator groups are limited by *different kinds* of constraint. Attention and sparse-attention indexers are time-limited with a cost that grows linearly in context length; indexer score buffers and chunked-scan workspaces are memory-limited with a footprint proportional to the product of chunk size and context length; MoE and kernel launches are amortization-limited and strictly prefer large chunks. The uniform chunk is therefore the minimum over all operators, and at long context it forces linear-attention and MoE layers to run at a fraction of their efficiency. We present HyPrefill, a scheduler that assigns each operator group its own chunk size derived from an offline-calibrated cost model: attention groups process small chunks every iteration, while linear-attention and MoE groups accumulate activations in a buffer and process k chunks at once, staggered across groups so that no iteration exceeds the TBT budget. On Qwen3-Next-80B-A3B, Qwen3.8-27B, Kimi-Linear-48B-A3B and Qwen3.8-Flash-Next, HyPrefill reduces time-to-first-token by X–Y% and end-to-end latency by Z% at unchanged P99 TBT compared with chunked prefill, layered prefill, and deadline-aware adaptive chunking, with gains growing with context length.

X, Y, Z điền từ oracle bước 2 và prototype bước 11–13.

---

## 1. Vấn đề

### 1.1 Bối cảnh

Một request sinh văn bản có hai pha. **Prefill** xử lý toàn bộ prompt trong một forward, compute-bound. **Decode** sinh từng token, memory-bound, để GPU rảnh nhiều. Để tận dụng phần compute rảnh đó mà không làm decode giật, Sarathi-Serve (OSDI 2024) đề xuất **chunked prefill**: cắt prompt thành chunk c token, mỗi iteration chạy toàn bộ batch decode cộng một chunk prefill. Kỹ thuật này là mặc định trong vLLM V1.

Quan hệ cơ bản:

```
T_iter(c) = D + T_chunk(c)          D = thời gian decode batch
TBT       ≈ T_iter(c)
TTFT      ≈ (L / c) · T_iter(c)
```

Với SLO dạng "P99 TBT ≤ B", người vận hành chọn c lớn nhất sao cho T_iter ≤ B. Trong vLLM đó là `max_num_batched_tokens`, trong SGLang là `chunked_prefill_size`. **Một số duy nhất cho toàn bộ model.**

### 1.2 Kiến trúc đã thay đổi, scheduler thì chưa

Model mở hàng đầu năm 2026 không còn là stack đồng nhất của attention và FFN. Chúng xen kẽ nhiều loại operator:

| Model | Bố cục | MoE |
|---|---|---|
| Qwen3-Next-80B-A3B | 12 × (3 GDN + 1 gated attention) | 512 expert, 10+1 active |
| Qwen3.8-27B | 16 × (3 GDN + 1 gated attention), FFN dense | không |
| Qwen3.8-Flash-Next | 12 × (3 GDN + 1 Qwen Sparse Attention) | 512 expert, 10+1 active |
| Kimi-Linear-48B-A3B | KDA : MLA = 3 : 1 | có |
| GLM-5.3-Flash | linear + sparse attention | 320B/18B active |
| Nemotron 3.x | Mamba2 + attention | có |

Các operator này có hình dạng chi phí khác nhau về chất, không chỉ khác hằng số.

### 1.3 Quan sát trung tâm

Với chunk c token tại vị trí context t:

**Full attention** — mỗi token mới nhìn t token cũ:
```
cost_FA(c, t) ≈ α · c · (t + c/2) + β · c
cost/token    ≈ α · (t + c/2) + β      → gần phẳng theo c, tăng tuyến tính theo t
```

**Sparse attention có indexer** (DSA, QSA, CSA) — indexer chấm điểm toàn bộ t key rồi attention trên top-k:
```
cost_SA(c, t) ≈ α_idx · c · t + κ · c · k + β · c      α_idx ≈ α/10 … α/15
mem_idx(c, t) ≈ c · t · H_idx · 4 byte                 ← buffer logits, phình theo c·t
```

**Linear attention / GDN** — state kích thước cố định, thuật toán chunkwise kernel C = 64:
```
cost_GDN(c) ≈ (c/C)·[γ·C²·d + δ·C·d²] + ε_launch
cost/token  ≈ γ·C·d + δ·d² + ε_launch/c                → giảm theo c, KHÔNG phụ thuộc t
```

**MoE** — thuế đọc trọng số expert mỗi chunk:
```
E_touched(c) ≈ E · [1 − (1 − k/E)^c]                   Qwen3-Next: c = 256 → ~505/512 expert
cost_MoE(c)  ≈ W_read · min(1, E_touched/E) + η · c
cost/token   ≈ W_read/c + η                            → giảm mạnh theo c, không phụ thuộc t
```

Ba **loại** ràng buộc, không phải ba hằng số:

| Loại ràng buộc | Operator | Hình dạng | Chunk muốn |
|---|---|---|---|
| Thời gian theo t | full attention; indexer sparse attention | cost/token ∝ t | nhỏ dần khi t tăng |
| Bộ nhớ theo c·t | buffer logits indexer (vLLM #56457); buffer chunked-scan GDN/KDA (vLLM #54775) | mem ∝ c·t | nhỏ dần khi t tăng |
| Khấu hao | MoE (đọc expert mỗi chunk); launch kernel GDN | cost/token ∝ 1/c | càng lớn càng tốt, không phụ thuộc t |

**Trạng thái kiểm chứng (2026-09-29, lượt đo sơ bộ 1830 MHz; chưa kết luận, đo lại theo chuẩn mới rồi mới quyết):**
- *Thời gian theo t:* FA3 tăng tuyến tính theo t như dự đoán.
- *Bộ nhớ theo c·t:* buffer indexer QSA tỉ lệ c·t, nhưng vLLM 0.30 tự chia query để buffer ≤ 512 MB, nên có thể không còn là ràng buộc của scheduler. Cần đo trên model thật và kiểm buffer chunked-scan GDN (#54775) trước khi kết luận (`plan/02`, G1c).
- *Khấu hao:* GDN có chi phí cố định lớn mỗi lần gọi (~53 µs GPU, ~0.2 ms CPU khi eager), không phụ thuộc t. MoE đọc hết expert từ c ≥ 256, nhưng trong batch trộn decode cũng đọc chung weight đó; mức ảnh hưởng phụ thuộc batch decode, đang đo (`plan/02` §2.4).

**Vì mọi layer phải xử lý cùng c mỗi iteration, chunk đồng nhất là `min` trên toàn bộ ràng buộc.** Ở context dài, attention hoặc indexer kẹp c xuống vài trăm token, và GDN cùng MoE bị kéo theo, chạy ở đúng vùng chúng kém hiệu quả nhất.

---

## 2. Giả thuyết và cơ chế

### 2.1 Giả thuyết

> Cho mỗi nhóm operator một chunk size riêng, lấy từ cost model hiệu chỉnh offline, giảm TTFT và E2E ở P99 TBT không đổi; khoảng cách mở rộng theo độ dài context.

### 2.2 Cost model

Cho budget prefill mỗi iteration `P = B − D` (B là TBT SLO, D là thời gian decode batch):

```
Đồng nhất:  r_u(t) = max c  s.t.  Σ_g cost_g(c, t) ≤ P   và   mem_g(c, t) ≤ M_g ∀g

Tách:       r_d(t) = max c  s.t.  cost_FA(c, t) + [Σ_{g≠FA} cost_g(k_g·c)] / k_g ≤ P    (khấu hao)
                            và   cost_FA(c, t) + max_{g≠FA} cost_g(k_g·c)       ≤ P    (đỉnh, nhờ so le)
                            và   mem_g(k_g·c, t) ≤ M_g ∀g

Gain(t) = r_d(t) / r_u(t)
```

`k_g ∈ {1, 2, 4, 8, 16}` là số chunk mà nhóm g gom lại trước khi chạy một lần.

**Cần làm rõ khi hiệu chỉnh (2026-09-29, giả thuyết từ lượt đo sơ bộ):** (a) trong vLLM, GEMM và MoE của decode và prefill chạy chung một lần gọi, nên với các nhóm này cost là `cost_g(n_decode + c)` chứ không phải `D_g + cost_g(c)`; attention và GDN là kernel riêng nên công thức cộng rời vẫn đúng với chúng. (b) Ở chunk nhỏ, một iteration có thể bị giới hạn bởi CPU (các op chạy eager giữa các mảnh CUDA graph), nên cost một iteration không chỉ là tổng kernel. (c) Với TP > 1 phải cộng all-reduce. Cả ba được đo ở bước 1–2 và kiểm bằng forward thật của vLLM trước khi dùng.

### 2.3 Cơ chế

1. **Operator groups.** Chia model thành các nhóm layer liên tiếp theo loại operator. Với Qwen3-Next: nhóm attention và nhóm GDN xen kẽ, nhóm MoE sau mỗi mixer.
2. **Chunk theo nhóm.** Nhóm attention xử lý `c_FA(t)` token mỗi iteration, lấy từ cost model closed-form. Nhóm GDN và MoE dùng `k · c_FA`.
3. **Activation buffer.** Output của nhóm attention cho chunk i giữ trong buffer đến khi đủ k chunk thì nhóm phi-attention xử lý một lần. Kích thước `k · c_FA · d_model · 2` byte, cỡ vài chục MB.
4. **Stagger.** Các nhóm phi-attention chạy chunk lớn ở iteration khác nhau để đỉnh mỗi iteration không vượt B.
5. **Ràng buộc kernel.** Kernel chunk GDN giữ nguyên C = 64 (`FLA_CHUNK_SIZE`, vLLM PR #49827); boundary align bội 64 và align block Mamba khi bật prefix cache (vLLM PR #54076); chunk attention chọn bội 128 vì FA3 có bậc thang theo tile 128 (bước 1); RFC #55524 yêu cầu chunk size kernel giống nhau giữa prefill và decode để bit-identical. **HyPrefill chỉ đổi số token scheduler đưa vào mỗi lần gọi, không đổi chunk size kernel.**
6. **Không ước lượng runtime.** Cost model đo offline, tra bảng theo (operator, t). Đây là điểm khác COREY.

### 2.4 Phạm vi (cập nhật 18/09/2026 sau blog hạ tầng GLM)

Luận điểm có ý nghĩa ở **mọi GPU chạy đồng thời prefill và decode dưới ràng buộc TBT**. Xếp theo độ bền của lập luận:

1. **Node decode của hệ PD/EPD chạy append-prefill nhiều lượt — phạm vi chính.** Token mới ít (1–4K) nhưng t rất lớn (64–256K), nên chênh lệch giữa ràng buộc của các operator group đạt cực đại. "Not All Prefills Are Equal" (arXiv 2603.13358, ICML'26) chỉ ra chuyển KV đi rồi về đắt hơn phần prefill nhỏ, nên append-prefill chạy ngay trên node decode, giảm 68% TTFT từ lượt hai. **Phạm vi này đúng ngay cả khi toàn ngành dùng disaggregation.**
2. **Triển khai colocated** mặc định của vLLM và SGLang, điển hình cho cụm 1–8 GPU trong lab và doanh nghiệp. "Prefill-Decode Aggregation or Disaggregation?" (arXiv 2508.01989) cho thấy mỗi chế độ thắng ở một vùng tải.
3. **Node prefill thuần không có ràng buộc TBT** nên HyPrefill không có gain. Paper nói thẳng và có một thí nghiệm chứng minh (ablation f, bước 12).

**Headwind phải trả lời thẳng.** Blog hạ tầng GLM (17/09/2026, `docs/07_...`) cho thấy một lab tuyến đầu phục vụ đúng loại model hybrid (KDA linear attention + sparse attention + MoE), ở quy mô hơn 100.000 accelerator, context 1M, đã chọn kiến trúc **Encode-Prefill-Decode disaggregated**. Kịch bản test của họ là "Prefill alone, Prefill + KV Transfer, Decode alone", và toàn bài **không nhắc chunked prefill một lần nào**. Reviewer sẽ hỏi: colocated còn quan trọng không?

Trả lời: (a) phạm vi 1 ở trên không phụ thuộc vào việc hệ có disaggregate hay không; (b) đa số triển khai không ở quy mô 100.000 accelerator, và mặc định của hai engine phổ biến nhất là colocated; (c) EPD đòi hỏi interconnect băng thông cao mà nhiều cụm không có. **Lưu ý trung thực:** blog GLM không nói rõ node decode của họ có chạy append-prefill hay không; lập luận (a) dựa trên PPD, không được gán suy đoán cho GLM.

### 2.5 Vì sao append-prefill là regime mạnh nhất, không chỉ là regime an toàn

Mục 2.4 nói append-prefill miễn nhiễm với câu hỏi disaggregation. Mục này nói thêm một lý do mạnh hơn: **đó cũng là nơi cơ chế cho gain lớn nhất.**

#### 2.5.1 Append-prefill là gì

Hội thoại nhiều lượt hoặc vòng lặp agent gọi tool:

- **Lượt 1**: prompt 100K. Cụm prefill xử lý, chuyển KV sang cụm decode, sinh câu trả lời.
- **Lượt 2**: thêm 2K token mới (tool output, tin nhắn tiếp). Nhưng KV của 100K token cũ **đã nằm sẵn trên cụm decode**.

Hai lựa chọn. Gửi ngược cả hội thoại về cụm prefill thì phải chuyển 100K KV đi rồi chuyển về, đắt hơn nhiều so với phần việc thật là 2K token. Hoặc prefill 2K token đó ngay trên cụm decode. PPD (arXiv 2603.13358, ICML'26) đo lựa chọn thứ hai giảm 68% TTFT từ lượt hai trở đi.

Hệ quả: **cụm decode vẫn phải chạy prefill, xen giữa các request đang decode, nên ràng buộc TBT vẫn sống.** Không disaggregate đi đâu được. Đây là lý do phạm vi 1 trong §2.4 không phụ thuộc vào kiến trúc triển khai.

#### 2.5.2 Cơ chế của gain — vì sao tách chunk làm attention chạy được chunk lớn hơn

Điểm dễ hiểu nhầm: tách chunk không chỉ giúp GDN và MoE, mà giúp **cả attention**.

```
Đồng nhất:  mỗi iteration trả  cost_FA(c,t) + cost_GDN(c) + cost_MoE(c) ≤ P
            → c bị kẹp bởi tổng của cả ba

Tách + so le: đa số iteration chỉ trả  cost_FA(c,t) + (phần khấu hao nhỏ) ≤ P
            → c_FA được dùng gần trọn P  →  c_FA(tách) > c_FA(đồng nhất)
```

Nhóm GDN và MoE gom k chunk rồi chạy một lần ở iteration riêng, so le nhau để không iteration nào vượt B. Vậy cả ba nhóm đều chạy ở chunk gần tối ưu của mình thay vì cùng chịu chung một `min`.

#### 2.5.3 Ví dụ số

Prefix t = 128K, token mới Δ = 2048, SLO P99 TBT B = 50 ms, decode batch ăn D = 20 ms → P = 30 ms.

| Operator | Chunk chịu được ở t = 128K | Số lần chạy cho Δ = 2048 |
|---|---|---|
| Attention | ~256 token (bị kẹp bởi context) | 8 |
| GDN | cả 2048 một lần | 1 nếu được tự chọn |
| MoE | cả 2048 một lần | 1 nếu được tự chọn |

Chunk đồng nhất ép GDN và MoE chạy 8 lần thay vì 1. Với MoE, mỗi lần chạy đọc gần như toàn bộ trọng số expert (coupon-collector: c = 256 đã chạm ~505/512 expert trên Qwen3-Next), nên đó là **8 lần đọc toàn bộ expert pool chỉ để xử lý 2048 token**. Với GDN, đó là 8 lần trả `ε_launch` thay vì 1.

Con số 256 và 2048 là minh hoạ; số thật đến từ bước 1–2.

#### 2.5.4 Vì sao gain không bị pha loãng — điểm chính

Gain là hàm tăng theo t: ở context ngắn, `cost_FA` nhỏ nên ràng buộc chặt nhất là GDN+MoE và chúng đã có chunk tử tế, tách ra không giúp nhiều. Ở context dài, `cost_FA` chiếm gần hết P và bóp mọi thứ, tách ra giúp nhiều.

Thời gian prefill tỉ lệ với số iteration, và số iteration là tích phân của nghịch đảo tốc độ:

```
N(L) = ∫₀^L dt / r(t)              r(t) = số token prefill mỗi iteration
Gain hiệu dụng = N_đồng nhất / N_tách
```

- **Long-context prefill từ đầu**: công việc trải trên t ∈ [0, L]. Phần đầu chạy ở context ngắn nơi gain(t) ≈ 1. Gain hiệu dụng là **trung bình có trọng số** của gain(t), luôn **nhỏ hơn** gain đỉnh tại t = L.
- **Append-prefill**: vì Δ ≪ t₀, **mọi chunk đều ở t ≈ t₀**. Gain hiệu dụng = gain(t₀), đúng bằng giá trị đỉnh, không bị pha loãng.

Nói cách khác: cùng một độ dài context tối đa, append-prefill cho toàn bộ gain còn long-context chỉ cho một phần. Đây là lý do số headline nên lấy từ append-prefill.

#### 2.5.5 Giới hạn phải nói rõ

Lợi ích bị chặn bởi `k ≤ Δ / c_FA(t)`. Nếu token mới quá ít (Δ = 512, c_FA = 256) thì k chỉ bằng 2 và gain nhỏ. Vậy workload phải quét cả Δ, không chỉ quét t:

```
Δ ∈ {256, 512, 1K, 2K, 4K}   ×   t ∈ {16K, 64K, 128K, 256K}
```

Báo cáo trung thực vùng Δ nhỏ không thắng. Đây là một trục của bản đồ regime (Hình 3), bên cạnh trục kiến trúc và trục context.

#### 2.5.6 Hệ quả cho kế hoạch

- Workload append-prefill là workload chính, quét thêm chiều Δ (bước 4, bước 11–13).
- Ablation (g) ở bước 12 mô phỏng node decode của hệ EPD: nếu gain còn đáng kể ở đó, HyPrefill đúng ngay cả trong thế giới disaggregated. Đây là lập luận bền nhất của bài.
- Long-context vẫn giữ trong bộ workload, làm đối chiếu và để chứng minh hiệu ứng pha loãng là thật.

---

---

## 3. Vị trí trong literature

Chi tiết và số liệu trong `docs/02_RELATED_WORK.md`. Tóm tắt:

| Công trình | Đơn vị chunk | Tín hiệu | Kiến trúc | Khác HyPrefill |
|---|---|---|---|---|
| Sarathi-Serve (OSDI'24) | token, một budget toàn cục | profiling offline theo TBT | dense | một số cho cả model |
| Layered Prefill (MLSys'26 Oral) | nhóm layer | N_lg(L) tĩnh | MoE + full attention | một granularity cho mọi nhóm; **không có SSM/GDN** |
| SLOWeave (2609.07883) | token, một chunk mỗi iteration | deadline B_t, T đơn điệu | dense | một chunk cho mọi layer |
| COREY (2604.10597) | kernel scan | entropy activation runtime | Mamba-1 | **kết quả âm**: chậm hơn static 0.7–8.8% |
| PrefillOnly §4.2 (2505.07203) | FFN chunk, attention nguyên khối | — | dense | mục tiêu bộ nhớ, không decode, không TBT |
| FlowPrefill (2602.16603) | biên operator để **preempt** | — | dense + MoE | cùng số token qua mọi operator |
| Marconi (MLSys'25) | — | FLOP-aware | attention + Mamba | caching, không scheduling |

**Bằng chứng ủng hộ từ công nghiệp (blog hạ tầng GLM, 17/09/2026).** Hai điểm dùng được trong Motivation:
- GLM cấp **chiến lược song song riêng cho linear attention**: "intra-node tensor parallelism for linear attention and the LM Head". Tiền lệ công nghiệp cho nguyên tắc "operator khác nhau cần chính sách khác nhau", trên trục parallelism thay vì trục chunk scheduling.
- Kernel KDA decode của họ có overhead dư thừa lớn, khấu hao được: bản gốc tile theo chiều V khiến cùng phép normalization và gating FP32 lặp bốn lần; gộp tile vào một thread block cho **1.71×**. Bằng chứng độc lập rằng `cost_GDN(c) = a + b·c` có hằng số `a` đáng kể, tức GDN thật sự thích chunk lớn.
- Phần cứng của họ "relatively limited chip memory capacity and bandwidth" → ràng buộc bộ nhớ (loại 2) càng quan trọng, không kém đi.

**Xác minh độ mới (18/09/2026).** Quét arXiv, GitHub API trên vLLM/SGLang/TensorRT-LLM, docs LMDeploy/Dynamo/Mooncake/MLX/MLC-LLM/llama.cpp, và tech report của tám họ model. Không paper hay engine nào cho các operator group khác nhau chunk size khác nhau trong cùng một forward pass, chọn từ cost model theo operator. Mọi engine vẫn một knob toàn cục. Layered Prefill chưa vào upstream nào.

---

## 4. Kế hoạch thực nghiệm

### 4.1 Model

| Model | Vai trò | GPU |
|---|---|---|
| Qwen3-Next-80B-A3B | **Anchor**, serving path chín | 2 × H200 bf16 |
| Qwen3.8-27B | Đối chứng **không MoE**, cùng họ GDN | 1 × H200 |
| Qwen3.8-Flash-Next | **Kiểm tra độ bền**: sparse attention + indexer | 4 × H200 bf16 / 2 × FP8 |
| Kimi-Linear-48B-A3B | Linear attention khác họ, khác vendor | 1 × H200 |
| Qwen3-30B-A3B | Đối chiếu Layered Prefill (full attention + MoE) | 1 × H200 |

### 4.2 Workload

1. **Long-context**: prompt 32K–256K (LongBench-v2, RULER), output 256–1024.
2. **Append-prefill agentic — WORKLOAD CHÍNH**: prefix cache hit t ∈ {16K, 64K, 128K, 256K}, token mới mỗi lượt Δ ∈ {256, 512, 1K, 2K, 4K}, decode chạy song song. Mô phỏng node decode của hệ PD/EPD. Đây vừa là phạm vi bền nhất trước xu hướng disaggregation vừa là regime gain lớn nhất và không bị pha loãng (xem §2.5). Lưới (t, Δ) ở trên dùng cho micro-benchmark và mô phỏng; số headline chạy trên trace thật:

   Trace (chốt 2026-09-29): **`semianalysisai/cc-traces-weka-062126-256k`** (HuggingFace, Apache-2.0, 570 MB). 393 phiên Claude Code thật, 68 266 request (28 444 lượt của agent chính, 39 822 request của 1 697 nhóm subagent chạy song song), input trung bình ~101K token, output trung bình ~860 token, mỗi request `input + output ≤ 256 000`. Mỗi request có timestamp tương đối `t`, `in`, `out`, và `hash_ids` theo block 64 token để biết phần prefix dùng lại; **không có text**. Hệ quả khi dùng:
   - Sinh token giả khớp `hash_ids` (cùng hash → cùng token), để prefix cache của vLLM hit đúng như trong trace. Độ dài đếm bằng tokenizer của Claude; dùng nguyên số token như trong trace và ghi rõ trong paper.
   - Bắt buộc bật prefix cache, nên chịu ràng buộc cắt chunk theo block Mamba (`plan/05` §2.5).
   - Context tới 256K giới hạn số request chạy song song trên 2×H200: tính dung lượng KV + state Mamba trước khi chọn mức tải.
   - Mức tải: co giãn timestamp theo một hệ số (ghi rõ hệ số), giữ thứ tự và độ chồng lấp của subagent.
   - Qwen3-30B-A3B (tối đa 40K) không chạy được trace này; model đó chỉ dùng kiểm chứng Layered trên arXiv/ShareGPT.
3. **Mixed**: 70% chat ngắn (1–4K) + 30% long-context, đến theo Poisson, quét tải.

### 4.3 Baseline

Sarathi static (quét chunk, bội 128), Layered Prefill (2510.08055, quét số nhóm layer), SLOWeave (2609.07883, tune biên δ), HyPrefill-static (ablation bỏ phụ thuộc vị trí), COREY như cautionary baseline. Mọi baseline cài lại trong cùng engine, chạy trên cùng phần cứng và cấu hình, dùng cùng bảng cost, và được tune tốt nhất ở từng SLO; giao thức đầy đủ ở `docs/03_MEASUREMENT.md` §7.

**Engine và cách cài baseline (quyết định 29/09/2026):** mọi policy chạy trong **cùng vLLM 0.30**. Layered Prefill được cài là cấu hình k = 1 của **cùng hạ tầng chạy theo nhóm layer** với HyPrefill, nên hai bên chỉ khác chính sách chunk. Bản cài lại được kiểm chứng với code gốc của tác giả (fork nanovllm) trên Qwen3-30B-A3B: tỉ lệ cải thiện layered/chunked phải khớp trong ±15%. Một bảng tham khảo khác engine (nanovllm gốc so với vLLM của mình) được báo cáo như tác giả Layered đã so với vLLM 0.10.2, không dùng làm claim. SLOWeave cũng cài trong cùng vLLM.

**Layered Prefill là baseline chính** (cập nhật 24/09/2026): phải cài lại cho hybrid có GDN để so công bằng, và mọi con số headline là **HyPrefill / Layered**, không phải HyPrefill / Sarathi. So với Sarathi sẽ gán nhầm phần gain của pipeline theo chiều sâu cho HyPrefill.

### 4.4 Metric và giao thức benchmark (chốt 2026-09-29, trước khi có số G1/G2)

Theo đúng cách baseline chính (Layered Prefill, arXiv 2510.08055 §6) và SLOWeave, DistServe đo, để số của mình so được với cách họ claim:

- **Metric headline: goodput** = mức tải (request/s, đến theo Poisson) lớn nhất mà **≥ 90% request đạt SLO**. Một request đạt SLO nếu TTFT ≤ SLO_TTFT **và** mọi TBT sau đó ≤ SLO_TBT (định nghĩa của Layered Prefill; SLOWeave dùng P99 TBT trong request, báo thêm cả biến thể này). Đường **tỉ lệ đạt SLO theo mức tải** là hình chính, quét dày quanh điểm sụp của baseline.
- **SLO_TBT (chốt 2026-09-29):** headline **50 ms**, quét thêm **25 và 100 ms** (như SLOWeave quét 10 / 25 / 50). Báo thêm mức tính từ phần cứng như Layered Prefill: 5 × thời gian một bước decode 32 request ở context 4096 (họ ra 125 ms trên 2×H100), tính lại trên 2×H200 cho từng model.
- **SLO_TTFT (chốt quy tắc 2026-09-29):** cố định theo từng (model, workload), như cả ba paper (Layered: 5 s ShareGPT, 10 s arXiv; SLOWeave: 1 s chat, 5 s long; DistServe: nới theo cỡ model). Chưa paper nào đặt ngưỡng cho trace agentic, nên dùng một quy tắc: **SLO_TTFT = 5 × TTFT không tải ở P90 của workload**, đo trên chính model đó (TTFT không tải: request chạy một mình, prefix cache đã ấm như trong trace, chunked mặc định của vLLM). Con số cụ thể tính sau khi đo TTFT không tải, **ghi vào đây trước khi chạy bất kỳ phép so policy nào**. Báo độ nhạy bằng cách nhân cả hai SLO với hệ số 0.5 / 1 / 2 (SLO scale của DistServe).
- **Metric phụ, bắt buộc có trong mọi bảng:** TTFT mean/P99 và TBT mean/P99 ở **70% và 90% goodput của baseline tốt nhất**; J/token; lưu lượng đọc expert (MoE).
- **So với baseline nào:** **baseline tốt nhất ở từng ô** (workload × SLO × model), sau khi tune theo `docs/03_MEASUREMENT.md` §7 (chunked quét chunk, Layered quét `N_lg`, SLOWeave tune δ). Tỉ số với riêng Layered vẫn báo để tách cơ chế pipeline khỏi cơ chế chunk theo operator.
- **Engine:** mọi phép so và số headline trong vLLM 0.30 (xem §4.3). Fork layered-prefill chỉ dùng để kiểm chứng bản Layered cài lại, cho bảng tham khảo khác engine, và làm phương án lui ở cổng cứng.
- **Cổng G1a, G1b, G2 đo bằng goodput** (trong simulator ở bước 2–4, trên engine từ bước 8), ở cùng định nghĩa SLO trên. Ngưỡng của các cổng giữ nguyên; đây là làm rõ đại lượng trước khi có số, không phải sửa tiêu chí.

### 4.5 Ablation

(a) bỏ phụ thuộc vị trí; (b) bỏ stagger; (c) k cố định so với k từ oracle; (d) 2 nhóm (FA vs phi-FA) so với 3 nhóm; (e) tỉ lệ attention:GDN:MoE; (f) prefill-only node (kỳ vọng không gain).

### 4.6 Hình cốt lõi

1. Chi phí mỗi token theo chunk size, một đường mỗi context length, một panel mỗi operator.
2. Chunk tối ưu c*(t) theo context: attention giảm ~1/t, GDN và MoE phẳng.
3. **Bản đồ regime**: gain theo (họ kiến trúc × context × Δ token mới). Đây là hình "ăn tiền".
4. TTFT theo context ở P99 TBT cố định, HyPrefill so với ba baseline.

---

## 5. Cổng quyết định

| Cổng | Sau bước | Tiêu chí | Nếu trượt |
|---|---|---|---|
| **G1a** | 3 | Layered / Sarathi ≥ 1.20× trên model hybrid (pipeline theo chiều sâu có đáng không) | Chuyển phương án lui số một |
| **G1b** | 3 | HyPrefill / Layered ≥ 1.25× ở ít nhất một chế độ (đóng góp riêng của HyPrefill) | < 1.10× mọi chế độ → bài co thành "Layered Prefill cho hybrid" hoặc chuyển phương án lui |
| **G1c** | 2 | Kernel indexer thật cấp phát buffer c·t theo mỗi lần gọi (chế độ bộ nhớ có thật) | Chế độ bộ nhớ không tồn tại → G1b gần như chắc trượt |
| **G2** | 4 | HyPrefill-oracle > SLOWeave ≥ 10% trên long-context (simulator) | Thu hẹp claim về long-context-only |
| **G3** | 7 | Overhead buffer + stagger ≤ 50% oracle gain | Paper measurement + oracle, nộp SIGMETRICS |
| **HARD** | 8 | Có số end-to-end trong vLLM (HyPrefill, Layered k = 1, chunked) | Phương án lui trên fork nanovllm nếu còn thời gian; nếu không, bài cost model + simulator đã kiểm chứng |
| **G4** | 10 | Prototype khớp simulator ±15% | Báo cáo sai lệch như một finding |
| **G5** | 13 | Đủ hình cho paper | Chọn venue cuối |


**G1a và G1b quyết sau bước 3 (cập nhật 29/09/2026, trước khi có số goodput):** hai cổng đo bằng goodput, cần mô phỏng mức request đã khớp hệ thật; oracle và mô phỏng dòng token ở bước 2 chỉ là cảnh báo sớm. G1c vẫn quyết ở bước 2. Ngưỡng giữ nguyên.

**Vì sao tách G1 (cập nhật 24/09/2026).** Cost model ở §2.2 chỉ mô hình hoá chunk theo operator, không có **pipeline theo chiều sâu** — tức cho một batch trải qua nhiều iteration trên đường đi xuống các layer, ý tưởng cốt lõi của Layered Prefill. Một mô phỏng dòng token (`sim/hyprefill_sim.js`, hằng số minh hoạ, chỉ là giả thuyết) gợi ý: khi attention bị giới hạn bởi **thời gian**, pipeline mang phần lớn gain và chunk theo operator chỉ thêm 0–10%; khi attention bị giới hạn bởi **bộ nhớ** (buffer indexer theo mỗi lần gọi), pipeline không gỡ được và chunk theo operator trở thành cơ chế chính. Lý do: budget TBT là ràng buộc mỗi iteration, còn buffer indexer là ràng buộc mỗi lần gọi. G1 cũ đo gộp hai cơ chế nên có thể đạt trong khi đóng góp riêng của HyPrefill bằng không. Claim trung tâm và abstract sẽ viết lại **sau** khi có số đo G1, không phải bây giờ. Giải thích đầy đủ: bài giảng `docs/00_FOUNDATIONS.html` mục 9.

**Câu hỏi độ bền bắt buộc trả lời ở G1:** trên Qwen3.8-Flash-Next (sparse attention), gain còn bao nhiêu? Nếu < 1.10 ở mọi t ≤ 256K thì claim thu hẹp về họ full-attention hybrid và nộp sớm.

---

## 6. Rủi ro

| Rủi ro | Xác suất | Đỡ |
|---|---|---|
| Gain nhỏ ở context ≤ 8K vì SLOWeave đã đóng gap | Cao | Định vị long-context (≥ 32K); báo cáo trung thực vùng không thắng |
| Reviewer: "chỉ là tuning" | Trung bình | Abstraction "min over operators" với ba loại ràng buộc; ablation HyPrefill-static thua HyPrefill-dynamic |
| Overhead buffer (COREY failure mode) | Trung bình | Cost model offline, không ước lượng runtime; đo overhead bước 6, trước khi đầu tư |
| **Engineering trong vLLM nặng hơn dự kiến** | **Cao** | Một hạ tầng chạy theo nhóm layer dùng chung cho Layered và HyPrefill; cổng cứng bước 8 với phương án lui trên fork nanovllm |
| Engine hybrid còn bug (SGLang #39342, vLLM #54076) | Trung bình | Tắt prefix cache trong eval chính; dùng Qwen3-Next làm anchor |
| Sparse attention làm yếu cơ chế 1/t | Trung bình | Khung ba loại ràng buộc; bản đồ regime; nói thẳng giới hạn |
| **Xu hướng chuyển chi phí prefill vào kiến trúc** (sparse attention; CED của DeepSeek V4.1; YOCO của HySparse2, arXiv 2609.26368). Ở kiến trúc thoát prefill sớm, sparse attention bị bỏ qua khi prefill nên chế độ indexer bị giới hạn bộ nhớ không xảy ra | Trung bình, về dài hạn | Giới hạn phạm vi rõ ràng vào hybrid kiểu stack thường (Qwen3.8-Flash-Next, GLM-5.3-Flash — sparse attention chạy trong prefill); nêu YOCO/CED trong Limitations; G1c kiểm chế độ bộ nhớ trên model thật. HySparse2 chưa công bố weights nên chưa phải model đánh giá |

**Loại rủi ro cần nhớ:** rủi ro độ mới chết lúc nộp, rủi ro kỹ thuật chết trước cả lúc nộp. Phần engineering (bước 5–10) là phần đáng lo nhất.

---

## 7. Phương án lui

| Ưu tiên | Phương án | Vì sao |
|---|---|---|
| 1 | FP4 precision residency (`docs/04_REVIEW_fp4_hopper_fallback.md` §5.3) | Số liệu có sẵn từ bước 1–2, không cần hạ tầng mới; nhanh nhất, đủ cho workshop paper |
| 2 | VeriPrefill | Điểm cao nhất nhưng phải dựng witness + conformal từ đầu |
| 3 | StateGraph | Hợp xu hướng sparse nhất nhưng cần DeepSeek V4.1 trên 4–8 GPU; lâu nhất |

---

## 8. Venue và deadline (cập nhật 24/09/2026)

**Ràng buộc:** MLSys không nằm trong danh sách xếp hạng của trường nên không được hỗ trợ kinh phí dự hội nghị. Vì vậy loại MLSys khỏi mục tiêu chính dù đó là venue hợp khẩu vị nhất về nội dung.

**Tiền lệ quyết định hướng ICML.** "Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving" (arXiv 2603.13358) ghi rõ **"Accepted at ICML 2026"**. Đây là bài lập lịch prefill cho serving nhiều lượt, cùng họ vấn đề và cùng chỉ số TTFT, nhóm tác giả có người của SGLang. ICML có khẩu vị cho loại bài này. Thêm nữa, PPD chính là anchor related work của HyPrefill về phạm vi append-prefill — một bài cho cả hai thứ.

| Venue | Deadline | Trạng thái | Vai trò |
|---|---|---|---|
| **ICML 2027** | ~28/01/2027 | Estimated (trang CFP chưa lên, suy từ 2026 — kiểm lại trước bước 14) | **Mục tiêu chính** |
| SIGMETRICS 2027 winter | 11/01/2027 | Confirmed | Bản measurement nếu cổng cứng bước 8 trượt |
| NeurIPS 2027 | ~05/2027 | Estimated | Nộp lại sau ICML kèm phản biện; +4 tháng cũng giảm rủi ro engineering |
| SOSP 2027 | ~01/04/2027 | Estimated | Nếu muốn venue hệ thống và có hệ thống đầy đủ |
| ASPLOS 2028 cycle 1 | ~15/04/2027 | Pattern | Dự phòng |
| ~~ATC 2027~~ | ~01/2027 | **Loại** | USENIX ngừng ATC sau 2025, ACM SIGOPS khôi phục; danh sách xếp hạng nhiều khả năng chưa cập nhật bản mới → rủi ro không được hỗ trợ |
| ~~MLSys 2027/2028~~ | — | **Loại** | Unranked, không hỗ trợ kinh phí |

**Xung đột lịch.** SIGMETRICS (11/01) chỉ cách ICML (~28/01) hai tuần rưỡi, không nộp cùng nội dung cho cả hai. Quy tắc chọn gắn vào cổng sẵn có:

- Cổng cứng bước 8 **đạt** → eval xong bước 13 → nộp **ICML** bản đầy đủ có prototype.
- Cổng cứng bước 8 **trượt** → nộp **SIGMETRICS 11/01** bản đo lường + oracle + simulator.

### 8.1 Bốn điều phải đổi khi viết cho ICML thay vì venue hệ thống

1. **Mở bài bằng tính đúng đắn.** HyPrefill chỉ đổi lịch, không đổi phép toán; output giống baseline (kiểm chứng 200 prompt greedy, bước 13). Ở venue hệ thống đây là mục phụ; ở ICML đây là điểm mạnh nhất, phải đưa lên sớm.
2. **Giải thích chunked prefill từ đầu.** Reviewer ICML không biết Sarathi-Serve. Thêm ~nửa trang background mà venue hệ thống không cần. Cắt bớt chi tiết engine để bù chỗ.
3. **Đổi trọng tâm từ SLO sang analysis.** Ba loại ràng buộc và `c*_FA(t) ≈ P/(α·t)` đọc như phân tích, đúng khẩu vị ICML. Bộ máy P99 TBT SLO để xuống Evaluation.
4. **Đặt cạnh PPD ngay trong Intro.** PPD quyết định append-prefill chạy **ở đâu**; HyPrefill quyết định khi đã ở đó thì **chia khối thế nào**. Bổ sung nhau, không cạnh tranh.

## 9. Cấu trúc tài liệu

```
HyPrefill/
├── PROPOSAL.md              ← file này
├── README.md                điều hướng
├── LOG.md                   nhật ký hằng ngày
├── docs/
│   ├── 00_FOUNDATIONS.html  bài giảng nền tảng + 9 paper (mở bằng trình duyệt)
│   ├── 01_READING_LIST.md   link PDF để in
│   ├── 02_RELATED_WORK.md   số liệu đã xác minh, nguyên liệu Related Work
│   ├── 03_MEASUREMENT.md    kỷ luật đo lường
│   ├── 04_REVIEW_fp4_hopper_fallback.md
│   ├── 05_DEADLINES_2027.md
│   └── lecture_notes/       ghi chú đầy đủ 9 paper
├── plan/00_SETUP.md … 16_SUBMIT.md
├── bench/   sim/   results/   figures/
```
