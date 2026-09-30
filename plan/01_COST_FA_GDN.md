# Bước 1 — Cost curve — attention và GDN

**Mục tiêu:** Đo cost_FA(c, t) và cost_GDN(c, t) bằng micro-benchmark kernel, và xác nhận thực nghiệm rằng GDN không phụ thuộc t.

---

## 1. Đầu ra bắt buộc

> Checklist mở lại 2026-09-29: đo lại theo chuẩn mới (1980 MHz, KV paged, state GDN theo config). Lượt 1830 đã đạt các mục cũ; kết quả ở mục KẾT QUẢ.

- [x] `bench/op_cost.py` chạy được, xuất CSV thô
- [x] Bảng cost_FA(c, t) cho c ∈ {64…8192}, t ∈ {0, 4K, 16K, 32K, 64K, 128K, 256K}, **KV paged** theo block size vLLM chọn
- [x] Bảng cost_GDN(c, t) cùng dải, state theo `mamba_ssm_dtype`
- [x] Bảng Dense (`dense_attn`, `dense_gdn`, `gdn_conv`, `dense_mlp`) theo c
- [x] **Chi phí all-reduce TP** theo số token (Qwen3-Next TP2): vLLM all-reduce 2 lần mỗi layer, chưa có trong bảng
- [x] **Kiểm chứng then chốt**: GDN không phụ thuộc t. Tiêu chí: lệch có hệ thống giữa nhóm t ≥ 128K và t ≤ 16K < 5% ở mọi c, và báo điểm lạc (> 10% so với median theo t) nếu có. (Tiêu chí max−min cũ quá nhạy với một điểm lạc; lượt 1830 đạt cả hai sau khi đổi cách đo)
- [ ] **Kiểm chứng tổng (§2.5), tiêu chí chính của bước** *(chưa đạt: c ≥ 1024 đạt, c = 512 và các dòng t = 0 chưa, xem KẾT QUẢ)*: chi phí một iteration dự đoán khớp prefill thật của vLLM trong ±15% ở mọi c ≥ 512, t ∈ {0, 16K, 32K, 64K}
- [x] Hình 1: thời gian mỗi token theo c, một đường mỗi t, một panel mỗi operator

## 2. Việc chi tiết

### 2.1 Kernel gọi trực tiếp, không qua vLLM

| Group | Kernel | Tham số |
|---|---|---|
| FA | `vllm.vllm_flash_attn.flash_attn_varlen_func`, `fa_version=3` (vLLM chọn FA3 trên H200), q_len = c, kv_len = t + c, causal | c ∈ {64,128,256,512,1024,2048,4096,8192}; t ∈ {0, 4K, 16K, 32K, 64K, 128K, 256K} |
| GDN | `fi_chunk_gated_delta_rule` của vLLM (FlashInfer, vLLM chọn mặc định trên H200; `--gdn-backend triton` để so với FLA), seq_len = c, **`initial_state` ≠ None** | cùng c; t ảnh hưởng qua initial_state (kỳ vọng phẳng) |
| Dense | Q/K/V/O projection, norm, gate | cùng c, để hoàn thiện tổng |

Shape lấy đúng từ `config.json` của từng model (bước 0). Nhân với số layer mỗi loại để ra cost toàn model.

### 2.2 Tại sao `initial_state` quan trọng

Đây là điểm dễ sai nhất của bước này. Nếu gọi `chunk_gated_delta_rule` không truyền `initial_state`, bạn đang đo trường hợp prefill từ đầu, không phải trường hợp chunked prefill có state tích luỹ. Sinh `initial_state` bằng cách chạy trước t token rồi lấy state ra, hoặc tạo tensor ngẫu nhiên đúng shape `d_k × d_v` mỗi head (đủ cho mục đích đo thời gian).

### 2.3 Chú ý c không chia hết 64

Kernel pad chunk cuối, nên c = 65 tốn gần bằng c = 128 (⌈65/64⌉ = 2). Đo cả c lẻ để thấy bậc thang này; scheduler sau sẽ chọn c là bội của 64. Với FA3, bậc thang theo tile 128 (lượt đo 28/09: c = 129 đắt hơn c = 128 rõ rệt), nên chunk attention chọn bội 128.

### 2.4 Tách thành phần của cost_GDN

Fit `cost_GDN(c) = a + b·⌈c/64⌉`:
- `a` = overhead cố định mỗi lần gọi → ngoại suy từ c rất nhỏ về c = 0
- `b` = chi phí biên mỗi chunk-64, gộp intra O(C²d) và inter O(Cd²)

Đổi `d_k`, `d_v`, `num_heads` để tách tỉ trọng intra so với inter. **Lượt 1830 (sơ bộ):** fit theo số phép tính cho hệ số inter âm, tức kernel không theo mô hình phép tính. Chưa kết luận: đo lại ở 1980 bằng backend triton (FlashInfer chỉ nhận d_k = d_v), kiểm lại cách fit. Mục này **không** chặn việc sang bước 2 vì cost model tra bảng đo trực tiếp, không dùng công thức.

### 2.5 Kiểm chứng tổng

So tổng cost dự đoán với một forward pass thật của model **qua vLLM** (cùng bộ kernel với bảng cost: FA3, GDN FlashInfer, MoE của vLLM) ở cùng (c, t), cùng TP. Không so với HuggingFace: HF dùng kernel khác (torch/fla) nên sai số sẽ đến từ kernel chứ không phải cost model. **Sai số ≤ 15% là chấp nhận.** Nếu lệch hơn, tìm thành phần bỏ sót (thường là norm, gate, hoặc residual).

**Bổ sung sau lượt 1830 (giả thuyết, cần kiểm lại):** phần hụt 22–35% ở c = 512 trùng với việc GPU chờ CPU (vLLM chạy attention và lõi GDN eager giữa các mảnh CUDA graph); ở c ≥ 1024 hụt 7–12%, một phần do KV paged. Việc cần làm:
1. Đo lại bằng bảng KV paged.
2. Profile vLLM ở nhiều c (512 → 8192) để đo **thời gian CPU mỗi iteration** và thời gian GPU rỗi; nếu đúng là CPU, cost model của một iteration = max(GPU, CPU) hoặc dạng chồng lấp đo được, không phải chỉ tổng kernel.
3. Kiểm có tuỳ chọn hợp lệ nào giảm chi phí CPU (ví dụ lớp log API của FlashInfer); nếu có, áp dụng cho **mọi** policy.
4. Chạy §2.5 cho cả Qwen3-Next TP2 (cần bảng MoE bước 2 và all-reduce), không chỉ Qwen3.8-27B.
5. **Chạy §2.5 có batch decode song song** (8 / 32 / 64 request đang decode trong lúc prefill). Chỉ phép đo này phân biệt được hai mô hình cost của `bench/oracle.py --batch-model additive|mixed`; kiểm thử trên bảng 1830 cho thấy hai mô hình cho gain rất khác nhau (additive 1.45–1.83, mixed ≈ 1.0 ở SLO 50 ms), nên **không dùng oracle trước khi biết mô hình nào khớp vLLM**. Cần thêm chế độ decode song song vào `bench/validate_forward.py`.

Công cụ cho các việc trên: `bench/profile_vllm_step.py --sweep` (thời gian thực mỗi step không profiler, so với thời gian GPU bận), `bench/allreduce_cost.py` (all-reduce qua đường của vLLM, chạy bằng `torchrun --nproc-per-node 2`).

## 3. Tiêu chí kiểm tra trước khi sang bước 2

- cost_FA/token gần phẳng theo c nhưng dâng rõ theo t → đúng dự đoán
- cost_GDN/token giảm theo c và **không đổi theo t** → đúng dự đoán
- Nếu GDN *có* phụ thuộc t đáng kể, dừng lại và tìm nguyên nhân (nhiều khả năng do cách sinh initial_state hoặc do cache allocator), vì toàn bộ luận điểm dựa vào tính chất này.


---

## KẾT QUẢ

> Điền 2026-09-29, chuẩn đo 1980 MHz, KV paged, state GDN theo config, GPU 6 và 7 (mỗi card một lượt). Số lượt 1830 cũ ở `~/hyprefill_data/clk1830/`. **Bước chưa xong:** §2.5 đạt ở c ≥ 1024, chưa đạt ở c = 512 và ở các dòng t = 0.

### R1. Số liệu chính

Per GPU, mỗi layer, median 2 card (`results/step01/cost_tables/`).

| Đại lượng | Qwen3-Next TP2 | Qwen3.8-27B TP1 | Ghi chú |
|---|---|---|---|
| Độ lặp lại giữa hai card (FA, GDN, Dense) | median 0.17%, 100% dòng không chạm trần < 3% | median 0.19%, 100% | Dòng chạm trần 700 W: median 0.9%, tối đa 9% |
| GDN FlashInfer: a + b·⌈c/64⌉ | 56 µs + 1.00 µs | 58 µs + 2.64 µs | Chi phí cố định trội tới c ≈ 3.6K / 1.4K |
| GDN theo t | lệch có hệ thống ≤ 0.2% | ≤ 0.3% | Kiểm chứng then chốt đạt |
| GDN triton: a + b·⌈c/64⌉ | 26 µs + 3.54 µs | 26 µs + 7.83 µs | Rẻ hơn FlashInfer khi c ≲ 470 / 210 |
| FA3 ở t = 256K, c ≥ 256 | 3.0–3.3 µs/token | 10.1–14.7 µs/token | Tăng tuyến tính theo t |
| FA3 bậc thang tile 128 | | c = 128: 10.4, c = 129: 17.6 µs/token (t = 256K) | Chunk attention là bội 128 |
| All-reduce TP2 (hidden 2048) | 3.2 µs khi ≤ 32 token; 39 µs ở 2048; 141 µs ở 8192 token (~240 GB/s) | — | 2 lần mỗi layer: ~3.7 ms mỗi iteration ở c = 2048 |
| §2.5 (Qwen3.8-27B), dự đoán so với prefill thật | | c = 8192: −3.7…−4.1%; 4096: −3.8…−5.1%; 2048: −7.8…−8.7%; 1024: −8.2…−9.1%; **512: −15…−32%** (t = 16K–64K) | Tiêu chí ±15%: đạt ở c ≥ 1024 |
| §2.5 tách phần hụt (cùng cấu hình) | | GPU bận mỗi step dài hơn bảng 2–10%; GPU chờ CPU 36.7 ms/step (44%) ở c = 512, t = 16K, còn ≤ 10% ở c ≥ 1024 hoặc t = 64K | `results/step01/2026-09-29_vllm_step_sweep_Qwen3.8-27B_tp1/` |
| §2.5 có decode song song (Qwen3.8-27B, bd = 8 / 32 / 64, 2026-09-30) | | Chi phí mà decode thêm vào, c ≥ 2048: đo 1608 / 731 ms (c = 2048 / 8192, tổng 6 cấu hình); mô hình `mixed` dự đoán −14% / −10% (từng cấu hình −4…−19%, bd = 8 lệch tới +26% vì số nhỏ); mô hình `additive` +99% / +68%. c = 512: không mô hình nào khớp (giới hạn bởi CPU) | **Cost model dùng mô hình batch trộn**: GEMM/MoE của decode và prefill chung một lần gọi. `results/step01/2026-09-30_validate_forward/decode_extra_cost.csv`. Chú ý: mỗi step prefill chỉ có c − bd token (decode chiếm budget) |
| §2.4 | FlashInfer (chỉ nhận d_k = d_v): mỗi chunk 64 token 0.54 / 0.81 / 1.59 µs ở 8 / 16 / 32 head v | | Triton: theo d_k 2.6 / 3.7 / 5.1 µs (d_k = 64 / 128 / 256), theo d_v 3.4 / 3.7 / 4.5 µs; có phần cố định lớn |

### R2. Hình sinh ra

| File | Nội dung | Dùng cho hình nào của paper |
|---|---|---|
| `figures/step01_fig1.png` | µs/token theo c, một đường mỗi t, panel FA và GDN, Qwen3-Next TP2 và Qwen3.8-27B TP1 | Hình 1 (bản nháp) |

### R3. Khác dự đoán / bất ngờ

- Đo KV paged làm sai số §2.5 ở c ≥ 1024 giảm khoảng một nửa so với lượt 1830 (−7…−12% còn −4…−9%).
- Ở c = 512, t = 16K, prefill trên vLLM bị giới hạn bởi CPU: GPU chờ 44% thời gian mỗi step, trong khi phần GPU của bảng đúng tới 2%. Ở t = 64K cùng c thì gần như hết, vì mỗi step có nhiều việc GPU hơn để che CPU (vLLM mặc định async scheduling).
- Phần GPU của engine luôn dài hơn tổng các op trong bảng 2–10%, lớn hơn ở t dài. Chưa rõ kernel nào; cần so theo từng loại kernel giữa trace và bảng.
- GDN triton của Qwen3-Next lệch có hệ thống giữa hai card (1–7%, không đổi theo t, lớn nhất ở c = 1024); FA, GDN FlashInfer và triton của Qwen3.8-27B khớp ~0.2%. Nghi autotune của Triton chọn cấu hình khác nhau ở hai tiến trình. Triton không nằm trong cost model.
- Các dòng t = 0 của §2.5 lệch +9…+104% là do cách đo: trừ thời gian của prompt 16 token cũng trừ luôn phần đọc weight, là phần chính của GEMM ở c nhỏ.
- All-reduce TP2 là chi phí theo token (không khấu hao), đáng kể và chưa có trong bảng cũ.
- Bản đầu của `profile_vllm_step.py --sweep` cho số sai: trong vLLM V1 engine core chạy ở tiến trình riêng nên `LLMEngine.step()` không ứng với một step của scheduler. Đã sửa (tổng thời gian request trừ tổng thời gian GPU bận, chia số step).

### R4. Quyết định rút ra

- Bảng cost chính thức: median 2 card, KV paged, 1980 MHz; cờ `power_capped` giữ lại.
- Cost model của một iteration cần thêm phần CPU cho chunk nhỏ, và kiểm mô hình chồng lấp GPU/CPU (async scheduling) trước khi dùng cho c = 512.
- Cộng all-reduce (2 lần mỗi layer) cho model chạy TP > 1.
- Oracle và mô phỏng dùng `--batch-model mixed` (đã kiểm chứng ở c ≥ 2048); bỏ mô hình cộng rời.

### R5. Việc chuyển sang bước sau

- Sửa cách đo t = 0 của §2.5 (không trừ prompt 16 token; dùng thời gian GPU bận hoặc trừ đúng phần cố định mỗi request) rồi chạy lại các dòng t = 0.
- Mô hình hoá phần CPU mỗi iteration từ `vllm_step_sweep`, kiểm lại c = 512.
- So theo loại kernel giữa trace engine và bảng để tìm phần GPU 2–10% còn thiếu.
- §2.5 cho Qwen3-Next TP2 (cần bảng MoE bước 2 + all-reduce; kiểm vLLM có gộp all-reduce với RMSNorm không) và §2.5 có decode song song (cần bảng decode bước 2).

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
