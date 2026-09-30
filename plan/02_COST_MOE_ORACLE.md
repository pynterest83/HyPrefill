# Bước 2 — Cost curve MoE, sparse attention, oracle bound — CỔNG G1

**Mục tiêu:** Hoàn tất bảng chi phí cho mọi operator (MoE theo batch trộn decode + prefill), trả lời G1c, và tính oracle như **cảnh báo sớm**. Quyết định G1a/G1b chuyển sang bước 3 (sửa 2026-09-29, xem §4).
**Cổng:** **G1 tách ba phần (cập nhật 24/09/2026), viết kết luận trước khi nhìn số.** **G1a** — Layered / Sarathi ≥ 1.20× trên model hybrid. **G1b** — HyPrefill / Layered ≥ 1.25× ở ít nhất một chế độ thực tế. **G1c** — kernel indexer thật có cấp phát buffer c·t theo mỗi lần gọi. Chi tiết và hành động khi trượt ở §3.

---

## 1. Đầu ra bắt buộc

- [ ] Bảng cost_MoE **theo batch trộn**: `cost_MoE(n_decode, c)` khi decode và prefill đi chung một lần gọi (như vLLM chạy), kèm số expert được chạm bởi decode, bởi prefill, và bởi hợp của hai
- [ ] Bảng cost_MoE(c) riêng prefill + số expert được chạm theo c, đối chiếu coupon-collector
- [ ] *(tuỳ chọn, chỉ cho phương án lui FP4)* Bảng cost_MoE(c, path) cho các đường kernel precision + phân bố M_e theo expert
- [ ] Bảng cost_QSA(c, t) **tách riêng thời gian và bộ nhớ đỉnh** của indexer trên Qwen3.8-Flash-Next
- [ ] `bench/oracle.py` tính r_u, r_d, gain với cost model batch trộn (§2.4); heatmap gain theo (t, B). **Chỉ là cảnh báo sớm**, không dùng để quyết G1
- [ ] Hình 2: c*(t) theo t cho từng operator
- [ ] Hình 3: bản đồ regime gain theo (họ kiến trúc × context)
- [ ] **(G1c, làm đầu tiên, ~nửa ngày)** Kiểm kernel indexer của Qwen3.8-Flash-Next: có cấp phát buffer logits c·t theo mỗi lần gọi, hay streaming top-k với workspace cố định? **Lượt 1830 (sơ bộ):** buffer tỉ lệ c·t nhưng vLLM chia query để buffer ≤ `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 MB), đỉnh ~1 GiB. Chưa kết luận: đo lại trên model thật khi đã tải (không chỉ tensor giả lập), kiểm cả buffer chunked-scan của GDN/KDA (vLLM #54775), và xem giới hạn 512 MB có làm tăng thời gian indexer ở c lớn không (bộ nhớ chuyển thành thời gian)
- [ ] **(+1–2 ngày)** Chạy mô phỏng dòng token (`bench/sim_tokenflow.py`, bản Python của `sim/hyprefill_sim.js`) với số đo thật, tách gain của **pipeline theo chiều sâu** (Layered, k = 1) khỏi gain của **chunk theo operator** (HyPrefill, k ≥ 2). **Chỉ là cảnh báo sớm:** lượt 1830 cho thấy nó đo độ trễ một request chứ không phải goodput, và chưa khớp demo thật; phải kiểm lại trên Qwen3-30B-A3B trước khi đọc số
- [ ] **(trọng tâm)** Đo tập expert bị chạm bởi batch decode so với bởi chunk prefill ở batch decode 8 / 32 / 64 / 128, trên trace thật (không chỉ router ngẫu nhiên): phần khấu hao MoE của HyPrefill phụ thuộc vào con số này. Công cụ: `bench/moe_routing_dump.py` (chạy vLLM một lần với `enable_return_routed_experts`, lưu expert của từng token), `bench/moe_overlap.py` (phân tích offline), và op `moe_mixed` của `bench/op_cost.py --routing <dump>` (thời gian MoE của batch trộn với routing thật)
- [ ] **Một trang kết quả gửi advisor**

## 2. Việc chi tiết

### 2.1 MoE

Đo toàn bộ MoE block (router + fused experts, backend giống vLLM) với c token. Ghi thêm **số expert được chạm** để đối chiếu với công thức coupon-collector:

```
E_touched(c) ≈ E · [1 − (1 − k/E)^c]
Qwen3-Next (E=512, k=10): c=64 → ~360; c=256 → ~505; c≥512 → gần như tất cả
```

Nếu số đo khớp công thức, đó là một câu trong Motivation. Nếu lệch (routing lệch, không đồng đều), càng thú vị — ghi lại.

### 2.2 Chiều precision (+2 ngày)

Lặp lại phép đo MoE với từng đường kernel:

| Path | Cách bật |
|---|---|
| Marlin W4A16 | mặc định cũ |
| Humming W4AFP8 | `--moe-backend humming` (mặc định SM90 hiện tại) |
| FlashInfer SM90 MXFP4×FP8 | `--moe-backend flashinfer_cutlass_humming` |
| FP8 materialized | `VLLM_DSV4_FP4_DEQUANT=1` |

Dump thêm phân bố `M_e` theo expert ở vài mức tải. Hai mục đích: (a) cost model biết `cost_MoE(c, path)`, (b) **dữ liệu dự phòng nếu G1 trượt** — xem `docs/04_REVIEW_fp4_hopper_fallback.md` §5.3.

### 2.3 Sparse attention — phần quyết định độ bền

Trên Qwen3.8-Flash-Next, đo layer QSA tách **hai** thành phần:

```
cost_indexer(c, t) ≈ α_idx · c · t        ← thời gian
mem_indexer(c, t)  ≈ c · (t / r) · H_idx · 4B   ← bộ nhớ đỉnh, đo bằng torch.cuda.max_memory_allocated()
cost_topk(c)       ≈ κ · c · k            ← không phụ thuộc t
```

Config thật (`results/step00/model_configs.md`): `H_idx = 4`, `d_idx = 128`, `r = indexer_compress_ratio = 4`. Mô phỏng đang giả định `H_idx = 64`, không nén, tức buffer lớn hơn khoảng 64 lần; công thức trên chỉ là giả định, số thật phải đến từ G1c.

Câu hỏi cần trả lời: ở t lớn, chunk của layer QSA bị kẹp bởi **thời gian** hay bởi **bộ nhớ**? Nếu là bộ nhớ, đó là bằng chứng trực tiếp cho loại ràng buộc thứ hai và là một đóng góp riêng. Đối chiếu với vLLM issue #56457 (buffer tăng 10.24 MB × chỉ số chunk).

### 2.4 Oracle

**Cost model batch trộn (sửa 2026-09-29, giả thuyết cần đo):** trong vLLM, decode và chunk prefill đi chung một lần gọi GEMM và MoE, nên phần đọc weight chỉ trả một lần; attention và GDN của decode và prefill là kernel riêng. Công thức cộng rời `D + Σ_g cost_g(c)` bên dưới có thể tính trùng phần đọc weight. Oracle phải dùng `cost_MoE(n_decode, c)` và `cost_dense(n_decode + c)` đo được ở §2.1, giữ `cost_FA`, `cost_GDN` tách riêng. So cả hai cách (cộng rời, batch trộn) để biết chênh bao nhiêu.

`bench/oracle.py` đọc CSV cost, không cần GPU, chạy vài giây:

```
P = B − D                         D đo với batch decode 8 / 32 / 64
r_u(t) = max c  s.t.  Σ_g cost_g(c,t) ≤ P  và  mem_g(c,t) ≤ M_g
r_d(t) = max c  s.t.  cost_FA(c,t) + Σ_{g≠FA} cost_g(k_g c)/k_g ≤ P   (khấu hao)
                 và  cost_FA(c,t) + max_{g≠FA} cost_g(k_g c)   ≤ P   (đỉnh)
                 và  mem_g(k_g c, t) ≤ M_g
Gain(t) = r_d(t) / r_u(t)          quét k_g ∈ {1,2,4,8,16}
```


### 2.5 Tách hai cơ chế — phép đo thêm (cập nhật 24/09/2026)

Công thức oracle ở §2.4 chỉ có chunk theo operator, **không có pipeline theo chiều sâu**. Mô phỏng (`sim/hyprefill_sim.js`, bài giảng mục 9) gợi ý phần lớn gain ở chế độ thời gian đến từ pipeline, tức ý tưởng của Layered Prefill, còn chunk theo operator chỉ quan trọng khi attention bị giới hạn bởi bộ nhớ. Mô phỏng dùng hằng số minh hoạ nên **chỉ là giả thuyết**; bước này đo để xác nhận hoặc bác.

**Việc 1 — G1c, làm trước (~nửa ngày).** Đọc mã kernel indexer QSA trong vLLM (`vllm/.../qsa*` hoặc tương đương, liên quan issue #56457, PR #56500). Đo `torch.cuda.max_memory_allocated()` khi gọi một attention layer QSA với c ∈ {256, 512, 1K, 2K, 4K} và t ∈ {32K, 128K, 256K}. Nếu bộ nhớ đỉnh tăng tỉ lệ c·t → chế độ bộ nhớ có thật. Nếu phẳng (workspace cố định) → không có.

**Việc 2 — thay hằng số minh hoạ bằng số đo.** Trong `sim/hyprefill_sim.js`, thay các hàm `FAl`, `Gl`, `Ml` và `cap` bằng bảng đo ở §2.1–2.3 (per-sublayer, không phải aggregate). Thay `LAYOUT` bằng bố cục thật từ `config.json`.

**Việc 3 — chạy ba chính sách** cho mỗi model và mỗi chế độ, lấy ba tỉ số. Mỗi chính sách được tune công bằng trước khi so (`docs/03_MEASUREMENT.md` §7): Sarathi quét chunk, Layered quét số nhóm layer, HyPrefill quét k. Không so HyPrefill đã tune với baseline chạy tham số mặc định.
```
Layered / Sarathi        → G1a   (pipeline có đáng trên hybrid không)
HyPrefill / Layered      → G1b   (đóng góp riêng của HyPrefill)
HyPrefill / Sarathi      → chỉ để tham khảo, KHÔNG dùng làm headline
```

**Việc 4 — kiểm lo ngại MoE.** Với batch decode 32/64/128 trên Qwen3-Next, dump tập expert bị chạm bởi decode và bởi một chunk prefill c. Nếu decode đã chạm > 90% expert thì phần tiết kiệm đọc trọng số của HyPrefill trên node decode bận là nhỏ; ghi nhận và điều chỉnh claim.

## 3. Cổng G1 — viết kết luận trước khi nhìn số

> **Sửa 2026-09-29, trước khi có số goodput:** G1a và G1b đo bằng **tỉ số goodput**, nên không quyết được bằng oracle (tỉ số chunk) hay mô phỏng dòng token (số iteration). Hai cổng này **quyết ở cuối bước 3**, bằng mô phỏng mức request đã khớp với hệ thật. Bước 2 chỉ quyết G1c và đưa ra cảnh báo sớm. Tiêu chí và ngưỡng giữ nguyên.

**Không sửa tiêu chí sau khi thấy kết quả.**

| Cổng | Đo gì | Đạt | Trượt thì |
|---|---|---|---|
| **G1a** | Layered / Sarathi trên model hybrid, **tỉ số goodput** (PROPOSAL §4.4; chốt 2026-09-29, trước khi có số) | ≥ 1.20× | Cả hướng yếu → chuyển phương án lui số một (bài đo độ trung thực FP4) |
| **G1b** | HyPrefill / Layered, **tỉ số goodput**, ở ít nhất một chế độ thực tế | ≥ 1.25× | Nếu < 1.10× ở mọi chế độ: bài co lại thành "Layered Prefill cho hybrid" — nhỏ hơn nhiều; bàn với advisor có đi tiếp hay chuyển phương án lui |
| **G1c** | Kernel indexer cấp phát buffer c·t theo mỗi lần gọi | Có | Chế độ bộ nhớ không có trong thực tế → G1b gần như chắc trượt; kiểm G1b ở chế độ thời gian trước khi quyết |

Đọc kết hợp:

- **G1a đạt, G1b đạt** → đi tiếp đúng kế hoạch; headline là HyPrefill / Layered, model sparse (Qwen3.8-Flash-Next, GLM-5.3-Flash) thành model headline nếu G1b đạt ở chế độ bộ nhớ.
- **G1a đạt, G1b trượt** → đóng góp riêng của HyPrefill không đủ. Lựa chọn: viết "depth-pipelined prefill cho hybrid" (mở rộng Layered Prefill sang GDN, venue nhỏ hơn) hoặc chuyển phương án lui.
- **G1a trượt** → chuyển phương án lui.

Câu hỏi độ bền vẫn giữ: trên Qwen3.8-Flash-Next, gain còn bao nhiêu? Giờ trả lời riêng cho G1a và G1b.

---

## KẾT QUẢ

> **Để trống — điền khi làm xong bước này.**

### R1. Số liệu chính

| Đại lượng | Giá trị | Ghi chú |
|---|---|---|
|  |  |  |

### R2. Hình sinh ra

| File | Nội dung | Dùng cho hình nào của paper |
|---|---|---|
|  |  |  |

### R3. Khác dự đoán / bất ngờ

-

### R4. Quyết định rút ra

-

### R5. Việc chuyển sang bước sau

-

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
