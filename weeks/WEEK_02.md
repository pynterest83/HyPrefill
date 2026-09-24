# Tuần 02 — Cost curve MoE, sparse attention, oracle bound — CỔNG G1

**Ngày:** 29/09–05/10/2026
**Mục tiêu:** Hoàn tất bảng chi phí cho mọi operator, tính oracle gain, và quyết định đi tiếp hay chuyển hướng.
**Cổng:** **G1 — GO nếu oracle gain ≥ 1.25 ở t ≥ 32K với B ∈ {25, 50, 100} ms trên ≥ 2 model.** PIVOT nếu chỉ đạt ở t ≥ 128K (thu hẹp claim về long-context). KILL nếu < 1.15 ở mọi t ≤ 256K trên cả họ full-attention lẫn họ sparse → chuyển FP4 precision residency.

---

## 1. Đầu ra bắt buộc

- [ ] Bảng cost_MoE(c) + số expert được chạm theo c
- [ ] **(+2 ngày)** Bảng cost_MoE(c, path) cho 4 đường kernel + phân bố M_e theo expert
- [ ] Bảng cost_QSA(c, t) **tách riêng thời gian và bộ nhớ đỉnh** của indexer trên Qwen3.8-Flash-Next
- [ ] `bench/oracle.py` tính r_u, r_d, gain; heatmap gain theo (t, B)
- [ ] Hình 2: c*(t) theo t cho từng operator
- [ ] Hình 3: bản đồ regime gain theo (họ kiến trúc × context)
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
mem_indexer(c, t)  ≈ c · t · H_idx · 4B   ← bộ nhớ đỉnh, đo bằng torch.cuda.max_memory_allocated()
cost_topk(c)       ≈ κ · c · k            ← không phụ thuộc t
```

Câu hỏi cần trả lời: ở t lớn, chunk của layer QSA bị kẹp bởi **thời gian** hay bởi **bộ nhớ**? Nếu là bộ nhớ, đó là bằng chứng trực tiếp cho loại ràng buộc thứ hai và là một đóng góp riêng. Đối chiếu với vLLM issue #56457 (buffer tăng 10.24 MB × chỉ số chunk).

### 2.4 Oracle

`bench/oracle.py` đọc CSV cost, không cần GPU, chạy vài giây:

```
P = B − D                         D đo với batch decode 8 / 32 / 64
r_u(t) = max c  s.t.  Σ_g cost_g(c,t) ≤ P  và  mem_g(c,t) ≤ M_g
r_d(t) = max c  s.t.  cost_FA(c,t) + Σ_{g≠FA} cost_g(k_g c)/k_g ≤ P   (khấu hao)
                 và  cost_FA(c,t) + max_{g≠FA} cost_g(k_g c)   ≤ P   (đỉnh)
                 và  mem_g(k_g c, t) ≤ M_g
Gain(t) = r_d(t) / r_u(t)          quét k_g ∈ {1,2,4,8,16}
```

## 3. Cổng G1 — viết kết luận trước khi nhìn số

Tiêu chí ở đầu file. **Không sửa tiêu chí sau khi thấy kết quả.**

Câu hỏi độ bền bắt buộc trả lời: trên Qwen3.8-Flash-Next, gain còn bao nhiêu? Nếu < 1.10 ở mọi t ≤ 256K nhưng họ full-attention vẫn ≥ 1.25, claim thu hẹp về hybrid full-attention và paper nói thẳng sparse attention là giới hạn.


---

## KẾT QUẢ

> **Để trống — điền khi làm xong tuần này.**

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

### R5. Việc chuyển sang tuần sau

-

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
