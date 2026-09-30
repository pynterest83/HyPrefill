# Bước 3 — Simulator sự kiện rời rạc — CỔNG G1a, G1b

**Mục tiêu:** Dựng simulator iteration-level từ bảng cost đã đo, cài 5 policy, xác nhận simulator khớp hệ thật, rồi **quyết cổng G1a và G1b** bằng goodput (chuyển từ bước 2, sửa 2026-09-29).

---

## 1. Đầu ra bắt buộc

- [ ] `sim/` chạy được, đầu vào là bảng cost + trace request
- [ ] Năm policy cài xong: static (quét chunk), Layered Prefill (quét `N_lg`), SLOWeave (tune δ), HyPrefill-oracle, HyPrefill-static
- [ ] Simulator khớp phép đo một iteration thật của vLLM trong **±15%** (các lượt §2.5 bước 1, cả Qwen3.8-27B và Qwen3-Next TP2, có batch decode)
- [ ] Simulator **tái hiện demo Layered/chunked trên Qwen3-30B-A3B** (bước 0): điểm sụp và tỉ lệ đạt SLO theo mức tải của cả hai chế độ, sai lệch goodput ≤ 15%. Chưa đạt thì không dùng simulator để quyết G1
- [ ] **Quyết G1a và G1b** (tiêu chí ở `plan/02` §3), viết kết luận trước khi nhìn số
- [ ] Đầu ra: TTFT P50/P99, phân phối TBT, E2E, goodput

## 2. Việc chi tiết

### 2.1 Mô hình mô phỏng

Iteration-level: mỗi iteration chọn batch decode cộng phần prefill theo policy, cộng cost theo bảng đã đo. Không mô phỏng kernel.

```
mỗi iteration:
  c_g = policy.choose(t, B, state)      # vector theo group với HyPrefill
  GPU = Σ_attn cost_FA(c_FA, t) + Σ_GDN cost_GDN(c_GDN)             # kernel riêng cho prefill
      + Σ_layer cost_GEMM/MoE(n_decode + tokens prefill của layer đó) # decode và prefill chung một lần gọi
      + decode attention/GDN + all-reduce (TP)
  T_iter = f(GPU, CPU(số lần gọi eager))  # chồng lấp GPU/CPU đo ở bước 1 §2.5
  cập nhật deadline, hàng đợi, TBT
```

Ba điểm khác bản đầu (đều là giả thuyết từ lượt 1830, cần đo lại): batch trộn cho GEMM/MoE, phần CPU mỗi iteration, all-reduce.

### 2.2 Cài baseline cho đúng

Mọi baseline dùng **cùng bảng cost** với HyPrefill và được tune tốt nhất ở từng (workload × SLO × model); báo cáo mức tốt nhất, ghi cả tham số đã chọn (`docs/03_MEASUREMENT.md` §7).

- **Static**: c cố định, giống Sarathi. Quét {256, 512, 1024, 2048, 4096, 8192} (bội 128 vì FA3 có bậc thang theo tile 128), lấy mức tốt nhất.
- **Layered Prefill**: mỗi iteration đúng một nhóm layer chạy prefill cho toàn bộ prompt, các nhóm khác chỉ decode. Mặc định của paper là `N_lg(L) = max(1, ⌈L/512⌉)`, nhưng Bảng 11 của họ cho thấy `N_lg` đổi TTFT lấy TBT, nên **quét `N_lg`** (gồm cả mặc định) và lấy mức tốt nhất.
- **SLOWeave**: `B_t = max(0, min_i(d_i − s_t))`; binary search c lớn nhất sao cho `T(n, c) + δ ≤ B_t`. Dùng chính bảng cost đã đo làm hàm T (họ chỉ yêu cầu đơn điệu), **tune δ**.
- **HyPrefill-oracle**: chunk theo group, gom k, so le, dùng bảng cost.
- **HyPrefill-static**: chunk theo group nhưng **không** phụ thuộc t. Đây là ablation quan trọng nhất, chứng minh phần "dynamic" có giá trị.

### 2.3 Kiểm chứng simulator

Lấy vài cấu hình (model, c, t, batch decode), chạy thật một iteration trên GPU, so với dự đoán của simulator. Ghi bảng sai số. Con số này sẽ vào paper — DistServe báo sai số simulator < 2%, mình đặt mục tiêu ≤ 15% vì mô hình thô hơn.

## 3. Ghi chú

Simulator là công cụ, không phải đóng góp. Đừng đầu tư quá nhiều. Nếu phần kiểm chứng kéo dài, giảm phạm vi: bỏ mô phỏng hàng đợi phức tạp, giữ đúng phần so sánh policy.


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
