# Tuần 03 — Simulator sự kiện rời rạc

**Ngày:** 06–12/10/2026
**Mục tiêu:** Dựng simulator iteration-level từ bảng cost đã đo, cài 5 policy, và xác nhận simulator khớp phép đo đơn lẻ.

---

## 1. Đầu ra bắt buộc

- [ ] `sim/` chạy được, đầu vào là bảng cost + trace request
- [ ] Năm policy cài xong: static (512/1024/2048), Layered Prefill, SLOWeave, HyPrefill-oracle, HyPrefill-static
- [ ] Simulator khớp phép đo một iteration thật trong **±15%**
- [ ] Đầu ra: TTFT P50/P99, phân phối TBT, E2E, goodput

## 2. Việc chi tiết

### 2.1 Mô hình mô phỏng

Iteration-level: mỗi iteration chọn batch decode cộng phần prefill theo policy, cộng cost theo bảng đã đo. Không mô phỏng kernel.

```
mỗi iteration:
  D = cost_decode(len(active_decode))
  P = B − D
  c_g = policy.choose(t, P, state)      # vector theo group với HyPrefill
  T_iter = D + Σ_g cost_g(c_g, t) / (số iteration mà group g trải ra)
  cập nhật deadline, hàng đợi, TBT
```

### 2.2 Cài baseline cho đúng

- **Static**: c cố định, giống Sarathi. Ba mức 512, 1024, 2048.
- **Layered Prefill**: `N_lg(L) = max(1, ⌈L/512⌉)`; mỗi iteration đúng một nhóm layer chạy prefill cho toàn bộ prompt, các nhóm khác chỉ decode.
- **SLOWeave**: `B_t = max(0, min_i(d_i − s_t))`; binary search c lớn nhất sao cho `T(n, c) ≤ B_t`. Dùng chính bảng cost đã đo làm hàm T (họ chỉ yêu cầu đơn điệu).
- **HyPrefill-oracle**: chunk theo group, gom k, so le, dùng bảng cost.
- **HyPrefill-static**: chunk theo group nhưng **không** phụ thuộc t. Đây là ablation quan trọng nhất, chứng minh phần "dynamic" có giá trị.

### 2.3 Kiểm chứng simulator

Lấy vài cấu hình (model, c, t, batch decode), chạy thật một iteration trên GPU, so với dự đoán của simulator. Ghi bảng sai số. Con số này sẽ vào paper — DistServe báo sai số simulator < 2%, mình đặt mục tiêu ≤ 15% vì mô hình thô hơn.

## 3. Ghi chú

Simulator là công cụ, không phải đóng góp. Đừng đầu tư quá một tuần. Nếu đến thứ Sáu chưa xong phần kiểm chứng, giảm phạm vi: bỏ mô phỏng hàng đợi phức tạp, giữ đúng phần so sánh policy.


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
