# Bước 7 — Scheduler lõi — CỔNG G3

**Mục tiêu:** Hoàn thành scheduler chọn chunk theo group với stagger, chạy đúng trong vLLM (cùng hạ tầng với chế độ Layered).
**Cổng:** **G3 — overhead buffer + stagger ≤ 50% oracle gain.** Nếu vượt: dừng nhánh prototype, viết bài measurement + oracle + simulator, nộp SIGMETRICS.

> **Sửa 10/10/2026:** chế độ HyPrefill bản tĩnh đã chuyển sang bước 5. Bước này làm phần thích nghi: chọn `c_g(t)` theo cost model khi context và tải thay đổi, stagger tối ưu.

---

## 1. Đầu ra bắt buộc

- [ ] Scheduler chọn `c_g(t)` cho từng group từ bảng cost
- [ ] Stagger hoạt động: đỉnh mỗi iteration không vượt B
- [ ] Output token-level giống baseline trên 20 prompt greedy (kiểm tra nhanh)
- [ ] Quyết định G3

## 2. Việc chi tiết

### 2.1 Cấu trúc scheduler

```
mỗi iteration:
  D          = thời gian decode batch hiện tại (đo hoặc dự đoán)
  P          = B − D
  c_FA       = lookup(cost_FA, t, P)              # bội của 128 (tile FA3), align block Mamba
  for g in nonFA_groups:
      if iteration % k_g == offset_g:
          chạy group g với k_g · c_FA token từ buffer
      else:
          bỏ qua group g trong iteration này
  kiểm tra: D + Σ (group chạy trong iteration này) ≤ B
```

### 2.2 Chọn offset

Với các nhóm nonFA có `k_g` khác nhau, chọn offset sao cho không hai nhóm nào cùng chạy chunk lớn trong một iteration. Bài toán nhỏ, giải bằng greedy hoặc quét vét cạn (số nhóm ít).

### 2.3 Xử lý đuôi

Khi prompt gần hết, buffer không đủ k chunk. Xử lý: chạy nốt với số chunk có sẵn. Ghi lại chi phí của trường hợp này.

### 2.4 Kiểm tra tính đúng sớm

Chạy 20 prompt greedy, so token-level với baseline chunked prefill. Nếu lệch, nguyên nhân thường là state GDN truyền sai thứ tự hoặc boundary không align. Sửa ngay, đừng để đến bước 13.

## 3. Cổng G3

Tiêu chí ở đầu file. Đây là cổng có xác suất trượt cao thứ hai sau G1. Nếu trượt, bài measurement vẫn là một bài thật: ba loại ràng buộc, bảng cost đo được, oracle bound, simulator với 4 baseline. SIGMETRICS nhận loại này.


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
