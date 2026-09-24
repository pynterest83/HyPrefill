# Tuần 01 — Cost curve — attention và GDN

**Ngày:** 22–28/09/2026
**Mục tiêu:** Đo cost_FA(c, t) và cost_GDN(c, t) bằng micro-benchmark kernel, và xác nhận thực nghiệm rằng GDN không phụ thuộc t.

---

## 1. Đầu ra bắt buộc

- [ ] `bench/op_cost.py` chạy được, xuất CSV thô
- [ ] Bảng cost_FA(c, t) cho c ∈ {64…8192}, t ∈ {0, 4K, 16K, 32K, 64K, 128K, 256K}
- [ ] Bảng cost_GDN(c, t) cùng dải
- [ ] **Kiểm chứng then chốt**: giữ c cố định, đổi t → latency GDN bất biến (sai lệch < 5%)
- [ ] Hình 1 bản nháp: thời gian mỗi token theo c, một đường mỗi t, một panel mỗi operator

## 2. Việc chi tiết

### 2.1 Kernel gọi trực tiếp, không qua vLLM

| Group | Kernel | Tham số |
|---|---|---|
| FA | `flash_attn_varlen_func`, q_len = c, kv_len = t + c, causal | c ∈ {64,128,256,512,1024,2048,4096,8192}; t ∈ {0, 4K, 16K, 32K, 64K, 128K, 256K} |
| GDN | `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, seq_len = c, **`initial_state` ≠ None** | cùng c; t ảnh hưởng qua initial_state (kỳ vọng phẳng) |
| Dense | Q/K/V/O projection, norm, gate | cùng c, để hoàn thiện tổng |

Shape lấy đúng từ `config.json` của từng model (tuần 0). Nhân với số layer mỗi loại để ra cost toàn model.

### 2.2 Tại sao `initial_state` quan trọng

Đây là điểm dễ sai nhất của tuần. Nếu gọi `chunk_gated_delta_rule` không truyền `initial_state`, bạn đang đo trường hợp prefill từ đầu, không phải trường hợp chunked prefill có state tích luỹ. Sinh `initial_state` bằng cách chạy trước t token rồi lấy state ra, hoặc tạo tensor ngẫu nhiên đúng shape `d_k × d_v` mỗi head (đủ cho mục đích đo thời gian).

### 2.3 Chú ý c không chia hết 64

Kernel pad chunk cuối, nên c = 65 tốn gần bằng c = 128 (⌈65/64⌉ = 2). Đo cả c lẻ để thấy bậc thang này; scheduler sau sẽ chọn c là bội của 64.

### 2.4 Tách ba thành phần của cost_GDN

Fit `cost_GDN(c) = a + b·⌈c/64⌉`:
- `a` = overhead cố định mỗi lần gọi → ngoại suy từ c rất nhỏ về c = 0
- `b` = chi phí biên mỗi chunk-64, gộp intra O(C²d) và inter O(Cd²)

Đổi `d_k`, `d_v`, `num_heads` để tách tỉ trọng intra so với inter.

### 2.5 Kiểm chứng tổng

So tổng cost dự đoán với một forward pass thật của model qua HuggingFace ở cùng (c, t). **Sai số ≤ 15% là chấp nhận.** Nếu lệch hơn, tìm thành phần bỏ sót (thường là norm, gate, hoặc residual).

## 3. Tiêu chí kiểm tra trước khi sang tuần 2

- cost_FA/token gần phẳng theo c nhưng dâng rõ theo t → đúng dự đoán
- cost_GDN/token giảm theo c và **không đổi theo t** → đúng dự đoán
- Nếu GDN *có* phụ thuộc t đáng kể, dừng lại và tìm nguyên nhân (nhiều khả năng do cách sinh initial_state hoặc do cache allocator), vì toàn bộ luận điểm dựa vào tính chất này.


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
