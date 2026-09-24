# Tuần 12 — Đánh giá phần 2 và ablation

**Ngày:** 08–14/12/2026
**Mục tiêu:** Hoàn tất eval cho ba model còn lại và chạy toàn bộ ablation.

---

## 1. Đầu ra bắt buộc

- [ ] Kết quả cho Kimi-Linear-48B-A3B, Qwen3-30B-A3B, Qwen3.8-Flash-Next
- [ ] Sáu ablation chạy xong
- [ ] Bản đồ regime (Hình 3) với số thật

## 2. Ba model còn lại

| Model | Câu hỏi nó trả lời |
|---|---|
| Kimi-Linear-48B-A3B | Cơ chế có tổng quát sang họ linear attention khác, vendor khác không? |
| Qwen3-30B-A3B | Trên model chỉ có attention + MoE, HyPrefill có suy biến về Layered Prefill và có thắng nó trên chính benchmark của họ không? |
| Qwen3.8-Flash-Next | Luận điểm còn đứng khi attention chuyển sang sparse có indexer không? Ràng buộc là thời gian hay bộ nhớ? |

## 3. Ablation

| # | Ablation | Câu hỏi |
|---|---|---|
| a | Bỏ phụ thuộc vị trí (HyPrefill-static) | Phần "dynamic" đáng bao nhiêu? |
| b | Bỏ stagger | Stagger đáng bao nhiêu? P99 TBT có vỡ không? |
| c | k cố định so với k từ oracle | Chọn k có cần tinh vi không? |
| d | 2 nhóm (FA/nonFA) so với 3 nhóm | Tách MoE riêng có đáng không? |
| e | Đổi tỉ lệ attention:GDN:MoE | Gain phụ thuộc bố cục thế nào? |
| f | **Node prefill-only (không ràng buộc TBT)** | Kỳ vọng **không gain** — chứng minh hiểu giới hạn |
| g | **Mô phỏng node decode của hệ EPD** (append-prefill + decode chung GPU, KV prefix đến từ node khác) | Gain còn bao nhiêu khi hệ đã disaggregate? Đây là câu trả lời trực tiếp cho headwind từ blog GLM |

Ablation (f) và (g) quan trọng về mặt trình bày. (f) trả lời "sao không PD-disaggregate cho xong" và cho thấy mình hiểu giới hạn. (g) quan trọng hơn: blog hạ tầng GLM (17/09/2026) cho thấy lab tuyến đầu phục vụ model hybrid ở quy mô 100.000 accelerator chọn **EPD disaggregated** và không nhắc chunked prefill. Nếu (g) cho gain đáng kể, HyPrefill đúng ngay cả trong thế giới disaggregated — đó là lập luận bền nhất của bài.

## 4. Bản đồ regime

Hình 3 của paper: gain theo (họ kiến trúc × context). Ba họ:
- Full-attention hybrid (Qwen3-Next, Qwen3.8-27B, Kimi-Linear) — kỳ vọng gain từ ≥ 32K
- Sparse-attention hybrid (Qwen3.8-Flash-Next) — kỳ vọng gain muộn hơn, từ ≥ 256K, qua ràng buộc bộ nhớ
- Full attention + MoE (Qwen3-30B-A3B) — đối chiếu Layered Prefill

Nói thẳng vùng nào không thắng.


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
