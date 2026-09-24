# Tuần 09 — Port sang vLLM (hoặc củng cố)

**Ngày:** 17–23/11/2026
**Mục tiêu:** Đưa cơ chế vào vLLM để kết quả thuyết phục reviewer, tôn trọng mọi ràng buộc kernel.

---

## 1. Đầu ra bắt buộc

- [ ] Scheduler HyPrefill chạy trong vLLM trên Qwen3-Next
- [ ] Không vi phạm ràng buộc kernel (danh sách bên dưới)
- [ ] Kết quả khớp fork trong ±10%

## 2. Ràng buộc bắt buộc tôn trọng

| Ràng buộc | Nguồn | Kiểm tra bằng |
|---|---|---|
| `FLA_CHUNK_SIZE = 64` không đổi | vLLM PR #49827 | grep trong code, không override |
| Chunk size kernel Mamba2 giống nhau prefill và decode | vLLM RFC #55524 | chạy test bit-identical nếu có |
| Chunk boundary align theo block size nhóm Mamba | vLLM PR #54076 | assert trong scheduler |
| Buffer chunked-scan GDN nằm ngoài memory profiling | vLLM issue #54775 | đặt trần thủ công, tránh OOM |

## 3. Chiến lược giảm rủi ro

- **Tắt prefix cache trong eval chính.** Hybrid + prefix cache còn nhiều bug (SGLang #39342 làm hỏng mamba radix checkpoint; vLLM #43587). Chạy một cấu hình riêng có prefix cache để báo cáo, nhưng không để nó chặn kết quả chính.
- **Bắt đầu từ TP1.** PR #49827 chỉ hỗ trợ TP1 cho mixed decode+prefill GDN. Mở rộng TP sau.
- **Giữ fork chạy song song.** Nếu vLLM vướng, vẫn có số từ fork.

## 4. Nếu tuần 8 đã quyết định phương án measurement

Bỏ qua việc port. Dùng tuần này để:
- Mở rộng bảng cost sang model thứ 4 và 5
- Làm sâu phần oracle: thêm ràng buộc bộ nhớ, thêm chiều precision path
- Bắt đầu viết sớm


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
