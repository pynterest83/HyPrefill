# Bước 15 — Bản nháp đầy đủ

**Mục tiêu:** Viết xong bản nháp và gửi hai người đọc chéo.

---

## 1. Đầu ra bắt buộc

- [ ] Bản nháp đầy đủ, đúng số trang của venue
- [ ] Gửi ít nhất hai người đọc chéo
- [ ] Mọi claim trong bài truy được về một số cụ thể trong `results/`

## 2. Thứ tự viết

Viết theo thứ tự này, không theo thứ tự đọc:

1. **Evaluation** trước — số đã có, dễ viết nhất, và nó quyết định claim nào giữ được
2. **Design** — mô tả đúng cái đã cài, không phải cái đã định
3. **Motivation** — giờ đã biết kết quả, viết motivation dẫn đến đúng kết quả đó
4. **Related Work** — từ `docs/02_RELATED_WORK.md`
5. **Intro** — viết sau cùng, khi đã biết bài nói gì
6. **Abstract** — cuối cùng, điền X, Y, Z từ số thật

## 3. Ba câu hỏi reviewer phải được trả lời trong bài

| Reviewer đọc gì | Câu hỏi | Trả lời ở mục nào |
|---|---|---|
| Layered Prefill | Activation giữa các group quản lý ra sao? Trên attention+MoE có suy biến về Layered Prefill và thắng không? | Design §buffer; Eval §Qwen3-30B-A3B |
| SLOWeave | Sao không chạy SLOWeave riêng cho mỗi group? | Design §vì sao không tách được (chung deadline, phụ thuộc qua buffer) |
| COREY | Overhead của cơ chế chọn chunk là bao nhiêu? | Eval §decision cost |

## 4. Tự kiểm tra trước khi gửi đọc chéo

- Mỗi con số trong abstract có xuất hiện lại trong Evaluation không?
- Có claim nào không có số đỡ lưng không?
- Có báo cáo vùng không thắng không? (Nếu không, reviewer sẽ nghi ngờ toàn bộ)
- Limitations có nói về phạm vi colocated và về model sparse không?


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
