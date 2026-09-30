# Bước 13 — Scaling, tính đúng, năng lượng — CỔNG G5

**Mục tiêu:** Hoàn tất hình scaling theo context, kiểm tra tính đúng, chốt venue.
**Cổng:** **G5 — đủ hình cho paper chưa? Chọn venue cuối: ICML hay SOSP (deadline ở PROPOSAL §8).**

---

## 1. Đầu ra bắt buộc

- [ ] Hình scaling: gain theo t từ 4K đến 256K — **hình ăn tiền của paper**
- [ ] Kiểm tra tính đúng: 200 prompt greedy, so token-level với baseline
- [ ] Bảng năng lượng
- [ ] Bảng decision cost (từ bước 6, cập nhật với số cuối)
- [ ] Quyết định venue

## 2. Hình scaling

Quét t ∈ {4K, 8K, 16K, 32K, 64K, 128K, 256K} ở P99 TBT cố định. Kỳ vọng: khoảng cách giữa HyPrefill và baseline mở rộng theo t. Đây là hình chứng minh cơ chế đúng chứ không phải trùng hợp.

Vẽ cả đường của SLOWeave để thấy nó bám sát ở context ngắn rồi tụt lại ở context dài.

## 3. Kiểm tra tính đúng

200 prompt, greedy decoding, so token-level với baseline chunked prefill. Nếu có lệch:
- Xác định lệch ở đâu (layer nào, boundary nào)
- Giải thích được không? GDN chunking về mặt toán học là chính xác (state truyền đúng thứ tự cho kết quả giống nguyên khối), nên lệch chỉ nên đến từ sai số dấu phẩy động
- Nếu lệch có hệ thống, đó là bug, phải sửa

Báo cáo tỉ lệ token khớp trong paper.

## 4. Cổng G5 — chọn venue

| Tình huống | Venue |
|---|---|
| Đủ hình, prototype chạy, ≥ 4 model | **ICML 2027**. Áp dụng 4 điều chỉnh viết cho venue ML, xem PROPOSAL §8.1 |
| Đủ hình nhưng chỉ có phương án lui (fork hoặc measurement), không có vLLM end-to-end | **SIGMETRICS 2027** bản measurement. Deadline sát ICML, không nộp cả hai |
| Cần thêm model hoặc thêm hệ thống | **NeurIPS 2027** (nộp lại kèm phản biện ICML) hoặc SOSP 2027 |

Quyết định ở đây, không để đến bước 15.


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
