# Tuần 06 — Đo overhead buffer và stagger

**Ngày:** 27/10–02/11/2026
**Mục tiêu:** Đo chi phí thật của activation buffer trước khi xây tiếp — đây là failure mode của COREY.

---

## 1. Đầu ra bắt buộc

- [ ] Activation buffer cài xong trong fork
- [ ] Bảng overhead tách riêng: copy buffer, launch thêm, đồng bộ hoá, dispatch
- [ ] So overhead với oracle gain tuần 2
- [ ] **Bảng "decision quality vs decision cost"** kiểu COREY Bảng 4–5

## 2. Vì sao tuần này quan trọng nhất về mặt rủi ro

COREY (arXiv 2604.10597) có decision quality cao — chọn đúng chunk như oracle — nhưng end-to-end **chậm hơn static 8.8%** vì decision cost: dựng histogram, synchronization boundary, dispatch overhead. Tác giả kết luận "no entropy-guided variant beats the best static chunk".

HyPrefill tránh phần ước lượng runtime (cost model offline, tra bảng), nhưng **vẫn có** chi phí buffer copy và launch thêm. Phải đo, không giả định.

## 3. Cách đo

Tách bốn thành phần, mỗi thành phần đo riêng:

1. **Copy buffer**: thời gian ghi và đọc activation giữa các group. So với băng thông HBM lý thuyết.
2. **Launch thêm**: số lần gọi kernel tăng bao nhiêu so với đồng nhất.
3. **Đồng bộ hoá**: có điểm sync nào mới không? Nếu có, đo.
4. **Dispatch**: chi phí tra bảng cost và chọn chunk. Kỳ vọng gần 0 vì offline.

Chạy với `torch.cuda.Event` quanh từng phần, không dùng wall clock.

## 4. Tiêu chí kiểm tra

**Overhead tổng ≤ 50% oracle gain.** Nếu vượt:
- Thử giảm k (gom ít chunk hơn → buffer nhỏ hơn)
- Thử gộp nhóm (2 nhóm FA/nonFA thay vì 3)
- Nếu vẫn vượt → đây là kết quả âm, dừng nhánh prototype và chuyển sang bài measurement + oracle cho SIGMETRICS 11/01

## 5. Lưu ý

Ghi lại **cả trường hợp overhead nhỏ**. Nếu buffer rẻ hơn dự đoán, đó cũng là một finding đáng viết: nó giải thích vì sao Layered Prefill bỏ ngỏ phần này mà vẫn chạy được.


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
