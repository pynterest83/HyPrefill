# Tuần 10 — Hoàn thiện prototype — CỔNG G4

**Ngày:** 24–30/11/2026
**Mục tiêu:** Prototype chạy ổn định trên hai model, khớp simulator.
**Cổng:** **G4 — prototype khớp simulator trong ±15%.** Nếu lệch hơn, tìm nguyên nhân và báo cáo sai lệch như một finding, không che.

---

## 1. Đầu ra bắt buộc

- [ ] Chạy ổn định trên Qwen3-Next-80B-A3B và Kimi-Linear-48B-A3B
- [ ] Bảng so sánh prototype với simulator trên ≥ 10 cấu hình
- [ ] Sửa xong các lỗi phát hiện ở tuần 8–9
- [ ] Đóng băng tính năng — từ tuần 11 chỉ đo, không thêm

## 2. Việc chi tiết

### 2.1 Kiểm chứng simulator

Chọn 10 cấu hình (model × workload × SLO), chạy cả prototype và simulator, lập bảng sai số. Nếu sai số có cấu trúc (ví dụ simulator luôn lạc quan ở tải cao), tìm thành phần thiếu — thường là chi phí hàng đợi hoặc preemption.

### 2.2 Ổn định

Chạy liên tục 2 giờ ở tải cao, kiểm tra không rò bộ nhớ, không deadlock. Tham khảo SGLang issue #37904 (PDMux deadlock với mixed decode + split-prefill trên hybrid GDN sau prompt ≥ 4K) để biết chỗ dễ vướng.

### 2.3 Đóng băng

Từ tuần 11 trở đi chỉ chạy đo và viết. Mọi ý tưởng thêm ghi vào `LOG.md` mục "future work", không code.

## 3. Ghi chú

Đây là tuần cuối cùng được sửa cơ chế. Nếu còn ý tưởng cải tiến, cân nhắc kỹ: một cải tiến 5% không đáng nếu nó làm trượt lịch eval.


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
