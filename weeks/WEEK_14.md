# Tuần 14 — Outline và toàn bộ hình

**Ngày:** 22–28/12/2026
**Mục tiêu:** Cố định outline và hoàn thành mọi hình với caption tự đứng được.

---

## 1. Đầu ra bắt buộc

- [ ] Outline chi tiết đến mức tiểu mục
- [ ] Mọi hình hoàn thành, mỗi hình có caption đọc riêng vẫn hiểu
- [ ] Mọi bảng hoàn thành
- [ ] Danh sách citation đầy đủ

## 2. Cấu trúc paper

| Mục | Trang | Nội dung |
|---|---|---|
| Intro | 1.5 | Hybrid + MoE là kiến trúc mặc định 2026 (liệt kê model); chunked prefill dùng một chunk cho mọi layer; quan sát: chunk đồng nhất = min over operators với **ba loại ràng buộc**; ba đóng góp |
| Background & Motivation | 2 | Hình 1–3 từ tuần 2. **Phần thuyết phục nhất** |
| Design | 3 | Cost model, chọn k, buffer, stagger, tương tác prefix cache và mixed batch, ràng buộc kernel |
| Implementation | 0.5 | Engine, LOC, ràng buộc |
| Evaluation | 3.5 | Setup → kết quả chính → scaling theo t → ablation → overhead → so SLOWeave/Layered Prefill |
| Related Work | 1 | Dùng `docs/02_RELATED_WORK.md` |
| Conclusion | 0.5 | |

## 3. Hình

| # | Nội dung | Từ tuần |
|---|---|---|
| 1 | Chi phí mỗi token theo chunk, một đường mỗi t, một panel mỗi operator | 1–2 |
| 2 | c*(t) theo t: attention giảm ~1/t, GDN và MoE phẳng | 2 |
| 3 | Bản đồ regime: gain theo (kiến trúc × context) | 12 |
| 4 | TTFT theo context ở P99 TBT cố định, 4 policy | 11–13 |
| 5 | Ablation | 12 |
| 6 | Decision cost | 6, 13 |

## 4. Nguyên tắc caption

Caption phải nói **kết luận**, không phải mô tả trục. Sai: "TTFT theo context length". Đúng: "Khoảng cách TTFT giữa HyPrefill và chunked prefill mở rộng từ 8% ở 8K lên 41% ở 256K, ở P99 TBT không đổi."


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
