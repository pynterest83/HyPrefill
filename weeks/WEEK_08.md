# Tuần 08 — End-to-end trên fork — CỔNG CỨNG

**Ngày:** 10–16/11/2026
**Mục tiêu:** Có số TTFT/TBT/E2E thật trên Qwen3-Next, chạy được workload long-context.
**Cổng:** **CỔNG CỨNG — cuối tuần 8 phải có số end-to-end.** Nếu chưa có: **dừng kế hoạch port sang vLLM**, viết bài dựa trên oracle + simulator, nộp SIGMETRICS 11/01. Đây là quyết định bảo vệ deadline, không phải thất bại.

---

## 1. Đầu ra bắt buộc

- [ ] Chạy được workload long-context end-to-end trên fork với Qwen3-Next-80B-A3B
- [ ] Số TTFT P50/P99, TBT P99, E2E cho HyPrefill và ít nhất một baseline
- [ ] So với dự đoán của simulator
- [ ] **Quyết định cổng cứng, ghi rõ trong file này**

## 2. Vì sao có cổng cứng ở đây

Rủi ro lớn nhất của toàn dự án không phải độ mới mà là **engineering**. Ước tính 6 tuần cho prototype có thể thành 10. Với deadline cuối tháng 1, mất thêm 4 tuần là trượt.

Cổng cứng tuần 8 bảo đảm: dù nhánh prototype có vấn đề, vẫn còn 8 tuần để viết một bài measurement chỉnh chu.

## 3. Nếu có số end-to-end

Tiếp tục theo kế hoạch: tuần 9 port vLLM, tuần 10 hoàn thiện, tuần 11–13 eval đầy đủ. Mục tiêu ATC/ICML cuối tháng 1.

## 4. Nếu chưa có

Chuyển sang **phương án measurement**, phạm vi thu hẹp nhưng hoàn chỉnh:

| Phần | Nội dung | Đã có từ tuần |
|---|---|---|
| Characterization | Ba loại ràng buộc, bảng cost đo được trên 4–5 model | 1–2 |
| Phân tích | Chunk đồng nhất = min over operators, có số | 2 |
| Oracle bound | Gain theo (t, B, kiến trúc) | 2 |
| Simulator | 4 baseline, 3 workload, 3 SLO | 3–4 |
| Overhead | Bảng decision cost | 6 |
| Giới hạn | Nói thẳng chưa có prototype đầy đủ | — |

Deadline SIGMETRICS 2027 winter: **11/01/2027**. Còn 8 tuần, dư.


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
