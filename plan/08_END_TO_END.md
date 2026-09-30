# Bước 8 — End-to-end trong vLLM — CỔNG CỨNG

**Mục tiêu:** Có số goodput, TTFT/TBT/E2E thật trong vLLM trên Qwen3-Next, cho HyPrefill, Layered (k = 1) và chunked đã tune, chạy được workload append-prefill và long-context.
**Cổng:** **CỔNG CỨNG — phải có số end-to-end trong vLLM trước khi sang bước 9.** Nếu không ra được: **chuyển sang phương án lui trên fork nanovllm** (thêm GDN vào fork, so trong fork) nếu còn đủ thời gian; nếu không, viết bài dựa trên cost model + simulator đã kiểm chứng, nộp SIGMETRICS. Đây là quyết định bảo vệ deadline, không phải thất bại.

---

## 1. Đầu ra bắt buộc

- [ ] Chạy được workload append-prefill (có prefix cache) và long-context end-to-end trong vLLM với Qwen3-Next-80B-A3B, TP2
- [ ] Goodput (PROPOSAL §4.4), TTFT P50/P99, TBT P99, E2E cho HyPrefill, **Layered** (k = 1, `N_lg` đã tune) và chunked đã tune, cùng engine, cùng GPU, xen kẽ (`docs/03_MEASUREMENT.md` §7)
- [ ] So với dự đoán của simulator bước 3
- [ ] **Quyết định cổng cứng, ghi rõ trong file này**

## 2. Vì sao có cổng cứng ở đây

Rủi ro lớn nhất của toàn dự án không phải độ mới mà là **engineering**, và sửa model runner của vLLM là phần nặng nhất. Phần prototype (bước 5–10) rất dễ kéo dài gấp rưỡi dự kiến, và kéo dài như vậy là trượt deadline.

Cổng cứng bảo đảm: dù nhánh vLLM có vấn đề, vẫn còn đường lui (fork, hoặc bài measurement chỉn chu).

## 3. Nếu có số end-to-end

Tiếp tục: bước 9 củng cố trong vLLM (TP, prefix cache, ràng buộc), bước 10 hoàn thiện, bước 11–13 eval đầy đủ. Mục tiêu ICML.

## 4. Nếu chưa có

| Còn thời gian | Làm |
|---|---|
| Đủ cho 3–4 bước nữa | Phương án lui trên fork: thêm GDN vào fork nanovllm, cài HyPrefill cạnh Layered gốc, so trong fork; cost model đo lại bằng kernel của fork |
| Không đủ | Phương án measurement (bảng dưới) |

| Phần | Nội dung | Đã có từ bước |
|---|---|---|
| Characterization | Các loại ràng buộc, bảng cost đo được trên 4–5 model | 1–2 |
| Phân tích | Chunk đồng nhất = min over operators, có số | 2 |
| Mô phỏng đã kiểm chứng | Goodput theo (t, B, kiến trúc), 3 baseline, 3 workload, 3 SLO | 3–4 |
| Overhead | Bảng decision cost | 6 |
| Giới hạn | Nói thẳng chưa có prototype đầy đủ | — |

Deadline SIGMETRICS xem PROPOSAL §8.


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
