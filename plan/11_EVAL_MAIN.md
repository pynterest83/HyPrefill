# Tuần 11 — Đánh giá phần 1 — model chính

**Ngày:** 01–07/12/2026
**Mục tiêu:** Chạy ma trận eval đầy đủ cho Qwen3-Next và Qwen3.8-27B.

---

## 1. Đầu ra bắt buộc

- [ ] Kết quả đầy đủ cho Qwen3-Next-80B-A3B (có MoE) và Qwen3.8-27B (không MoE)
- [ ] Ba workload × ba SLO × bốn baseline
- [ ] Số năng lượng (J/token)
- [ ] Bảng chính của paper thành hình

## 2. Ma trận

| Trục | Giá trị |
|---|---|
| Model | Qwen3-Next-80B-A3B, Qwen3.8-27B |
| Workload | long-context, append-prefill agentic, mixed |
| SLO P99 TBT | 25, 50, 100 ms |
| Baseline | vLLM static (512/2048/8192), Layered Prefill, SLOWeave, HyPrefill-static |
| Metric | TTFT P50/P99, TBT P99, E2E, goodput, J/token |
| Tải | quét đến bão hoà |

## 3. Vì sao cặp model này

Qwen3.8-27B cùng họ GDN với anchor nhưng **không có MoE**. So sánh cặp này tách được: bao nhiêu phần gain đến từ GDN, bao nhiêu từ MoE. Đây là ablation mà reviewer sẽ hỏi.

## 4. Kỷ luật

- Mỗi cấu hình chạy ≥ 3 lần, lấy median, ghi cả khoảng dao động
- Khoá clock GPU trước mỗi phiên đo
- CSV thô vào `results/week11/<ngày>/`, không ghi đè
- Đo năng lượng bằng `nvidia-smi --query-gpu=power.draw -lms 100` chạy nền, tích phân


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
