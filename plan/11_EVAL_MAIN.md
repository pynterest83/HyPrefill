# Bước 11 — Đánh giá phần 1 — model chính

**Mục tiêu:** Chạy ma trận eval đầy đủ cho Qwen3-Next và Qwen3.8-27B.

---

## 1. Đầu ra bắt buộc

- [ ] Kết quả đầy đủ cho Qwen3-Next-80B-A3B (có MoE) và Qwen3.8-27B (không MoE)
- [ ] Ba workload × ba SLO × ba baseline đã tune (static, Layered Prefill, SLOWeave) + ablation HyPrefill-static
- [ ] Số năng lượng (J/token)
- [ ] Bảng chính của paper thành hình

## 2. Ma trận

| Trục | Giá trị |
|---|---|
| Model | Qwen3-Next-80B-A3B, Qwen3.8-27B |
| Workload | long-context, append-prefill agentic, mixed |
| SLO | TBT 50 ms (headline), 25, 100 ms; TTFT theo quy tắc PROPOSAL §4.4; độ nhạy × 0.5 / 1 / 2 |
| Baseline | static chunk (quét {256…8192}, bội 128), Layered Prefill (quét `N_lg`), SLOWeave (tune δ); ablation HyPrefill-static. Mức tốt nhất ở từng ô |
| Metric | TTFT P50/P99, TBT P99, E2E, goodput, J/token |
| Tải | quét đến bão hoà |

## 3. Vì sao cặp model này

Qwen3.8-27B cùng họ GDN với anchor nhưng **không có MoE**. So sánh cặp này tách được: bao nhiêu phần gain đến từ GDN, bao nhiêu từ MoE. Đây là ablation mà reviewer sẽ hỏi.

## 4. Kỷ luật

- Theo đủ `docs/03_MEASUREMENT.md` §7: cùng GPU, TP, cặp GPU, ghim CPU cùng NUMA node, cùng bộ kernel và cấu hình engine cho mọi policy
- Chạy **xen kẽ** HyPrefill và baseline trong cùng phiên; mỗi cấu hình chạy ≥ 3 lần, lấy median, ghi cả khoảng dao động
- Khoá clock GPU trước mỗi phiên đo, xác nhận đã có hiệu lực; GPU dùng riêng, ghi tải lạ và clock thực tế theo từng lần chạy
- CSV thô vào `results/step11/<ngày>/`, không ghi đè
- Đo năng lượng bằng `nvidia-smi --query-gpu=power.draw -lms 100` chạy nền, tích phân; chỉ trên GPU dùng riêng


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
