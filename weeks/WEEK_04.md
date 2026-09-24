# Tuần 04 — Workload và baseline trên simulator — CỔNG G2

**Ngày:** 13–19/10/2026
**Mục tiêu:** Chạy ba workload qua năm policy, quét tải và ba mức SLO, xác định vùng thắng.
**Cổng:** **G2 — HyPrefill-oracle vượt SLOWeave ≥ 10% (goodput hoặc TTFT P99) trên workload long-context và append-prefill.** Nếu không, thu hẹp claim hoặc xem lại thiết kế trước khi đầu tư 6 tuần prototype.

---

## 1. Đầu ra bắt buộc

- [ ] Ba workload dựng xong
- [ ] Bảng gain của HyPrefill-oracle so với 4 baseline theo (workload × SLO × model)
- [ ] Xác định regime thắng: ngưỡng context, tỉ lệ GDN:attention, số expert
- [ ] Bản nháp Hình 4 (TTFT theo context ở P99 TBT cố định)

## 2. Ba workload

### 2.1 Long-context
Prompt 32K–256K từ LongBench-v2 hoặc RULER, output 256–1024. Đến theo Poisson, quét tải đến bão hoà.

### 2.2 Append-prefill agentic — WORKLOAD CHÍNH
Prefix cache hit 64K–256K, thêm 1K–4K token mới mỗi lượt (tool output), decode chạy song song. Mô phỏng node decode của hệ PD/PPD theo "Not All Prefills Are Equal" (arXiv 2603.13358).

**Vì sao đây là vùng mạnh nhất:** t rất lớn nhưng token mới ít, nên chunk attention bị ép cực nhỏ trong khi GDN và MoE muốn xử lý cả vài nghìn token mới một lần. Chênh lệch giữa ràng buộc của các group đạt cực đại.

**Vì sao đây là phạm vi bền nhất (cập nhật 18/09/2026):** blog hạ tầng GLM cho thấy lab tuyến đầu phục vụ model hybrid ở quy mô lớn chọn EPD disaggregated, không phải colocated. Workload này đúng **ngay cả trong hệ disaggregated**, vì node decode vẫn có ràng buộc TBT và vẫn phải chạy append-prefill (PPD, arXiv 2603.13358: giảm 68% TTFT từ lượt hai). Nếu chỉ có một workload cho số headline, chọn workload này.

Trace: dùng transcript SWE-agent hoặc tool-use thật nếu có; nếu không, sinh tổng hợp với phân bố độ dài tool output thực tế.

### 2.3 Mixed
70% chat ngắn (1–4K) + 30% long-context.

## 3. Ma trận chạy

| Trục | Giá trị |
|---|---|
| Workload | long-context, append-prefill, mixed |
| SLO P99 TBT | 25, 50, 100 ms |
| Model | Qwen3-Next, Qwen3.8-27B, Kimi-Linear, (Qwen3.8-Flash-Next nếu có số tuần 2) |
| Policy | static×3, Layered Prefill, SLOWeave, HyPrefill-static, HyPrefill-oracle |
| Tải | quét đến bão hoà |

## 4. Cổng G2

Tiêu chí ở đầu file. Lưu ý: **thua ở short-context là bình thường và phải báo cáo trung thực**. SLOWeave đã đóng phần lớn gap ở context ngắn; đóng góp của HyPrefill nằm ở long-context.

Nếu HyPrefill-static ≈ HyPrefill-oracle, nghĩa là phần phụ thuộc vị trí không đáng, và paper đơn giản đi nhưng cũng yếu đi. Ghi nhận và điều chỉnh claim.


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
