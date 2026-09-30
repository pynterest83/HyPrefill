# Bước 4 — Workload và baseline trên simulator — CỔNG G2

**Mục tiêu:** Chạy ba workload qua năm policy, quét tải và ba mức SLO, xác định vùng thắng.
**Cổng:** **G2 — HyPrefill-oracle vượt SLOWeave ≥ 10% goodput (định nghĩa ở PROPOSAL §4.4; chốt 2026-09-29, trước khi có số) trên workload long-context và append-prefill.** Nếu không, thu hẹp claim hoặc xem lại thiết kế trước khi đầu tư vào prototype (bước 5–10).

---

## 1. Đầu ra bắt buộc

- [ ] Tải trace `semianalysisai/cc-traces-weka-062126-256k`, viết bộ sinh request (token giả khớp `hash_ids`, co giãn timestamp), kiểm tỉ lệ prefix hit trong vLLM khớp với trace
- [ ] Đo TTFT không tải ở P90 cho từng (model, workload), tính SLO_TTFT theo quy tắc PROPOSAL §4.4 và **ghi vào PROPOSAL trước khi so policy**
- [ ] Tính dung lượng KV + state Mamba ở context 256K trên 2×H200 để chọn dải mức tải
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

Trace (chốt 2026-09-29): **`semianalysisai/cc-traces-weka-062126-256k`** (HuggingFace, Apache-2.0, 570 MB). 393 phiên Claude Code thật, 68 266 request (28 444 lượt của agent chính, 39 822 request của 1 697 nhóm subagent chạy song song), input trung bình ~101K token, output trung bình ~860 token, mỗi request `input + output ≤ 256 000`. Mỗi request có timestamp tương đối `t`, `in`, `out`, và `hash_ids` theo block 64 token để biết phần prefix dùng lại; **không có text**. Hệ quả khi dùng:
- Sinh token giả khớp `hash_ids` (cùng hash → cùng token), để prefix cache của vLLM hit đúng như trong trace. Độ dài đếm bằng tokenizer của Claude; dùng nguyên số token như trong trace và ghi rõ trong paper.
- Bắt buộc bật prefix cache, nên chịu ràng buộc cắt chunk theo block Mamba (`plan/05` §2.5).
- Context tới 256K giới hạn số request chạy song song trên 2×H200: tính dung lượng KV + state Mamba trước khi chọn mức tải.
- Mức tải: co giãn timestamp theo một hệ số (ghi rõ hệ số), giữ thứ tự và độ chồng lấp của subagent.
- Qwen3-30B-A3B (tối đa 40K) không chạy được trace này; model đó chỉ dùng kiểm chứng Layered trên arXiv/ShareGPT.

### 2.3 Mixed
70% chat ngắn (1–4K) + 30% long-context.

## 3. Ma trận chạy

| Trục | Giá trị |
|---|---|
| Workload | long-context, append-prefill, mixed |
| SLO | TBT 50 ms (headline), 25, 100 ms, và mức 5 × bước decode của từng model; TTFT theo quy tắc PROPOSAL §4.4; độ nhạy × 0.5 / 1 / 2 |
| Model | Qwen3-Next, Qwen3.8-27B, Kimi-Linear, (Qwen3.8-Flash-Next nếu có số bước 2) |
| Policy | static (quét chunk), Layered Prefill (quét `N_lg`), SLOWeave (tune δ), HyPrefill-static, HyPrefill-oracle; baseline lấy mức tốt nhất ở từng ô |
| Tải | quét đến bão hoà |

Mọi policy chạy trên cùng trace, cùng seed, cùng quá trình đến; baseline tune theo `docs/03_MEASUREMENT.md` §7.

## 4. Cổng G2

Tiêu chí ở đầu file. Lưu ý: **thua ở short-context là bình thường và phải báo cáo trung thực**. SLOWeave đã đóng phần lớn gap ở context ngắn; đóng góp của HyPrefill nằm ở long-context.

Nếu HyPrefill-static ≈ HyPrefill-oracle, nghĩa là phần phụ thuộc vị trí không đáng, và paper đơn giản đi nhưng cũng yếu đi. Ghi nhận và điều chỉnh claim.


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
