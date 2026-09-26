# HyPrefill

**Operator-Decoupled Chunked Prefill for Hybrid LLM Serving**

Chunked prefill hiện nay dùng một chunk size cho toàn bộ model. Với model hybrid, các nhóm operator bị giới hạn bởi **những loại ràng buộc khác nhau**, nên chunk đồng nhất là `min` trên tất cả. HyPrefill cho mỗi nhóm operator một chunk size riêng, lấy từ cost model hiệu chỉnh offline.

| | |
|---|---|
| Khởi động | 15/09/2026 |
| Mục tiêu nộp | **ICML 2027** (~28/01/2027) |
| Dự phòng | SIGMETRICS 11/01/2027 (bản measurement) · NeurIPS ~05/2027 · SOSP ~01/04/2027 |
| Phần cứng | 1–8 × H200 (141 GB) |
| Thời lượng | 16 tuần + 2 tuần đệm |

---

## Bắt đầu từ đâu

1. Đọc `PROPOSAL.md` — toàn bộ luận điểm, cost model, kế hoạch, rủi ro.
2. Mở `docs/00_FOUNDATIONS.html` bằng trình duyệt — bài giảng nền tảng, có bộ tính chunk tương tác và bài giảng chi tiết chín paper.
3. In bài đọc từ `docs/01_READING_LIST.md` (có sẵn lệnh tải toàn bộ PDF).
4. Mở `weeks/WEEK_00.md` và bắt đầu.

---

## Ba loại ràng buộc — luận điểm trong một bảng

| Loại ràng buộc | Operator | Hình dạng | Chunk muốn |
|---|---|---|---|
| Thời gian theo t | full attention; indexer của sparse attention | cost/token ∝ t | nhỏ dần khi t tăng |
| Bộ nhớ theo c·t | buffer logits indexer; buffer chunked-scan GDN/KDA | mem ∝ c·t | nhỏ dần khi t tăng |
| Khấu hao | MoE (đọc expert mỗi chunk); launch kernel GDN | cost/token ∝ 1/c | càng lớn càng tốt |

---

## Cổng quyết định

| Cổng | Tuần | Ngày | Tiêu chí |
|---|---|---|---|
| G1a | 2 | 05/10 | Layered / Sarathi ≥ 1.20× trên hybrid |
| G1b | 2 | 05/10 | HyPrefill / Layered ≥ 1.25× ở ít nhất một chế độ |
| G1c | 2 | 05/10 | Kernel indexer cấp phát buffer c·t theo mỗi lần gọi |
| G2 | 4 | 19/10 | Thắng SLOWeave ≥ 10% trên long-context (simulator) |
| G3 | 7 | 09/11 | Overhead buffer ≤ 50% oracle gain |
| **CỨNG** | 8 | 16/11 | **Có số end-to-end, nếu không thì dừng port vLLM** |
| G4 | 10 | 30/11 | Prototype khớp simulator ±15% |
| G5 | 13 | 21/12 | Đủ hình, chốt venue |

---

## Cấu trúc

```
HyPrefill/
├── PROPOSAL.md                    luận điểm, cost model, kế hoạch, rủi ro
├── README.md                      file này
├── LOG.md                         nhật ký hằng ngày
├── docs/
│   ├── 00_FOUNDATIONS.html        bài giảng nền tảng + 9 paper (mở bằng trình duyệt)
│   ├── 01_READING_LIST.md         link PDF để in + lệnh tải
│   ├── 02_RELATED_WORK.md         số liệu đã xác minh, nguyên liệu Related Work
│   ├── 03_MEASUREMENT.md          kỷ luật đo lường
│   ├── 04_REVIEW_fp4_hopper_fallback.md   phương án lui số 1
│   ├── 05_DEADLINES_2027.md       deadline hội nghị
│   └── lecture_notes/             ghi chú đầy đủ 9 paper
├── weeks/WEEK_00.md … WEEK_16.md  kế hoạch từng tuần, phần KẾT QUẢ để trống
├── bench/                         micro-benchmark cost model
├── sim/hyprefill_sim.js          mô phỏng dòng token (bản khởi đầu, hằng số minh hoạ)
├── results/                       CSV thô, không bao giờ ghi đè
└── figures/                       hình cho paper
```

---

## Mỗi file tuần có gì

- Mục tiêu một câu, ngày, cổng (nếu có)
- Đầu ra bắt buộc dạng checklist
- Việc chi tiết kèm lệnh và công thức
- Tiêu chí kiểm tra
- **Phần KẾT QUẢ để trống** — điền khi làm xong: số liệu chính, hình sinh ra, điều bất ngờ, quyết định, việc chuyển tuần sau
- Nhật ký theo ngày

---

## Phương án lui

| Ưu tiên | Phương án | Thời gian |
|---|---|---|
| 1 | FP4 precision residency (`docs/04_...` §5.3) — số liệu có sẵn từ tuần 1–2 | ~6 tuần |
| 2 | VeriPrefill — phải dựng witness + conformal từ đầu | ~10 tuần |
| 3 | StateGraph — cần DeepSeek V4.1 trên 4–8 GPU | 6–10 tuần, quá chậm |

---

## Nguyên tắc

- **Viết tiêu chí GO/KILL trước khi chạy, không sửa sau khi thấy số.**
- Rủi ro độ mới chết lúc nộp; rủi ro kỹ thuật chết trước cả lúc nộp. Cổng cứng tuần 8 bảo vệ deadline.
- Báo cáo cả vùng không thắng. Reviewer tin một bài nói rõ giới hạn hơn một bài toàn thắng.
- Thứ Sáu hằng tuần gửi advisor 10 dòng kết quả kèm hình mới.
