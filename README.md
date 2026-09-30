# HyPrefill

**Operator-Decoupled Chunked Prefill for Hybrid LLM Serving**

Chunked prefill hiện nay dùng một chunk size cho toàn bộ model. Với model hybrid, các nhóm operator bị giới hạn bởi **những loại ràng buộc khác nhau**, nên chunk đồng nhất là `min` trên tất cả. HyPrefill cho mỗi nhóm operator một chunk size riêng, lấy từ cost model hiệu chỉnh offline.

| | |
|---|---|
| Mục tiêu nộp | **ICML 2027** |
| Dự phòng | SIGMETRICS 2027 (bản measurement) · NeurIPS 2027 · SOSP 2027 — deadline ở PROPOSAL §8 |
| Phần cứng | 1–8 × H200 (141 GB) |

---

## Bắt đầu từ đâu

1. Đọc `PROPOSAL.md` — toàn bộ luận điểm, cost model, kế hoạch, rủi ro.
2. Mở `docs/00_FOUNDATIONS.html` bằng trình duyệt — bài giảng nền tảng, có bộ tính chunk tương tác và bài giảng chi tiết chín paper.
3. In bài đọc từ `docs/01_READING_LIST.md` (có sẵn lệnh tải toàn bộ PDF).
4. Mở `plan/00_SETUP.md` và làm lần lượt theo thứ tự.

---

## Ba loại ràng buộc — luận điểm trong một bảng

| Loại ràng buộc | Operator | Hình dạng | Chunk muốn |
|---|---|---|---|
| Thời gian theo t | full attention; indexer của sparse attention | cost/token ∝ t | nhỏ dần khi t tăng |
| Bộ nhớ theo c·t | buffer logits indexer; buffer chunked-scan GDN/KDA | mem ∝ c·t | nhỏ dần khi t tăng |
| Khấu hao | MoE (đọc expert mỗi chunk); launch kernel GDN | cost/token ∝ 1/c | càng lớn càng tốt |

---

## Cổng quyết định

| Cổng | Sau bước | Tiêu chí |
|---|---|---|
| G1a | 3 | Layered / Sarathi ≥ 1.20× trên hybrid |
| G1b | 3 | HyPrefill / Layered ≥ 1.25× ở ít nhất một chế độ |
| G1c | 2 | Kernel indexer cấp phát buffer c·t theo mỗi lần gọi |
| G2 | 4 | Thắng SLOWeave ≥ 10% trên long-context (simulator) |
| G3 | 7 | Overhead buffer ≤ 50% oracle gain |
| **CỨNG** | 8 | **Có số end-to-end trong vLLM, nếu không thì phương án lui (fork hoặc measurement)** |
| G4 | 10 | Prototype khớp simulator ±15% |
| G5 | 13 | Đủ hình, chốt venue |

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
├── plan/00_SETUP.md … 16_SUBMIT.md  kế hoạch theo thứ tự bước, phần KẾT QUẢ để trống
├── bench/                         micro-benchmark cost model
├── sim/hyprefill_sim.js          mô phỏng dòng token (bản khởi đầu, hằng số minh hoạ)
├── results/                       CSV thô, không bao giờ ghi đè
└── figures/                       hình cho paper
```

---

## Mỗi file bước có gì

- Mục tiêu một câu, cổng (nếu có)
- Đầu ra bắt buộc dạng checklist
- Việc chi tiết kèm lệnh và công thức
- Tiêu chí kiểm tra
- **Phần KẾT QUẢ để trống** — điền khi làm xong: số liệu chính, hình sinh ra, điều bất ngờ, quyết định, việc chuyển bước sau
- Nhật ký theo ngày

---

## Phương án lui

| Ưu tiên | Phương án |
|---|---|
| 1 | FP4 precision residency (`docs/04_...` §5.3) — số liệu có sẵn từ bước 1–2 |
| 2 | VeriPrefill — phải dựng witness + conformal từ đầu |
| 3 | StateGraph — cần DeepSeek V4.1 trên 4–8 GPU, lâu nhất |

---

## Nguyên tắc

- **Viết tiêu chí GO/KILL trước khi chạy, không sửa sau khi thấy số.**
- Rủi ro độ mới chết lúc nộp; rủi ro kỹ thuật chết trước cả lúc nộp. Cổng cứng (bước 8) bảo vệ deadline.
- Báo cáo cả vùng không thắng. Reviewer tin một bài nói rõ giới hạn hơn một bài toàn thắng.
- Thứ Sáu hằng tuần gửi advisor 10 dòng kết quả kèm hình mới.
