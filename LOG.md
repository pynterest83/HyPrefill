# Nhật ký thí nghiệm HyPrefill

Mỗi ngày 5 dòng. Viết trong ngày, không viết bù cuối tuần.
Mẫu: **Làm gì** · **Số liệu chính** · **Bất ngờ** · **Bị chặn** · **Mai làm gì**

---

## Nhật ký

### 2026-09-28
- Làm gì: dựng env `hyprefill` (vllm 0.30.0, torch 2.13.0+cu130), tải xong 3 model bước 0, viết `bench/op_cost.py` + `bench/analyze_step01.py`, chạy lượt đo bước 1 (FA3 + GDN FlashInfer/triton) cho Qwen3-Next TP2 và Qwen3.8-27B TP1 trên GPU 4. **Clock chưa khoá** → số tham khảo, không đưa vào paper.
- Số liệu chính: GDN không phụ thuộc t (xu hướng t≥128K so với t≤16K ≤ 5%, một điểm lạc c=4096 t=16K). GDN FlashInfer Qwen3-Next TP2: cost = 53.5 µs + 1.05 µs·⌈c/64⌉ (chi phí cố định trội tới c ≈ 3.2K). FA3 Qwen3-Next TP2 ở t=256K: ~3.0 µs/token/layer, gần phẳng theo c ≥ 256.
- Bất ngờ: (1) đo eager một lần gọi bị giới hạn bởi CPU (wrapper GDN ~0.2 ms CPU so với ~0.04 ms GPU) và tạo ra phụ thuộc t giả, phải chuyển sang đo replay CUDA graph. (2) Triton FLA rẻ hơn FlashInfer ở c nhỏ (a = 32 µs so với 53 µs) nhưng đắt hơn 3× mỗi chunk; điểm cắt ~640 token. (3) FA3 có bậc thang theo tile 128: c=129 đắt hơn c=128 rõ (Qwen3.8-27B t=256K: 15.8 so với 9.7 µs/token).
- Bị chặn: root trong container không khoá được clock GPU (cần admin ở host). GPU dùng chung, lúc có lúc không.
- Mai: nhờ admin khoá clock GPU 4–7; đo phần Dense (projection, norm, gate); mục 2.4 (tách intra/inter) và 2.5 (so tổng cost với forward thật).

### 2026-09-29
- Làm gì: đo lại bước 1 bằng cách đo mới (graph K lần + xoá L2), thêm Dense, §2.4, §2.5 (vLLM thật + profile); demo Layered quét tải 1.3–3.5 req/s; bước 2: MoE, decode, oracle đầu tiên; G1c đọc mã indexer; chốt benchmark (PROPOSAL §4.4).
- Số liệu chính: bảng cost lặp lại median 0.2–0.9%; GDN lệch theo t ≤ 0.6%; §2.5 phần GPU +2%, cả iteration −6..−10% ở c ≥ 2048 nhưng −22..−35% ở c = 512; demo: chunked và layered cùng sụp ở 3.0 req/s (throughput 2.67 / 2.61); oracle Qwen3-Next TP2 ở SLO 50 ms: r_d / r_u = 1.27–2.0 (t ≤ 64K).
- Bất ngờ: vLLM prefill chunk nhỏ bị giới hạn bởi CPU; mô hình số phép tính intra/inter không khớp; indexer QSA trong vLLM 0.30 tự chia query để buffer logits ≤ 512 MB (G1c có thể trượt); Layered không thắng trên H200 với cấu hình của fork.
- Bị chặn: chưa đo được G1c (đang tải Flash-Next, mạng lỗi); cần sim có pipeline + goodput mới quyết G1a/G1b.
- Mai: dừng lại tổng hợp, bàn hướng với advisor trước khi chạy tiếp.

### 2026-09-29 (chiều, chuẩn đo 1980 MHz)
- Làm gì: dọn dữ liệu cũ; đo lại bước 1 trên GPU 6, 7 (mỗi card một lượt), all-reduce TP2 trên GPU 4, 5; bắt đầu kiểm chứng Layered trên GPU 4, 5.
- Số liệu chính: lặp lại giữa hai card median 0.17–0.19%, 100% dòng không chạm trần < 3% (mục bước 0 đạt); GDN không phụ thuộc t (≤ 0.3%); §2.5 −4…−9% ở c ≥ 1024, −15…−32% ở c = 512; all-reduce TP2 39 µs ở 2048 token.
- Bất ngờ: ở c = 512, t = 16K GPU chờ CPU 44% mỗi step; phần GPU của engine dài hơn bảng 2–10%; GDN triton lệch giữa hai card (nghi autotune).
- Bị chặn: các dòng t = 0 của §2.5 đo sai cách (trừ nhầm phần đọc weight); bản đầu của `--sweep` sai, đã sửa và chạy lại.
- Mai: sửa §2.5 t = 0, mô hình hoá phần CPU; bảng bước 2 (MoE, decode) để chạy §2.5 cho Qwen3-Next và §2.5 có decode song song.

---

## Ý tưởng để dành (không code trong lúc chạy kế hoạch)

-

## Quyết định lớn và lý do

| Ngày | Quyết định | Lý do | Dữ liệu đỡ lưng |
|---|---|---|---|
| 2026-09-28 | Đo bằng kernel vLLM 0.30 (FA3, GDN FlashInfer), không dùng flash-attn/fla riêng | Bảng cost phải cùng kernel với engine sẽ đánh giá; flash-attn 2.8.3 không build được với torch mới | `AGENTS.md` mục Server |
| 2026-09-29 | Trace workload chính: `semianalysisai/cc-traces-weka-062126-256k`. SLO: TBT 50 ms headline + 25 / 100 ms + mức 5 × bước decode; TTFT = 5 × TTFT không tải P90 (con số chốt trước khi so policy); độ nhạy × 0.5 / 1 / 2 | Trace agentic thật, có hash prefix, đúng workload append-prefill; quy tắc SLO theo cách Layered, SLOWeave, DistServe đặt | PROPOSAL §4.2, §4.4; `plan/04` |
| 2026-09-29 | Mọi policy trong cùng vLLM 0.30; Layered = cấu hình k = 1 của hạ tầng chạy theo nhóm layer dùng chung với HyPrefill; fork nanovllm chỉ để kiểm chứng bản Layered, bảng tham khảo khác engine, và phương án lui ở cổng cứng | Công bằng (cùng engine, cùng đường code) và áp dụng được vào vLLM; tác giả Layered cũng chỉ so khác engine với vLLM như tham khảo | `plan/05`, `08`, `09`; PROPOSAL §4.3–4.4, §5 |
| 2026-09-29 | Bật prefix cache cho workload append-prefill (bản cũ: tắt trong eval chính) | Workload chính dựa vào context cũ đã nằm trong cache; khi bật, vLLM cắt chunk theo block Mamba (`mamba_cache_mode = "align"`), là ràng buộc thật cho chunk size | `plan/05` §2.5, `plan/09` §3 |
| 2026-09-29 | Rà lại kế hoạch sau lượt 1830: chuẩn đo mới (1980 MHz, KV paged, state GDN theo config); tiêu chí lặp lại theo median/95%; thêm all-reduce, MoE batch trộn, phần CPU của iteration; G1a/G1b quyết ở bước 3 bằng mô phỏng đã khớp hệ thật; bỏ công thức intra/inter khỏi đường găng. Các phát hiện bất lợi (G1c, MoE dùng chung với decode, Layered trên H200) **giữ là giả thuyết**, đo lại theo chuẩn mới rồi mới quyết | Lượt 1830 cho thấy cách đo và proxy cũ có thể sai; chưa đủ để bỏ trụ nào của luận điểm | `plan/00`–`03`, PROPOSAL §1.3, §2.2, §5 |
| 2026-09-29 | Chốt benchmark trước khi có số G1/G2: headline = goodput (≥ 90% request đạt TTFT ≤ SLO_TTFT và mọi TBT ≤ SLO_TBT), so với baseline tốt nhất từng ô, số headline trên vLLM 0.30 | Theo cách Layered Prefill, SLOWeave, DistServe claim; trước đó metric của G1/G2 chưa định nghĩa ("goodput hoặc TTFT P99") | PROPOSAL §4.4; demo bước 0: theo TTFT trung bình thì layered chậm hơn chunked ~20% ở tải nhẹ, nhưng theo tỉ lệ đạt SLO thì cả hai 100% tới 2.5 req/s, chunked sụp ở 3.0 req/s |
