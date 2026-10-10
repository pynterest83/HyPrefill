# Bước 4 — Workload

| Thư mục | Là gì |
|---|---|
| `2026-10-10_trace_stats/` (`bench/trace_stats.py`) | Đặc trưng trace `semianalysisai/cc-traces-weka-062126-256k` (393 phiên coding agent, 68 266 request gồm cả sub-agent). Prefix cache theo phiên phục vụ được 96.2% token input; phần prefill thật mỗi request: trung vị 1 600, p90 6 848, p99 49 408 token; context trung vị 88 768, p90 204 288. Phân bố công prefill: append ≥ 8 192 chiếm 56%, ≥ 16 384 chiếm 44%; context ≥ 64K chiếm 63%. Vùng M1 thấy HyPrefill thắng rõ (context ≥ 128K, append ≥ 8 192) là 15% công prefill; context ≥ 64K, append ≥ 8 192 là 37% |
| `2026-10-10_reqsim_prelim_v2cal_B{50,25}.csv` (`bench/reqsim.py`) | **Sơ bộ** (hiệu chỉnh M1 v2, chunk không bội 544): goodput trên trace, SLO_TTFT 1.02 s. TBT 50 ms: Layered = HyPrefill 2.31, SLOWeave 2.13, chunked 1.00 req/s (HyPrefill / tốt nhất = 1.00). TBT 25 ms: Layered 2.06, HyPrefill 2.00, chunked = SLOWeave = 0 (không chunk nào đi hết 48 layer vừa budget) |
| `2026-10-10_reqsim_prelim_v2cal_diagnostics.txt` | Nghẽn ở đâu (sơ bộ, phụ thuộc mô hình CPU với h0 = 10 ms): 60–92% iteration bị chặn bởi CPU (lõi GDN eager + chi phí cố định), GPU dành ~70% cho decode và ~10% cho prefill, KV chỉ dùng 35–50% → tối ưu xếp prefill khó nhích goodput; cần xác nhận trên vLLM thật |
