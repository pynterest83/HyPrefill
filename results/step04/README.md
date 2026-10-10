# Bước 4 — Workload

| Thư mục | Là gì |
|---|---|
| `2026-10-10_trace_stats/` (`bench/trace_stats.py`) | Đặc trưng trace `semianalysisai/cc-traces-weka-062126-256k` (393 phiên coding agent, 68 266 request gồm cả sub-agent). Prefix cache theo phiên phục vụ được 96.2% token input; phần prefill thật mỗi request: trung vị 1 600, p90 6 848, p99 49 408 token; context trung vị 88 768, p90 204 288. Phân bố công prefill: append ≥ 8 192 chiếm 56%, ≥ 16 384 chiếm 44%; context ≥ 64K chiếm 63%. Vùng M1 thấy HyPrefill thắng rõ (context ≥ 128K, append ≥ 8 192) là 15% công prefill; context ≥ 64K, append ≥ 8 192 là 37% |
