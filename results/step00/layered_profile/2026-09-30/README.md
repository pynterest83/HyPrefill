Profile nsys của fork layered-prefill (Qwen3-30B-A3B, TP2, GPU 4–5 khoá 1980 MHz, arXiv 2.5 req/s, 240 s).
Cửa sổ nsys 60 s, bắt đầu 200 s sau khi khởi động server (giữa lúc tải; dưới nsys cả hai chế độ đều bão hoà, ~2.27 req/s).
Lệnh: `MODES="chunked layered" bash bench/step00_layered_profile.sh`; phân tích: `python bench/analyze_fork_profile.py <dir>`.
Trace thô (.nsys-rep, .sqlite): ~/hyprefill_data/step00/layered_profile/2026-09-30_0557/.
