Profile nsys của fork layered-prefill, làm lại sau khi đổi CPU (máy hyprefill-dev-0, governor `performance` ở CPU 48–71), cùng cấu hình với `../2026-09-30/`: Qwen3-30B-A3B, TP2, GPU 4–5 khoá 1980 MHz, arXiv 2.5 req/s, 240 s, cửa sổ nsys 60 s bắt đầu sau 200 s.
Lệnh: `QUEUE=B bash bench/run_cpu_recheck.sh` (gọi `bench/step00_layered_profile.sh`). Trace thô: `~/hyprefill_data/step00/layered_profile/2026-10-09_0208/`.

| | chunked 30/09 → 09/10 | layered 30/09 → 09/10 |
|---|---|---|
| Throughput (req/s, offered 2.5) | 2.28 → 2.44 | 2.26 → 2.44 |
| Mean / P99 TTFT (s) | 7.8 / 20.0 → 1.35 / 4.1 | 10.6 / 23.1 → 1.38 / 5.0 |
| Mean TPOT (ms) | 22.8 → 17.4 | 24.0 → 17.5 |
| `ModelRunner::run` median (ms) | 27.0 → 19.3 | 26.8 → 18.8 |

Máy cũ bão hoà vì CPU ở mức tải này; máy mới theo kịp. Layered vẫn ít GPU hơn chunked 11% (MoE −25%, attention −24%) và `prepare` vẫn ×2.9, TTFT không tốt hơn. Không lượt nào bị cgroup bóp (`cgroup_cpu.csv`).
