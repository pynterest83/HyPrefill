# Bước 5 — M1: bộ chạy thử bằng kernel thật (`bench/hyprefill_emulator.py`, `bench/m1_eval.py`)

Tiêu chí chốt trước khi chạy: PROPOSAL §5 (10/10/2026). Máy hyprefill-dev-0, GPU 4–7 khoá 1980 MHz, Qwen3-Next-80B-A3B shape TP2 mỗi GPU.

| Thư mục / file | Là gì |
|---|---|
| `2026-10-10_m1_D*` | Lượt đầu, **bị thay** (xem `README_m1_run1.md`) |
| `2026-10-10_m1v2_D*` | Lượt chính: B ∈ {25, 50, 100}, D ∈ {8, 32}, t ∈ {128K, 256K}, Δ ∈ {2048, 8192}, hệ số an toàn 1.04 |
| `2026-10-10_m1v2s{1.06,1.08}_*` | Cùng ô then chốt với hệ số an toàn 1.06 / 1.08 cho mọi policy |
| `m1_eval_2026-10-10_m1v2_strict.csv` | Chỉ lượt 1.04: GO theo câu chữ (2 ô ≥ 1.30), nhưng cả hai ô có cấu hình tốt nhất của Layered vượt budget 0.2–0.6% |
| `m1_eval_2026-10-10_m1v2_tuned.csv` | **Kết luận chính thức**: mỗi policy tune cả biên an toàn (1.04/1.06/1.08): tối đa 1.21, 0 ô ≥ 1.30, 2 ô ≥ 1.15, 6 ô ≥ 1.10 → **ở giữa** |
| `m1_ttft_2026-10-10.csv` (`bench/m1_ttft.py`) | TTFT một append đơn lẻ và thời gian phục vụ khi append nối tiếp, từ lịch và chi phí hiệu chỉnh của M1 (chỉ CPU). Có tải: HyPrefill nhanh hơn Layered 9–17% ở TBT 25–50 ms, t dài. **Request đơn lẻ: HyPrefill chậm hơn 1.1–2.7×** (attention chunk nhỏ, GDN/MoE chờ gom k·c token, các layer attention của cùng một request không chồng lên nhau); TTFT đơn lẻ của cả hai policy pipeline cao (2.4–10 s) vì lịch "sâu trước" chỉ cho token tiến một sublayer mỗi iteration |
