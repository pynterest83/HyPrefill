# Danh mục kết quả (cập nhật 2026-10-07)

**Chuẩn đo hiện tại:** GPU 4–7 dành riêng cho dự án, khoá clock **1980 MHz**; FA đo với KV paged theo block size của vLLM; state GDN theo `mamba_ssm_dtype` của config. Bước 1 đo xong 2026-09-29 (`bench/run_step01_1980.sh`); bước 0 (kiểm chứng Layered) xong 2026-09-30; bước 2 có bảng MoE, decode và routing thật, còn `moe_mixed` với routing thật và G1c.

| Đường dẫn | Nội dung |
|---|---|
| `step00/model_configs.md`, `step00/configs/` | Bố cục layer và shape của 6 model, đọc từ `config.json` (không phụ thuộc phép đo) |
| `env/2026-09-29/` | phiên bản gói, driver, clock, commit lúc đo (`scripts/record_env.sh`) |
| `step00/layered_demo/2026-09-30/{summary,energy}.csv` | Fork Layered: chunked và layered, Qwen3-30B-A3B, arXiv, 9 mức tải, tỉ lệ đạt SLO và năng lượng mỗi token |
| `step00/layered_demo/2026-10-09/{summary,energy}.csv` | Như 30/09 nhưng trên máy mới sau khi đổi CPU, 10 mức tải 1.3–4.0 req/s. Chunked bão hoà ~3.0 req/s (máy cũ ~2.64); Layered giảm TTFT 3–58% từ 2.5 req/s, SLO đạt ở 3.2 req/s: 82% so với 33%; năng lượng −10…−14%. JSON thô ở `~/hyprefill_data/step00/layered_demo/2026-10-09/` |
| `step00/layered_profile/2026-09-30/` | Profile nsys của fork ở 2.5 req/s: thời gian kernel theo loại và NVTX phía CPU, chunked so với layered (`analysis.txt`) |
| `step01/cost_tables/<model>.csv` | **Bảng cost chính thức bước 1**: median 2 card (GPU 6, 7), kèm `*_repeatability.txt` và `analysis.txt`. Từng lượt ở `step01/2026-09-29_<model>_tp<N>_gpu{6,7}/` |
| `step01/2026-09-29_allreduce_*` | All-reduce TP2 qua đường của vLLM |
| `step01/2026-09-29_vllm_step_sweep_*` | §2.5: thời gian thực mỗi step so với GPU bận (chi phí CPU mỗi iteration) |
| `step01/2026-09-29_intra_*`, `step01/2026-09-29_validate_forward/` | §2.4 và §2.5 (các dòng t = 0 của §2.5 đo sai cách, xem `plan/01` R3) |
| `step01/2026-09-30_validate_forward*/` | §2.5 làm lại: t = 0 đã sửa; có decode chạy song song (`decode_extra_cost.csv`: mô hình `mixed` khớp ở c ≥ 2048) |
| `step02/cost_tables/` | Bảng MoE, decode, Qwen3-30B-A3B gộp từ GPU 6, 7; từng lượt ở `step02/2026-09-30_s02_*` |
| `step02/moe_overlap_*.csv` | Routing MoE thật (Qwen3-Next, arXiv, 64 request): số expert decode và chunk prefill chạm, phần prefill thêm vào; dump thô ở `~/hyprefill_data/step02/moe_routing/` |
| `step02/kt2_capacity_*_estimate.csv` | KT2 (PROPOSAL §5, sửa 07/10): dung lượng prefill mỗi iteration, đồng nhất / Layered / HyPrefill, MoE **ước lượng** (cảnh báo sớm, chưa có KT1); `_kt1.csv` khi có KT1 |
| `step02/2026-10-08_s02_*_kt1_draw{0,1,2}/` | KT1 (máy hyprefill-dev-0, GPU 4, 1980 MHz): `moe_mixed` routing thật (dump 192 request, `~/hyprefill_data/step02/moe_routing/2026-10-08_*`), D ∈ {8, 32, 64}, c ∈ {0 … 8192}, 3 lần rút ghép cặp theo D |
| `step02/kt1_eval_*_2026-10-08.csv` | Kết luận KT1 (`bench/kt1_eval.py`): inc(D, c), tỉ lệ 2048 / 4×512, f. **GO** |
| `step02/kt2_capacity_*_kt1.csv` | KT2 với MoE đo thật (KT1), **bản có lỗi NaN**: KT2-a ghi KILL (1.04) là sai, xem `*_kt1_fix.csv`; KT2-b GO (1.875) |
| `step02/kt2_capacity_*_kt1_fix.csv` | KT2 sau khi sửa lỗi NaN (ô chunk đồng nhất không khả thi): KT2-a 1.18 → vùng giữa. **Dùng file này thay `*_kt1.csv`** |
| `step02/2026-10-08_moe_diag/` | `fused_experts` theo số expert bị chạm × số token, routing đặt tay (`bench/moe_diag.py`) |
| `step02/2026-10-08_moe_config_cf/` | KT1 dưới mọi cấu hình tune của `fused_moe` (`bench/moe_config_cf.py`); lượt mặc định tái hiện KT1 ≤ 1.1% |
| `step02/kt1_eval_*_bestcfg.csv`, `kt2_capacity_*_kt1_bestcfg.csv` | KT1 / KT2 với cấu hình MoE tốt nhất mỗi ô: KT1 vùng chưa quyết (0.63), KT2-a 1.14, KT2-b 1.51 |
| `step02/kt2_diag_*_2026-10-08/` | Chẩn đoán (`bench/kt2_diag.py`): chi phí mỗi token theo nhóm, bản đồ (t, P), độ nhạy g và `h_fire`; xem `docs/12` |
| `step01/2026-10-09_cpu_recheck_*_gpu6/` | Đo lại `host_ms` sau khi đổi CPU (governor `performance` ở NUMA 2–3): FA −32%, GDN ổn định 0.275 ms; thời gian GPU khớp 29/09 ±5% |
| `step01/2026-10-09_vllm_step_sweep_Qwen3.8-27B_tp1/` | §2.5 làm lại trên máy mới: CPU/idle mỗi step ở c = 512 giảm 34 → 9 ms. (Bản Qwen3-Next TP2 cùng ngày **không hợp lệ**, đã chuyển ra `~/hyprefill_data/invalid_runs/`) |
| `step00/layered_profile/2026-10-09/` | Profile fork làm lại sau khi đổi CPU: máy cũ bão hoà ở 2.5 req/s, máy mới không (TTFT 7.8 → 1.35 s) |
| `step02/kt2_capacity_*_kt1_host1009.csv` | KT2 (đã sửa NaN) với `host_ms` đo 09/10: không đổi so với `*_kt1_fix.csv` |
| `step02/2026-10-09_g1c_indexer_mem_{cap512,uncapped}/` | G1c: bộ nhớ đỉnh và thời gian indexer QSA (Qwen3.8-Flash-Next) theo (c, t), có / không giới hạn 512 MB: bị chặn ở ~1 GiB, giới hạn tốn ≤ 6% thời gian. **G1c trượt** (`plan/02` R4) |
| `step02/2026-10-09_g1c_kv_capacity/` | G1c trên model thật Flash-Next-FP8 TP2: KV cache theo chunk tối đa (−1.6% ở 8192, −15% ở 32768) |
| `step02/moe_overlap_*_arxiv192_2026-10-08dump.csv` | Expert decode / prefill / hợp trên dump 192 request, D ∈ {8, 32, 64, 128} |
| `step01/2026-10-09_validate_forward_Qwen3-Next-80B-A3B-Instruct_tp2/` | §2.5 lần đầu cho Qwen3-Next TP2: GPU thật lệch bảng cost −3 … −7% ở các ô GPU là ràng buộc. **Lượt chạy với `max_num_seqs = 16` (mọi step eager)**: hai dòng t = 64K ở c ≤ 2048 không dùng được (`docs/13` §1) |
| `step01/2026-10-09_cudagraph_check/` | Step chunked prefill Qwen3-Next TP2 theo cỡ CUDA graph được capture: c = 2048 mặc định 87 ms/step, capture tới 2048: 44 ms/step |
| `step00/layered_demo/2026-10-09_run2/` | Quét mịn 3.05–3.3 req/s: goodput (≥ 90% SLO) chunked ~3.05, layered ~3.15 req/s (~1.03×) |
| `step02/kt2_sim_check*_2026-10-09.csv` | Mô phỏng từng iteration thay công thức KT2 (`bench/kt2_sim_check.py`): HyPrefill / Layered tới 1.31–1.36, 5/46 ô ≥ 1.25 (TBT 25 ms, t ≥ 128K); `_bestcfg`: MoE theo cấu hình tốt nhất. Xem `docs/13` |
| `step02/2026-10-08_moe_config_cf/best_draw{0,1,2}/` | Chi phí `moe_mixed` theo cấu hình tốt nhất mỗi ô, dạng đầu vào `--kt1` |
| `step02/oracle_*`, `step02/sim_*` | (sẽ có) oracle, mô phỏng |

## Dữ liệu cũ, không trộn với chuẩn mới (ở `~/hyprefill_data/` trên server)

| Thư mục | Là gì | Dùng được cho |
|---|---|---|
| `clk1830/results/` | Toàn bộ số đo 28–29/09 ở clock khoá 1830 MHz, KV liền mạch, state GDN luôn fp32 (cùng cấu trúc với `results/`) | Kết luận định tính và căn cứ cho các quyết định đã ghi trong `LOG.md`; so số cũ với số mới. **Không** dùng số tuyệt đối cho paper. Tóm tắt ở `plan/00`, `plan/01` mục KẾT QUẢ |
| `clk1830/raw/` | JSON từng request của demo Layered, trace profiler của lượt đó | Như trên |
| `clk1830/logs/`, `clk1830/figures/` | Log các lượt đo và hình của lượt đó | Như trên |
| `setup_logs/` | Log dựng môi trường và tải model | Tra lỗi khi dựng lại môi trường |

Đã xoá 2026-09-29: số đo trước khi khoá clock, số đo bằng cách đo cũ (một lần replay graph), các lượt chạy dở và chạy thử code. Lý do loại từng loại ghi ở `LOG.md` và mục KẾT QUẢ `plan/00`, `plan/01`.
