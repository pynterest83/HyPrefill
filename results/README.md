# Danh mục kết quả (cập nhật 2026-10-07)

**Chuẩn đo hiện tại:** GPU 4–7 dành riêng cho dự án, khoá clock **1980 MHz**; FA đo với KV paged theo block size của vLLM; state GDN theo `mamba_ssm_dtype` của config. Bước 1 đo xong 2026-09-29 (`bench/run_step01_1980.sh`); bước 0 (kiểm chứng Layered) xong 2026-09-30; bước 2 có bảng MoE, decode và routing thật, còn `moe_mixed` với routing thật và G1c.

| Đường dẫn | Nội dung |
|---|---|
| `step00/model_configs.md`, `step00/configs/` | Bố cục layer và shape của 6 model, đọc từ `config.json` (không phụ thuộc phép đo) |
| `env/2026-09-29/` | phiên bản gói, driver, clock, commit lúc đo (`scripts/record_env.sh`) |
| `step00/layered_demo/2026-09-30/{summary,energy}.csv` | Fork Layered: chunked và layered, Qwen3-30B-A3B, arXiv, 9 mức tải, tỉ lệ đạt SLO và năng lượng mỗi token |
| `step00/layered_profile/2026-09-30/` | Profile nsys của fork ở 2.5 req/s: thời gian kernel theo loại và NVTX phía CPU, chunked so với layered (`analysis.txt`) |
| `step01/cost_tables/<model>.csv` | **Bảng cost chính thức bước 1**: median 2 card (GPU 6, 7), kèm `*_repeatability.txt` và `analysis.txt`. Từng lượt ở `step01/2026-09-29_<model>_tp<N>_gpu{6,7}/` |
| `step01/2026-09-29_allreduce_*` | All-reduce TP2 qua đường của vLLM |
| `step01/2026-09-29_vllm_step_sweep_*` | §2.5: thời gian thực mỗi step so với GPU bận (chi phí CPU mỗi iteration) |
| `step01/2026-09-29_intra_*`, `step01/2026-09-29_validate_forward/` | §2.4 và §2.5 (các dòng t = 0 của §2.5 đo sai cách, xem `plan/01` R3) |
| `step01/2026-09-30_validate_forward*/` | §2.5 làm lại: t = 0 đã sửa; có decode chạy song song (`decode_extra_cost.csv`: mô hình `mixed` khớp ở c ≥ 2048) |
| `step02/cost_tables/` | Bảng MoE, decode, Qwen3-30B-A3B gộp từ GPU 6, 7; từng lượt ở `step02/2026-09-30_s02_*` |
| `step02/moe_overlap_*.csv` | Routing MoE thật (Qwen3-Next, arXiv, 64 request): số expert decode và chunk prefill chạm, phần prefill thêm vào; dump thô ở `~/hyprefill_data/step02/moe_routing/` |
| `step02/kt2_capacity_*_estimate.csv` | KT2 (PROPOSAL §5, sửa 07/10): dung lượng prefill mỗi iteration, đồng nhất / Layered / HyPrefill, MoE **ước lượng** (cảnh báo sớm, chưa có KT1); `_kt1.csv` khi có KT1 |
| `step02/*g1c*`, `step02/oracle_*`, `step02/sim_*` | (sẽ có) G1c, oracle, mô phỏng |

## Dữ liệu cũ, không trộn với chuẩn mới (ở `~/hyprefill_data/` trên server)

| Thư mục | Là gì | Dùng được cho |
|---|---|---|
| `clk1830/results/` | Toàn bộ số đo 28–29/09 ở clock khoá 1830 MHz, KV liền mạch, state GDN luôn fp32 (cùng cấu trúc với `results/`) | Kết luận định tính và căn cứ cho các quyết định đã ghi trong `LOG.md`; so số cũ với số mới. **Không** dùng số tuyệt đối cho paper. Tóm tắt ở `plan/00`, `plan/01` mục KẾT QUẢ |
| `clk1830/raw/` | JSON từng request của demo Layered, trace profiler của lượt đó | Như trên |
| `clk1830/logs/`, `clk1830/figures/` | Log các lượt đo và hình của lượt đó | Như trên |
| `setup_logs/` | Log dựng môi trường và tải model | Tra lỗi khi dựng lại môi trường |

Đã xoá 2026-09-29: số đo trước khi khoá clock, số đo bằng cách đo cũ (một lần replay graph), các lượt chạy dở và chạy thử code. Lý do loại từng loại ghi ở `LOG.md` và mục KẾT QUẢ `plan/00`, `plan/01`.
