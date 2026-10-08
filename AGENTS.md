# AGENTS.md

File này hướng dẫn các AI coding agent làm việc với code trong repository này.

## Repo này là gì

Một dự án nghiên cứu (mục tiêu: ICML 2027), không phải sản phẩm phần mềm. Ý tưởng chính là **chunked prefill tách theo operator cho LLM hybrid** (attention + linear attention GDN/KDA + MoE): mỗi nhóm operator có chunk size riêng, lấy từ cost model hiệu chỉnh offline, thay vì một chunk đồng nhất bằng `min` trên mọi ràng buộc. Phần lớn repo là tài liệu kế hoạch và tham khảo. Code gồm: script dựng môi trường (`scripts/`), bộ mô phỏng dòng token (`sim/`), và benchmark ở `bench/`: `op_cost.py` (cost từng op bằng kernel vLLM), `analyze_step01.py` (lặp lại, gộp bảng, hình), `oracle.py`, `validate_forward.py`, `profile_vllm_step.py` (có `--sweep`), `allreduce_cost.py`, `moe_routing_dump.py` + `moe_overlap.py` (routing MoE thật), `cgroup_cpu.py` (ghi CPU bị cgroup bóp), `sim_tokenflow.py`, các script chạy `run_*.sh` và `step00_layered_demo.sh`. Danh mục kết quả hợp lệ ở `results/README.md`.

Mọi tài liệu văn xuôi (README, PROPOSAL, plan/, LOG, docs/) viết bằng **tiếng Việt**. Khi sửa thì giữ tiếng Việt. Code và comment trong code viết bằng tiếng Anh.

## Lệnh

```bash
# Dựng server một lần (8×H200): conda env `hyprefill` với vllm==0.30.0 (kéo theo torch 2.13.0+cu130),
# clone vllm (cùng tag) + layered-prefill vào third_party/, chạy kiểm tra môi trường, tải model bước 0.
bash scripts/setup_server.sh              # HF_HOME mặc định là $HOME/hf_cache
bash scripts/setup_server.sh --no-models

# Kiểm tra nhanh: phiên bản thư viện, GPU, đo thử FA3 và GDN FlashInfer của vLLM
python scripts/check_env.py

# Toàn bộ campaign đo (clock khoá 1980 MHz, GPU 4–7): 4 hàng đợi tmux, xem đầu file
tmux new -d -s cA 'QUEUE=A bash bench/run_campaign_1980.sh'      # tương tự B, DEMO, POST
bash scripts/record_env.sh                                        # ảnh chụp phiên bản gói + GPU vào results/env/

# Bộ mô phỏng dòng token: bản Python dùng bảng cost thật (bench/sim_tokenflow.py);
# bản JS gốc (sim/hyprefill_sim.js) chỉ có hằng số minh hoạ, cần Node
python bench/sim_tokenflow.py --model Qwen3-Next-80B-A3B-Instruct_tp2
```

Không có build, lint hay bộ test. Dựng trên máy khác: `docs/11_SETUP_NEW_SERVER.md`.

## Các phần ghép với nhau thế nào

- **PROPOSAL.md** chứa luận điểm. §2.2 là cost model (`r_u(t)` đồng nhất so với `r_d(t)` tách, với `k_g ∈ {1,2,4,8,16}`), §2.3 là cơ chế, §4 là kế hoạch thực nghiệm (model, workload, baseline, ablation, hình cốt lõi), §5 là các cổng quyết định. Các file khác đều trỏ về đây, nên đọc nó trước khi sửa claim ở bất kỳ đâu.
- **plan/NN_*.md** là kế hoạch chia theo bước, làm lần lượt theo số thứ tự, không gắn ngày hay tuần. Mỗi file có mục tiêu, checklist đầu ra bắt buộc, việc chi tiết kèm lệnh, tiêu chí kiểm tra, và phần **KẾT QUẢ** để trống, điền khi làm xong. **LOG.md** là nhật ký 5 dòng mỗi ngày. Cả hai là template để điền, không đổi cấu trúc.
- **sim/hyprefill_sim.js** so sánh ba chính sách dưới cùng budget prefill mỗi iteration `P`: `sarathi` (một chunk, đi hết chiều sâu trong một iteration), `layered` (pipeline theo chiều sâu, cùng chunk cho mọi operator, k=1) và `hyprefill` (pipeline theo chiều sâu, attention chunk `c`, các operator khác `k·c`). Bố cục stack giống Qwen3-Next (12 × `[G,M,G,M,G,M,A,M]`). **Mọi hằng số cost chỉ là giá trị minh hoạ.** Phải thay bằng số đo bước 1–2 trước khi rút ra kết luận. Bộ mô phỏng có mục đích tách pipeline theo chiều sâu (gain của Layered Prefill) khỏi chunk theo operator (gain riêng của HyPrefill). Vì phải tách hai cơ chế này nên cổng G1 chia thành G1a, G1b và G1c.
- **Cách đặt vấn đề then chốt:** so sánh headline là **HyPrefill / Layered Prefill**, không phải HyPrefill / Sarathi. So với Sarathi sẽ gán nhầm gain của pipeline theo chiều sâu cho HyPrefill (PROPOSAL §4.3).
- **Ràng buộc kernel:** HyPrefill chỉ đổi số token scheduler đưa vào mỗi lần gọi. Không bao giờ đổi chunk size của kernel GDN (`FLA_CHUNK_SIZE` = 64). Chunk size của GDN là bội của 64; chunk của attention nên là bội của 128, vì FA3 có bậc thang theo tile 128 (bước 1: c = 129 đắt hơn c = 128 rõ rệt).

## Server hiện tại (hyprefill-dev-0, dựng 08/10/2026)

Máy mới thay `quangch1-dev-0`: user `hyprefill`, `$HOME=/home/hyprefill` (NVMe 3.5 TB), repo ở `~/work/HyPrefill`, 8×H200, 192 luồng CPU, 2 TiB RAM, `sudo` không cần mật khẩu, CUDA 13.0 ở `/usr/local/cuda`. Bố cục NUMA/GPU giống máy cũ (GPU0-1 → CPU 0-23, 2-3 → 24-47, 4-5 → 48-71, 6-7 → 72-95). Khác máy cũ:
- cgroup cho **128 core** (`cpu.max` = 12800000/100000), không phải 32.
- `/mnt/models/hf` chỉ có model nhỏ (Qwen3-0.6B, Qwen3.5-2B…), **không có** model của HyPrefill. Ba model bước 0 (Qwen3-Next-80B-A3B-Instruct, Qwen3.8-27B, Qwen3-30B-A3B, tổng 261 GB) đã tải về `~/hf_cache`. Đã tải thêm Qwen3.8-Flash-Next-FP8 và Kimi-Linear-48B-A3B-Instruct (tổng `~/hf_cache` 525 GB). `gh` đã đăng nhập (pynterest83), Node 18 đã cài.
- Env `hyprefill` (vllm 0.30.0, torch 2.13.0+cu130) và `layered-prefill` (torch 2.8.0+cu128, flash-attention build xong) đã dựng; nsys 2026.3.2 đã cài bằng apt (mất khi pod restart). Ảnh chụp môi trường: `results/env/2026-10-08_hyprefill-dev-0/`.
- `check_env.py` đã chạy OK trên GPU 4 (FA3, GDN FlashInfer). Clock **đã được khoá 1980 MHz** (GPU nhàn rỗi đứng ở 1980, 08/10/2026). GPU 0–3 thường có job của container khác; GPU 4–7 trống lúc kiểm tra: luôn xem `memory.used` và `utilization.gpu` trước khi đo. Chưa có lượt đo nào trên máy này; số đo là bộ số mới, không trộn với `~/hyprefill_data/clk1830/` hay số của máy cũ.
- Các script điều phối gán sẵn GPU 4–7 (`run_campaign_1980.sh`, `run_gpu45_*.sh`, `run_gpu67_*.sh`): xem GPU nào thực sự trống rồi truyền `GPU=`/`CPUS=`.

Phần dưới là ghi chú của máy cũ; đường dẫn `/home/quangch1` tương ứng `$HOME` ở máy mới.

## Server cũ (quangch1-dev-0, kiểm tra 28/09/2026)

Là container chạy trên máy 8×H200: 192 luồng CPU, 2 TiB RAM, 4 NUMA node, có `sudo` không cần mật khẩu. **Nhưng root trong container không khoá được clock GPU** (`nvidia-smi -lgc` báo không có quyền), phải nhờ admin khoá ở host. **Mức khoá chuẩn: 1980 MHz** (mức boost tối đa của H200, từ 2026-09-29; lúc nhàn rỗi clock đứng ở 1980, chưa khoá thì ~345). Số đo cũ ở 1830 MHz (28–29/09) đã chuyển sang `~/hyprefill_data/clk1830/`, không trộn với số mới. `bench/run_step01.sh` tự nhận ra khoá này và không mở nó. Khoá vẫn tụt khi chạm trần 700 W (FA3 ở context dài), nên clock và công suất được ghi theo từng dòng đo. Số đưa vào paper phải đo khi clock đã khoá; `bench/run_step01.sh` dừng hẳn nếu khoá thất bại, còn `LOCK=0` chỉ dùng cho lượt đo tham khảo.

```
/home/quangch1/                  NVMe RAID0 7 TB, ghi được. RAID0 không có dự phòng
├── miniforge3/                  conda (setup_server.sh tự cài nếu chưa có)
├── hf_cache/                    HF_HOME: model tải về, tải lại được nếu mất
├── hyprefill_data/stepNN/<ngày>/  dump thô, nsys/ncu, trace (ngoài repo)
└── work/HyPrefill/              repo; results/ commit và push sau mỗi phiên đo
/mnt/models/hf/                  kho model dùng chung, chỉ đọc: trỏ thẳng đường dẫn, không copy, không ghi
```

- CUDA toolkit 13.0 ở `/usr/local/cuda` (không có sẵn trong PATH). Torch là bản cu130 do vLLM pin.
- Không dùng package `flash-attn` và `fla` riêng: flash-attn 2.8.3 không có wheel cho torch mới và build lỗi (cần C++20). Mọi benchmark gọi đúng kernel vLLM dùng khi serve trên H200: FA3 qua `vllm.vllm_flash_attn`, GDN qua FlashInfer (`fi_chunk_gated_delta_rule` trong `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`).
- pypi.org hay timeout từ server này (riêng trang `simple/python-dateutil/` treo hẳn). `pip install` dùng `--timeout 120 --retries 10` và `--only-binary=:all:`; tải model bằng `hf download` (`huggingface-cli` đã ngừng hoạt động).
- Container bị giới hạn **32 core CPU** bởi cgroup (`/sys/fs/cgroup/cpu.max`). Vòng lặp engine là Python đơn luồng nên thêm core không giúp gì, nhưng bị bóp CPU thì GPU phải chờ. Mọi script đo ghi số lần bị bóp (`bench/cgroup_cpu.py`, `docs/03_MEASUREMENT.md` §1); lượt nào có `nr_throttled` > 0 thì kiểm lại số đo theo thời gian thực.
- GPU dùng chung với container khác. Tiến trình của container khác không hiện trong `nvidia-smi`, nên trước khi đo phải xem `memory.used` và `utilization.gpu`. GPU đang có tải thì số đo không dùng được.
- Việc chạy lâu (tải model, chạy benchmark dài) phải chạy trong tmux để không chết khi VS Code/SSH ngắt: `tmux new -d -s hyprefill-dl 'bash scripts/download_models.sh [repo ...]'`, xem bằng `tmux attach -t hyprefill-dl`, log ở `~/hyprefill_data/download_*.log`.
- **Nsight Systems 2026.3.2** cài bằng `sudo apt-get install -y nsight-systems-2026.3.2` (kho CUDA của NVIDIA đã có trong `/etc/apt/sources.list.d/cuda.list`; bản trong kho Ubuntu là 2022.4, quá cũ). **Mọi thứ cài bằng apt nằm ngoài `$HOME` mất khi pod khởi động lại** (đã mất nsys ngày 07/10), còn conda env, model và dữ liệu trong `$HOME` thì giữ; sau mỗi lần pod restart chạy lại lệnh trên và kiểm lại khoá clock. Dùng `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none`: lấy mẫu CPU không chạy được trong container (perf_event_open bị chặn, paranoid level 4), còn ghi CUDA và NVTX thì được.
- Khi đo, ghim CPU cùng NUMA node với GPU: GPU0-1 → CPU 0-23, GPU2-3 → 24-47, GPU4-5 → 48-71, GPU6-7 → 72-95 (ví dụ `CUDA_VISIBLE_DEVICES=4 taskset -c 48-71 python ...`).

## Quy tắc đo (docs/03_MEASUREMENT.md)

Mọi code benchmark phải theo các quy tắc sau:
- Đo thời gian bằng `torch.cuda.Event`, không bao giờ dùng `time.time()`. Warmup ≥ 10 lần, chạy ≥ 30 lần, báo p50/p99/min/max. Khoá clock GPU (`sudo nvidia-smi -lgc 1980,1980`) hoặc ghi lại clock trong lúc chạy.
- Chi phí kernel = thời gian **replay CUDA graph chứa K lần [xoá L2 → op]**, trừ graph chỉ xoá L2, chia K (`gpu_time` trong `bench/op_cost.py`; chi tiết `docs/03_MEASUREMENT.md` §2). Không đo event quanh một lần gọi eager, cũng không đo event quanh một lần replay graph (lẫn ~4 µs CPU phát lệnh). Với các kernel nhỏ, gọi eager bị giới hạn bởi CPU: wrapper GDN FlashInfer tốn ~0.2 ms CPU mỗi lần gọi trong khi GPU chỉ ~0.04 ms ở c = 64, và tốc độ CPU trên máy dùng chung dao động, từng tạo ra một phụ thuộc t giả. `bench/op_cost.py` ghi cả ba: GPU (graph), eager, và `host_ms` (thời gian CPU đẩy một lần gọi; quan trọng vì vLLM chạy GDN và attention ngoài CUDA graph).
- FA đo với KV **paged** theo đúng block size vLLM chọn; state GDN theo `mamba_ssm_dtype` của config (chi tiết `docs/03_MEASUREMENT.md` §2).
- Với GDN, luôn truyền `initial_state` khác None vào `chunk_gated_delta_rule`. Không truyền thì đang đo prefill từ đầu, không phải chunked prefill.
- Với indexer/sparse attention, đo bộ nhớ đỉnh bằng cả `max_memory_allocated()` và `max_memory_reserved()`.
- Kết quả lưu ở `results/stepNN/<ngày>/{config.json, raw.csv, summary.csv}`. **Không bao giờ ghi đè.** Chạy lại thì tạo thư mục ngày mới. `config.json` ghi model, shape, phiên bản thư viện, commit, lệnh chạy và clock có khoá hay không.
- So sánh với baseline theo `docs/03_MEASUREMENT.md` §7: chạy lại mọi baseline trên chính máy này (không dùng số trong paper), cùng GPU/TP/NUMA/kernel/cấu hình engine, tune baseline tốt nhất (static quét chunk, Layered quét `N_lg`, SLOWeave tune δ), chạy xen kẽ, cost model đo trên đúng bộ kernel của engine đang đánh giá.
- Fork `third_party/layered-prefill` cần env riêng (torch 2.8.0, CUDA 12.8, `vllm-flash-attn` tự build) và dùng kernel khác vLLM 0.30, nên **không** so trực tiếp với HyPrefill (`plan/05` §2.7, §3).
- CSV và JSON nhỏ trong `results/` được commit. Dump thô (`.npy`, `.parquet`, `.pkl`, report nsys/ncu) và weight model nằm trong gitignore, để trên server.

## Kỷ luật nghiên cứu

- Viết tiêu chí GO/KILL **trước** khi chạy thí nghiệm, không bao giờ sửa sau khi thấy số.
- Báo cáo cả các vùng HyPrefill không thắng.
- Mọi policy (chunked, Layered, HyPrefill, SLOWeave) chạy trong **cùng vLLM 0.30**; Layered là cấu hình k = 1 của cùng hạ tầng chạy theo nhóm layer với HyPrefill. Fork nanovllm chỉ để kiểm chứng bản Layered cài lại, bảng tham khảo khác engine, và phương án lui.
- Cổng cứng sau bước 8: chưa có số end-to-end trong vLLM → phương án lui trên fork nanovllm nếu còn thời gian, nếu không viết bài cost model + simulator. Các phương án lui có trong README và PROPOSAL §7.
