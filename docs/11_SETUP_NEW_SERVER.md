# Dựng HyPrefill trên một server mới

Hướng dẫn dựng lại toàn bộ môi trường trên một máy khác `quangch1-dev-0`. Phần lớn đã tự động trong `scripts/setup_server.sh`; tài liệu này gom các bước thủ công và các điểm phụ thuộc vào máy.

> **Đọc trước:** số đo phụ thuộc phần cứng. Trên máy mới, **mọi bảng cost và kết quả đo phải đo lại** (bước 1–2, demo bước 0) và lưu thành thư mục ngày mới; không trộn với số của máy cũ. Nếu GPU không phải H200 thì ghi rõ trong `config.json` và `results/env/`, và coi là một bộ số riêng (PROPOSAL §4, `docs/03_MEASUREMENT.md` §7).

## 0. Yêu cầu

| Thứ | Cần | Ghi chú |
|---|---|---|
| GPU | ≥ 2 GPU Hopper (H100/H200), driver hỗ trợ CUDA 13 | Qwen3-Next bf16 cần TP2 trên 141 GB; TP4 nếu GPU 80 GB |
| Đĩa | ≥ 800 GB cho 3 model chính (khoảng 2 TB nếu thêm Flash-Next, Kimi-Linear) | `HF_HOME` đặt ở ổ lớn ghi được |
| CPU | ≥ 32 core cho container | Vòng lặp engine là Python đơn luồng; tốc độ đơn luồng quan trọng hơn số core |
| Mạng | HuggingFace, GitHub, pypi | pypi có thể chậm, script đã có timeout dài |
| Quyền | `sudo` (cài nsys); khoá clock GPU thường phải nhờ admin host | |

## 1. Lấy code

Repo `pynterest83/HyPrefill` là private, cần một trong các cách xác thực:

```bash
# cách 1: GitHub CLI
sudo apt-get install -y gh && gh auth login
gh repo clone pynterest83/HyPrefill ~/work/HyPrefill

# cách 2: SSH key (thêm ~/.ssh/id_ed25519.pub vào GitHub, hoặc làm deploy key cho repo)
ssh-keygen -t ed25519 && cat ~/.ssh/id_ed25519.pub
git clone git@github.com:pynterest83/HyPrefill.git ~/work/HyPrefill
```

Có xác thực trong shell thì agent cũng push/pull được, không phải đi qua VS Code.

## 2. Môi trường chính (`hyprefill`, vLLM 0.30)

```bash
cd ~/work/HyPrefill
export HF_HOME=/duong/dan/o/lon/hf_cache        # mặc định $HOME/hf_cache
tmux new -d -s setup 'bash scripts/setup_server.sh 2>&1 | tee ~/setup.log'   # có tải 3 model
# hoặc: bash scripts/setup_server.sh --no-models, rồi tải riêng:
tmux new -d -s dl 'bash scripts/download_models.sh'
```

Script tự làm: cài Miniforge nếu thiếu, tạo env `hyprefill` (Python 3.12, `vllm==0.30.0` kéo theo torch 2.13.0+cu130), clone `third_party/vllm` (cùng tag) và `third_party/layered-prefill`, chạy `scripts/check_env.py` (đo thử FA3 và GDN FlashInfer). Chạy lại an toàn, bước nào xong thì bỏ qua.

Thêm vào `~/.bashrc` (script in ra cuối):

```bash
export HF_HOME=...; export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
```

Gói thêm cho một số script: `pip install --timeout 120 --retries 10 datasets` (dump routing, demo arXiv).

## 3. Env của fork Layered (chỉ cần cho bước 0 và phương án lui)

```bash
tmux new -d -s lp 'bash scripts/setup_layered_prefill.sh'     # build flash-attention, khoảng 1 giờ
```

Env riêng `layered-prefill` (torch 2.8.0, CUDA 12.8 từ conda-forge). Biến `BUILD_CPUS` mặc định `0-23,96-119` theo NUMA của máy cũ; sửa theo §5.

## 4. Công cụ hệ thống

```bash
# Nsight Systems: kho CUDA của NVIDIA (bản kho Ubuntu quá cũ)
sudo apt-get install -y nsight-systems-2026.3.2    # nếu chưa có kho: thêm cuda-keyring của NVIDIA trước
nsys --version
```

Trong container: chỉ dùng `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none` (lấy mẫu CPU bị chặn). **Gói apt nằm ngoài `$HOME` có thể mất khi pod khởi động lại**; conda env, model và dữ liệu trong `$HOME` thì còn. Sau mỗi lần restart: cài lại nsys, kiểm lại khoá clock.

## 5. Những chỗ phụ thuộc máy phải kiểm và sửa

**NUMA và ghim CPU.** Xem `nvidia-smi topo -m`, cột "CPU Affinity". Máy cũ: GPU 0-1 → CPU 0-23, 2-3 → 24-47, 4-5 → 48-71, 6-7 → 72-95. Các script nhận biến môi trường, đừng sửa code:

```bash
GPU=2 CPUS=24-47 bash bench/run_step01.sh
GPUS=2,3 CPUS=24-47 bash bench/step00_layered_demo.sh
```

Các script đã gán sẵn GPU 4–7 / CPU 48–95 (`run_campaign_1980.sh`, `run_gpu45_*.sh`, `run_gpu67_*.sh`) là script điều phối của máy cũ: sửa số GPU/CPU ở đầu file hoặc truyền biến trước khi dùng.

**Clock.** Thử `sudo nvidia-smi -i <gpu> -lgc 1980,1980`. Trong container thường không được, phải nhờ admin host khoá. Kiểm: lúc nhàn rỗi `nvidia-smi --query-gpu=clocks.sm --format=csv` phải đứng ở mức khoá (chưa khoá thì khoảng 345 MHz). Mức 1980 MHz là của H200; GPU khác thì chọn mức boost tối đa mà GPU giữ được, ghi vào `docs/03_MEASUREMENT.md` §1, đặt `LOCK_MHZ=<mức>` khi chạy script.

**GPU dùng riêng.** Trước khi đo xem `memory.used` và `utilization.gpu` (tiến trình container khác không hiện trong `nvidia-smi`). Máy cũ từng bị job khác cùng tài khoản chiếm GPU giữa chừng; thống nhất trước GPU nào dành cho đo.

**CPU của container.** `cat /sys/fs/cgroup/cpu.max` cho biết quota. Mọi script đo ghi số lần bị bóp (`bench/cgroup_cpu.py`); lượt nào `nr_throttled` > 0 thì kiểm lại số theo thời gian thực.

**Block size KV và kiểu state GDN** do vLLM chọn theo model và TP (Qwen3.8-27B TP1: 784, Qwen3-Next TP2: 544); đổi TP thì kiểm lại (`docs/03_MEASUREMENT.md` §2).

## 6. Dữ liệu không có trong repo

| Dữ liệu | Ở máy cũ | Trên máy mới |
|---|---|---|
| Weight model | `$HF_HOME` | `scripts/download_models.sh` |
| Dataset arXiv, ShareGPT | cache của `datasets` | tự tải lần đầu |
| Trace `semianalysisai/cc-traces-weka-062126-256k` | chưa tải | `hf download --repo-type dataset ...` khi tới bước 4 |
| Dump routing MoE (`~/hyprefill_data/step02/moe_routing/`) | dump thô, không commit | chạy lại `bench/moe_routing_dump.py` (cần cho `moe_overlap.py`, `op_cost.py --routing`) hoặc copy bằng `rsync` |
| Trace nsys, log | `~/hyprefill_data/` | không cần, trừ khi muốn xem lại |

Kết quả nhỏ (CSV, `config.json`) đã commit trong `results/`; đọc danh mục ở `results/README.md`.

## 7. Kiểm tra sau khi dựng

```bash
conda activate hyprefill
python scripts/check_env.py                        # phiên bản, GPU, FA3 + GDN chạy được
bash scripts/record_env.sh results/env/$(date +%F)_<tenmay>   # ảnh chụp môi trường để commit
GPU=0 CPUS=0-23 LOCK=0 bash bench/run_step01.sh    # lượt thử, LOCK=0 chỉ để tham khảo
python bench/kt2_capacity.py --decode 32 --budgets 50 --t 131072 --delta 4096 --h0 10   # chỉ CPU
```

Sau đó làm tiếp theo kế hoạch (`plan/00` → ...), đo lại từ bước 0 trên máy mới theo đúng chuẩn đo, rồi mới so với số của máy cũ.
