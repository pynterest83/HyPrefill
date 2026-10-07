# Bước 0 — Chuẩn bị môi trường và nền tảng

**Mục tiêu:** Dựng được môi trường chạy, tải xong model, đọc xong hai paper nền, và chạy được demo Layered Prefill.

---

## 1. Đầu ra bắt buộc

- [x] Môi trường `hyprefill` chạy được, import được `vllm.vllm_flash_attn` và GDN FlashInfer của vLLM
- [x] Tải xong Qwen3-Next-80B-A3B và Qwen3.8-27B; in được số layer mỗi loại từ `config.json`
- [x] Chạy được demo `scale-snu/layered-prefill` trên Qwen3-30B-A3B (chạy với 2 H200, TP = 2, như giao thức gốc của fork)
- [ ] Ghi chú 10 dòng cho Sarathi-Serve và Layered Prefill
- [x] `LOG.md` có mục ngày đầu tiên
- [x] Khoá clock GPU ở mức chuẩn **1980 MHz** và xác nhận số đo ổn định giữa hai lần chạy giống nhau *(đạt 2026-09-29: median 0.17–0.19%, 100% dòng không chạm trần < 3%, trên FA, GDN FlashInfer, Dense; riêng GDN triton của Qwen3-Next lệch 3.1% median, nghi do autotune, xem `plan/01` R3)*. Tiêu chí (sửa 2026-09-29, trước lượt đo 1980): median lệch < 1% **và** ≥ 95% số dòng lệch < 3%; dòng chạm trần 700 W (clock đo được thấp hơn mức khoá) được đánh dấu và báo riêng, không tính vào 95%. Lý do: ở mọi mức khoá, dòng chạm trần công suất dao động theo clock mà GPU giữ được lúc đó (lượt 1830: median 0.2–0.9%, 87–100% dòng < 3%, các dòng lệch đều là dòng chạm trần)
- [ ] **Kiểm chứng Layered Prefill trên H200 trước khi dùng làm baseline chính** (lượt 1830 chưa tái hiện được lợi thế của nó; chưa rõ do phần cứng hay do cách mình chạy): quét dày quanh điểm sụp (với cấu hình torch.compile mặc định như tác giả chạy; **không** nâng `recompile_limit`: lượt 2026-09-29 nâng lên 256 làm fork biên dịch lại 256 lần rồi kiểm 256 guard mỗi lần gọi, TPOT tăng gấp đôi ở cả hai chế độ, kết quả không hợp lệ), đo lưu lượng đọc expert của hai chế độ, đối chiếu cấu hình với Bảng 6 của paper. Đầu ra: goodput chunked và layered theo định nghĩa PROPOSAL §4.4

## 2. Việc chi tiết

### 2.1 Môi trường

Trên server hiện tại chạy `bash scripts/setup_server.sh` là đủ: script tự cài Miniforge nếu chưa có, cài vllm==0.30.0 (kéo theo torch 2.13.0+cu130), và đặt `HF_HOME=$HOME/hf_cache`. Các lệnh dưới đây là phần script làm, để tham khảo.

```bash
conda create -n hyprefill python=3.12 -y && conda activate hyprefill
pip install --timeout 120 --retries 10 vllm==0.30.0
pip install --only-binary=:all: transformers accelerate safetensors pandas matplotlib seaborn huggingface_hub
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm third_party/vllm
git clone https://github.com/scale-snu/layered-prefill third_party/layered-prefill
```

Không cài `flash-attn` và `flash-linear-attention` riêng: benchmark gọi đúng kernel vLLM dùng trên H200 (FA3, GDN FlashInfer), xem `AGENTS.md` mục Server.

### 2.2 Model và xác minh kiến trúc

Tải bằng `hf download` (`huggingface-cli` đã ngừng hoạt động từ huggingface_hub 1.x). Với mỗi model, in bố cục layer từ `config.json` và **đối chiếu với bảng trong PROPOSAL.md §1.2**. Số trong proposal lấy từ model card, phải xác nhận lại bằng config.

```python
import json
cfg = json.load(open("config.json"))
for k in ["num_hidden_layers", "layer_types", "linear_attn_config",
          "num_experts", "num_experts_per_tok", "decoder_sparse_step",
          "hidden_size", "num_attention_heads", "num_key_value_heads",
          "head_dim", "max_position_embeddings"]:
    if k in cfg: print(k, "=", cfg[k])
```

Ghi kết quả vào `results/step00/model_configs.md`. Riêng Qwen3.8-Flash-Next cần lấy thêm `H^I` và `d^I` của indexer QSA — đây là hai con số mà paper DSA không công bố tường minh và bước 2 sẽ cần.

### 2.3 Đọc

Hai bài, mỗi bài ghi 10 dòng (vấn đề, cơ chế, kết quả, baseline, điểm yếu):

1. Sarathi-Serve — https://arxiv.org/pdf/2403.02310 — đọc kỹ §3–4. Tìm: định nghĩa stall-free, cách chọn token budget, hiệu ứng tile quantization.
2. Layered Prefill — https://arxiv.org/pdf/2510.08055 — đọc toàn bộ. Tìm: sparsity erosion, công thức `N_lg(L)`, Bảng 6 và Bảng 7, ablation trên model dense.

Bản giảng đã soạn sẵn ở `docs/00_FOUNDATIONS.html` mục 8.1 và 8.3; đọc sau khi đọc paper gốc, không thay thế.

### 2.4 Chạy demo Layered Prefill

Mục tiêu không phải tái tạo số của họ, mà là (a) biết codebase có chạy được không, (b) ước lượng công sức thêm layer GDN vào đó ở bước 5. Ghi lại thời gian dựng và các chỗ vướng.

- **Cần env riêng**, không dùng env `hyprefill`: fork pin torch 2.8.0, CUDA toolkit 12.8 (cài qua conda), tự build `vllm-flash-attn` ở commit `d9e577e` có vá (`flash-attention.patch`) và tự compile kernel riêng (`csrc/`: MoE, norm, all-reduce). Làm theo README của fork trong env `layered-prefill`.
- Demo của họ chạy **TP = 2** (2 GPU), cùng một engine với `--schedule-mode chunked-prefill` và `--schedule-mode layered-prefill`. Chạy cả hai chế độ, trên cùng GPU, xen kẽ, để có **tỉ lệ cải thiện** của Layered so với chunked trên máy này và so với Bảng 6 của paper (TTFT −56%). Đây là bước đầu của việc kiểm chứng bản baseline (xem `docs/03_MEASUREMENT.md` §7).
- Kernel của fork **khác** kernel vLLM 0.30 mà bước 1 đo. Ghi lại danh sách kernel fork dùng; bước 5 phải quyết định dùng bộ kernel nào (xem `plan/05_DESIGN_PROTOTYPE.md` §2.7).

### 2.5 Kỷ luật đo lường

Đọc và áp dụng `docs/03_MEASUREMENT.md` **từ hôm nay**, không để đến khi có số thật.

## 3. Rủi ro của bước này

- Qwen3.8-Flash-Next có serving path mới ~1 tháng; nếu tải hoặc chạy trục trặc thì để lại, không chặn bước 1.
- Nếu build vLLM từ source quá lâu, dùng bản pip trước, build sau.


---

## KẾT QUẢ

> Điền 2026-09-29, cập nhật 2026-10-07 với số đo ở 1980 MHz (số 1830 MHz cũ ở `~/hyprefill_data/clk1830/`, không dùng nữa). Còn thiếu: ghi chú 10 dòng cho Sarathi-Serve và Layered Prefill (việc đọc paper).

### R1. Số liệu chính

| Đại lượng | Giá trị | Ghi chú |
|---|---|---|
| Môi trường | vllm 0.30.0, torch 2.13.0+cu130, flashinfer 0.6.18; env riêng `layered-prefill` (torch 2.8.0+cu128) cho fork | `scripts/setup_server.sh`, `scripts/setup_layered_prefill.sh` |
| Bố cục model | Qwen3-Next 12 attention + 36 GDN + 48 MoE; Qwen3.8-27B 16 + 48, FFN dense; Qwen3.8-Flash-Next 12 QSA + 36 GDN; Kimi-Linear 7 MLA + 20 KDA (2.9 : 1, không đúng 3 : 1) | `results/step00/model_configs.md` |
| Indexer Qwen3.8-Flash-Next | H^I = 4, d^I = 128, nén 4, budget 2048 | Mô phỏng giả định H = 64, không nén |
| Clock | GPU 4–7 khoá 1980 MHz (admin); root trong container không khoá được; FA3 ở context dài vẫn tụt tới ~1140 MHz vì trần 700 W | `docs/03_MEASUREMENT.md` §1 |
| Độ lặp lại (bước 1, 1980 MHz) | median lệch 0.17–0.19% giữa hai card; 100% dòng không chạm trần < 3% | Đạt tiêu chí bước 0 |
| Demo Layered (Qwen3-30B-A3B, arXiv, 2×H200, 1980 MHz, cấu hình mặc định của tác giả) | Tỉ lệ đạt SLO (TTFT ≤ 10 s, TBT ≤ 125 ms) ở 2.6 / 2.7 / 2.8 req/s: chunked 98.5 / 90.7 / 55.6%, layered 99.7 / 83.2 / 21.2%. Goodput: chunked ~2.7, layered ~2.6 req/s | Paper (2×H100): layered 1.7, chunked 1.5 req/s. **Lợi thế goodput không tái hiện trên H200**. `results/step00/layered_demo/2026-09-30/summary.csv` |
| TTFT trung bình ở tải chưa bão hoà | layered chậm hơn chunked 7–17% (1.3 / 2.0 / 2.5 req/s) | Không mâu thuẫn với claim của paper: họ claim tỉ lệ đạt SLO, không claim TTFT ở tải nhẹ |
| Năng lượng mỗi token (cùng workload, 9 mức tải) | layered ít hơn chunked 9.9–12.7% (vd 2.5 req/s: 43.7 so với 49.4 mJ/token) | Paper ~22%. **Hiện tượng tiết kiệm có tái hiện**, nhỏ hơn. `energy.csv` cùng thư mục |
| Profile nsys (2.5 req/s, cửa sổ 60 s giữa tải, cả hai bão hoà dưới nsys; cộng 2 rank) | Thời gian kernel GPU: chunked 90.0 s, layered 79.9 s (−11%). MoE 30.4 → 22.8 s (−25%; `fused_moe_kernel` 28.1 → 20.8 s); attention 22.5 → 17.5 s (−22%); all-reduce 19.2 → 20.5 s; GEMM, norm, khác gần như không đổi | **Hiện tượng giảm đọc lại expert có tái hiện ở phía GPU.** `results/step00/layered_profile/2026-09-30/` |
| Phía CPU trong cùng cửa sổ (NVTX, cộng 2 rank) | `prepare` 4.3 → 12.5 s (0.9 → 3.05 ms mỗi step), `sample` 3.5 → 8.4 s; GPU bận 75% (chunked) so với 67% (layered) thời gian thực | Phần GPU tiết kiệm được (~5 s mỗi rank mỗi phút) bị phần CPU tăng thêm (~6.5 s) ăn hết |

### R2. Hình sinh ra

| File | Nội dung | Dùng cho hình nào của paper |
|---|---|---|
| (chưa vẽ) | Tỉ lệ đạt SLO theo mức tải, chunked và layered, từ `results/step00/layered_demo/2026-09-29/summary.csv` | Không; kiểm chứng baseline |

### R3. Khác dự đoán / bất ngờ

- Cùng 1.3 req/s như Bảng 6 của paper nhưng trên H200 là tải nhẹ (TTFT chunked 0.6 s so với 2.8 s trên H100); phải so ở cùng mức tải tương đối, tức quét tải.
- Trên H200, Layered không có lợi thế goodput với cấu hình của chính fork, **nhưng cơ chế của nó vẫn hoạt động**: MoE −25%, attention −22%, tổng kernel GPU −11%, năng lượng −10…−13%. Lợi ích không thành goodput vì fork bị giới hạn bởi CPU: chế độ layered tăng thời gian `prepare` gấp ~3.4 lần và `sample` gấp ~2.7 lần mỗi step, GPU bận giảm từ 75% xuống 67%. Trên H100 (HBM chậm hơn ~30%) phần GPU tiết kiệm lớn hơn so với phần CPU, nên paper thấy được lợi thế. Đây là giải thích khớp với số đo, chưa kiểm bằng thí nghiệm riêng (vd fork trên H100, hoặc giảm phần CPU của layered).
- Nâng `recompile_limit` lên 256 làm TPOT tăng gấp đôi; số dùng ở trên chạy mặc định như tác giả.
- Hệ quả cho HyPrefill: (1) cài Layered/HyPrefill trong vLLM phải giữ phần CPU mỗi iteration nhỏ, nếu không phần GPU tiết kiệm bị ăn mất như ở fork; (2) trên H200 phần đọc lại expert nhỏ hơn trên H100, nên lợi ích dựa vào MoE cần đo trên chính máy này, không suy từ paper.
- flash-attn 2.8.3 không build được với torch mới; fork cần CUDA 12.8 đúng bản (trộn kênh conda cho ra nvcc 12.4).
- Khi tắt server của fork, các tiến trình con của engine không tắt theo và giữ ~120 GB GPU; phải tắt cả process group.

### R4. Quyết định rút ra

- Đo bằng kernel của vLLM 0.30 (FA3, GDN FlashInfer, MoE Triton), không dùng flash-attn/fla riêng.
- Chốt benchmark (PROPOSAL §4.4): headline là goodput theo định nghĩa SLO của paper Layered, so với baseline tốt nhất từng ô, số headline trên vLLM 0.30.
- Mọi việc chạy lâu chạy trong tmux; script đo dừng hẳn nếu clock chưa khoá.

### R5. Việc chuyển sang bước sau

- Ghi chú đọc Sarathi-Serve và Layered Prefill.
- Layered trong vLLM (bước 5) cần đo riêng chi phí CPU mỗi iteration so với chunked; mục tiêu là phần CPU không tăng đáng kể so với chunked.
- Mô hình hoá phần CPU mỗi iteration (bước 1 còn mở) cần cả cho chế độ chạy theo nhóm layer.

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
