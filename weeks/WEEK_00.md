# Tuần 00 — Chuẩn bị môi trường và nền tảng

**Ngày:** 15–21/09/2026
**Mục tiêu:** Dựng được môi trường chạy, tải xong model, đọc xong hai paper nền, và chạy được demo Layered Prefill.

---

## 1. Đầu ra bắt buộc

- [ ] Môi trường `hyprefill` chạy được, import được `flash_attn` và `fla`
- [ ] Tải xong Qwen3-Next-80B-A3B và Qwen3.8-27B; in được số layer mỗi loại từ `config.json`
- [ ] Chạy được demo `scale-snu/layered-prefill` trên Qwen3-30B-A3B với 1 H200
- [ ] Ghi chú 10 dòng cho Sarathi-Serve và Layered Prefill
- [ ] `LOG.md` có mục ngày đầu tiên
- [ ] Khoá clock GPU và xác nhận số đo ổn định giữa hai lần chạy giống nhau (sai lệch < 3%)

## 2. Việc chi tiết

### 2.1 Môi trường

```bash
conda create -n hyprefill python=3.12 -y && conda activate hyprefill
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install flash-linear-attention
pip install transformers accelerate safetensors pandas matplotlib seaborn
git clone https://github.com/vllm-project/vllm && cd vllm && pip install -e . && cd ..
git clone https://github.com/scale-snu/layered-prefill
```

### 2.2 Model và xác minh kiến trúc

Tải bằng `huggingface-cli download`. Với mỗi model, in bố cục layer từ `config.json` và **đối chiếu với bảng trong PROPOSAL.md §1.2**. Số trong proposal lấy từ model card, phải xác nhận lại bằng config.

```python
import json
cfg = json.load(open("config.json"))
for k in ["num_hidden_layers", "layer_types", "linear_attn_config",
          "num_experts", "num_experts_per_tok", "decoder_sparse_step",
          "hidden_size", "num_attention_heads", "num_key_value_heads",
          "head_dim", "max_position_embeddings"]:
    if k in cfg: print(k, "=", cfg[k])
```

Ghi kết quả vào `results/week00/model_configs.md`. Riêng Qwen3.8-Flash-Next cần lấy thêm `H^I` và `d^I` của indexer QSA — đây là hai con số mà paper DSA không công bố tường minh và tuần 2 sẽ cần.

### 2.3 Đọc

Hai bài, mỗi bài ghi 10 dòng (vấn đề, cơ chế, kết quả, baseline, điểm yếu):

1. Sarathi-Serve — https://arxiv.org/pdf/2403.02310 — đọc kỹ §3–4. Tìm: định nghĩa stall-free, cách chọn token budget, hiệu ứng tile quantization.
2. Layered Prefill — https://arxiv.org/pdf/2510.08055 — đọc toàn bộ. Tìm: sparsity erosion, công thức `N_lg(L)`, Bảng 6 và Bảng 7, ablation trên model dense.

Bản giảng đã soạn sẵn ở `docs/00_FOUNDATIONS.html` mục 8.1 và 8.3; đọc sau khi đọc paper gốc, không thay thế.

### 2.4 Chạy demo Layered Prefill

Mục tiêu không phải tái tạo số của họ, mà là (a) biết codebase có chạy được không, (b) ước lượng công sức thêm layer GDN vào đó ở tuần 5. Ghi lại thời gian dựng và các chỗ vướng.

### 2.5 Kỷ luật đo lường

Đọc và áp dụng `docs/03_MEASUREMENT.md` **từ hôm nay**, không để đến khi có số thật.

## 3. Rủi ro tuần này

- Qwen3.8-Flash-Next có serving path mới ~1 tháng; nếu tải hoặc chạy trục trặc thì để lại, không chặn tuần 1.
- Nếu build vLLM từ source quá lâu, dùng bản pip trước, build sau.


---

## KẾT QUẢ

> **Để trống — điền khi làm xong tuần này.**

### R1. Số liệu chính

| Đại lượng | Giá trị | Ghi chú |
|---|---|---|
|  |  |  |

### R2. Hình sinh ra

| File | Nội dung | Dùng cho hình nào của paper |
|---|---|---|
|  |  |  |

### R3. Khác dự đoán / bất ngờ

-

### R4. Quyết định rút ra

-

### R5. Việc chuyển sang tuần sau

-

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
