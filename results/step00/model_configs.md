# Bố cục model đọc từ config.json

Nguồn: `config.json` tải thẳng từ HuggingFace (`resolve/main`) ngày 2026-09-28, lưu nguyên bản ở `configs/`. Chưa tải weight.

## Bố cục layer

| Model | Layer | Linear attention | Attention | Đối chiếu PROPOSAL §1.2 |
|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 48 | 36 GDN | 12 gated attention (`full_attention_interval = 4`) | Khớp: 12 × (3 GDN + 1 attention) |
| Qwen3.8-27B | 64 | 48 GDN | 16 gated attention | Khớp: 16 × (3 + 1), FFN dense |
| Qwen3.8-Flash-Next | 48 | 36 GDN | 12 Qwen Sparse Attention | Khớp: 12 × (3 + 1) |
| Kimi-Linear-48B-A3B | 27 | 20 KDA | 7 MLA (layer 4, 8, …, 24, 27) | Gần khớp: tỉ lệ 20 : 7 ≈ 2.9 : 1, không đúng 3 : 1 |
| Qwen3-30B-A3B | 48 | — | 48 full attention | Khớp |

## Shape dùng cho micro-benchmark

| Model | hidden | Attention: heads / kv heads / head_dim | GDN/KDA: k heads / v heads / d_k / d_v | MoE: expert / active / intermediate |
|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 2048 | 16 / 2 / 256 | 16 / 32 / 128 / 128 | 512 / 10 + 1 shared / 512 |
| Qwen3.8-27B | 5120 | 24 / 4 / 256 | 16 / 48 / 128 / 128 | dense, intermediate 17408 |
| Qwen3.8-Flash-Next | 2560 | 24 / 2 / 256 | 16 / 48 / 128 / 128 | 512 / 10 + 1 shared / 640 |
| Kimi-Linear-48B-A3B | 2304 | MLA 32 heads, kv_lora_rank 512, qk 128 + 64 rope, v 128 | 32 / 32 / 128 / 128 | 256 / 8 + 1 shared / 1024; layer đầu dense |
| Qwen3-30B-A3B | 2048 | 32 / 4 / 128 | — | 128 / 8 / 768 |

Mọi model Qwen có `linear_conv_kernel_dim = 4` và `max_position_embeddings = 262144`, trừ Qwen3-30B-A3B là 40960.

## Indexer của Qwen3.8-Flash-Next

Hai con số `H^I` và `d^I` mà paper DSA không công bố:

| Tham số | Giá trị |
|---|---|
| `indexer_n_heads` (H^I) | **4** |
| `indexer_head_dim` (d^I) | **128** |
| `indexer_kv_heads` | 1 |
| `indexer_compress_ratio` | 4 |
| `indexer_budget` | 2048 |

**Ảnh hưởng tới cổng G1c và G1b.** `sim/hyprefill_sim.js` đang giả định `H_IDX = 64` và không nén key. Nếu kernel thật cấp phát buffer logits cỡ `c · (t / ratio) · H^I · 4` byte, buffer nhỏ hơn giả định khoảng 64 lần. Khi đó, với t = 256K và giới hạn 8 GB, chunk tối đa là khoảng 7.6K token chứ không phải khoảng 120. Chế độ bị giới hạn bởi bộ nhớ có thể yếu hơn nhiều so với mô phỏng. Đây chưa phải kết luận: phải đọc mã kernel indexer (G1c, Việc 1 của bước 2) để biết buffer thật được cấp phát theo shape nào.
