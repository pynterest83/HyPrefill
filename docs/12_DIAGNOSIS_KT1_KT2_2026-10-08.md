# Chẩn đoán KT1/KT2: vì sao chunk theo operator gần như không thắng Layered (08/10/2026)

Viết sau khi KT1 và KT2 đã chạy (máy hyprefill-dev-0, GPU 4, khoá 1980 MHz, Qwen3-Next-80B-A3B TP2). Tài liệu này **không đổi tiêu chí** ở PROPOSAL §5. Nó chỉ sửa một lỗi tính, đo thêm để giải thích kết quả, và nêu các hướng còn lại để quyết định.

## 0. Tóm tắt

1. **KT2-a tính sai ở các ô mà chunk đồng nhất không khả thi.** Sau khi sửa, kết quả là **1.18**, rơi vào **vùng giữa** chứ không phải KILL. Tuy vậy, cả hai ô ≥ 1.10 đều nằm ở vùng suy biến: phần budget còn cho prefill P = 0.94 ms, Layered chỉ đẩy được 39 token prefill mỗi iteration.
2. **Nguyên nhân gốc:** pipeline theo chiều sâu đã gỡ bỏ ràng buộc mà HyPrefill định giải. Chi phí mỗi token của attention gần như không đổi theo chunk. Vì vậy Layered cho attention dùng chunk lớn chỉ mất phần tam giác causal (4–8% của attention). Đó là trần ~1.04 trong vùng P thực tế.
3. **Vùng HyPrefill / Layered ≥ 1.25 chỉ có ở hai nơi.** Một là P ≲ 1–3 ms, khi decode chiếm gần hết GPU. Hai là khi mỗi lần chạy phải gom ≥ 12 layer (pipeline ≤ 4 tầng). Layered thực tế dùng `N_lg = ⌈L/512⌉` (16 nhóm, khoảng 3 layer mỗi nhóm cho append 8K), không thuộc vùng đó.
4. **Khoảng một nửa lợi ích khấu hao MoE của KT1 đến từ bảng cấu hình `fused_moe`, không từ việc dùng chung expert.** Bảng tune của vLLM cho H200 chọn `BLOCK_SIZE_M = 16` cho M ∈ 385–768 token, phù hợp router ngẫu nhiên. Routing thật tập trung token hơn. Với cấu hình tốt nhất cho từng ô, KT1 ở D = 32 là 0.63 (trên ngưỡng 0.6, vùng chưa quyết) và f = 10.5%, thay vì 0.45 và 19.4%.
5. **KT2-b vẫn GO** nhưng yếu hơn khi MoE được tune: pipeline / đồng nhất tối đa 1.51 (trước đó 1.875).

## 1. Số liệu và file

| Phép đo / phân tích | File | Ghi chú |
|---|---|---|
| KT1, 3 lần rút ghép cặp | `results/step02/2026-10-08_s02_*_kt1_draw{0,1,2}/`, `kt1_eval_*_2026-10-08.csv` | cấu hình `fused_moe` mặc định |
| KT2 bản đầu (lỗi NaN) | `kt2_capacity_*_kt1.csv` | giữ nguyên để đối chiếu |
| KT2 đã sửa lỗi | `kt2_capacity_*_kt1_fix.csv` | chỉ khác các ô từng là NaN |
| Quét `fused_experts` theo E_t × T | `2026-10-08_moe_diag/` (`bench/moe_diag.py`) | routing đặt tay |
| KT1 dưới mọi cấu hình tune | `2026-10-08_moe_config_cf/` (`bench/moe_config_cf.py`) | lượt mặc định tái hiện KT1 sai lệch ≤ 1.1% |
| KT1 / KT2 với cấu hình tốt nhất | `kt1_eval_*_bestcfg.csv`, `kt2_capacity_*_kt1_bestcfg.csv` | min theo cấu hình cho từng ô |
| Bản đồ (t, P), độ nhạy g và `h_fire` | `kt2_diag_*_2026-10-08/` (`bench/kt2_diag.py`) | chỉ CPU, MoE theo KT1 |

Mọi lượt GPU: `nr_throttled` = 0. Clock thấp nhất 1860–1920 MHz ở vài dòng, còn lại 1980 MHz.

## 2. Lỗi tính KT2-a

`bench/kt2_capacity.py` đặt `hy_over_best = NaN` khi chunk đồng nhất không khả thi (R = 0), dù mẫu số max(đồng nhất, Layered) = Layered > 0. Các ô B = 25 ms bị bỏ khỏi kết luận. Đã sửa: các cột khác không đổi, chỉ các ô từng là NaN có giá trị.

| h0 | KT2-a trước khi sửa | KT2-a sau khi sửa |
|---|---|---|
| 0 ms | 1.073 | 1.181 |
| 10 ms (quyết định) | 1.040 → KILL | **1.181 → vùng giữa** |
| 20 ms | 1.040 | 1.040 |

Các ô đạt ≥ 1.10 (h0 = 10 ms): D = 64, t = 64K, B = 25 ms, Δ ∈ {4K, 8K}, với P = 0.94 ms, Layered R = 39, HyPrefill R = 45–46 token mỗi iteration.

## 3. Phân rã chi phí mỗi token

µs mỗi token prefill trên toàn model (12 attention, 36 GDN, 48 MoE), D = 8, t = 64K (`kt2_diag_*/pertok.csv`):

| chunk n | Attention | GDN | MoE | Tổng | Một layer attention chạy một lần |
|---|---|---|---|---|---|
| 512 | 10.1 | 6.9 | 20.7 | 37.6 | 0.43 ms |
| 2048 | 10.1 | 4.4 | 9.0 | 23.5 | 1.72 ms |
| 8192 | 10.9 | 3.6 | 7.0 | 21.4 | 7.44 ms |

- Attention tỉ lệ với t, không với n. Từ 2048 lên 8192 chỉ đắt thêm phần tam giác causal (≈ n / 2t). Ở t = 256K, attention chiếm 79% tổng.
- GDN (chi phí cố định mỗi lần gọi) và MoE khấu hao mạnh, gần chạm đáy ở n ≈ 4096.
- Vì vậy chunk tối ưu của mọi operator đều là "lớn nhất có thể". Attention không muốn chunk nhỏ, nó chỉ chịu được chunk lớn với chi phí rất nhỏ. Lợi ích của việc tách chunk bị chặn bởi phần chênh của attention giữa chunk lớn và chunk nhỏ.

## 4. Bản đồ vùng HyPrefill thắng

HyPrefill / Layered theo P = B − D (h0 = 0, Δ = 8192, `phase.csv`):

| D, t | P = 1 | 2 | 4 | 8 | 16 | 32 ms |
|---|---|---|---|---|---|---|
| 32, 64K | 1.26 | 1.10 | 1.05 | 1.04 | 1.04 | 1.04 |
| 32, 128K | 1.48 | 1.19 | 1.08 | 1.04 | 1.03 | 1.03 |
| 32, 256K | 1.40 | 1.31 | 1.18 | 1.08 | 1.02 | 1.02 |
| Layered R (32, 128K), token / iteration | 23 | 57 | 125 | 259 | 521 | 1042 |

- **Ngưỡng 1.25:** cần P ≲ 1 ms ở 64K, ≲ 1.5 ms ở 128K, ≲ 2–3 ms ở 256K. Ở đó Layered chỉ được 30–55 token mỗi iteration, tức một append 8K mất 150–270 iteration.
- **Ở B = 50 ms:** P thực tế 15–40 ms, nên HyPrefill / Layered = 1.02–1.04.
- **Δ = 2048:** chunk nào cũng bị Δ chặn, nên tỉ lệ = 1.00 khi P ≥ 4 ms.

**Vì sao vùng này gần như rỗng.** HyPrefill có lợi khi Layered bị ép phải dùng chunk nhỏ, tức khi P nhỏ hơn chi phí một layer attention chạy một lần. Nhưng phần thưởng chỉ lớn khi phần GDN/MoE chưa khấu hao còn lớn so với attention, tức t không quá dài. Hai điều kiện đi ngược nhau: t càng dài thì ràng buộc càng dễ bị chặn, nhưng attention càng chiếm phần lớn chi phí và phần thưởng càng nhỏ. Chúng chỉ gặp nhau khi P rất nhỏ.

## 5. Độ nhạy theo cách cài

Lưới KT2, h0 = 10 ms (`group_fire.csv`). g là số decoder layer mỗi lần chạy, `h_fire` là CPU mỗi lần chạy một đoạn:

| g | max HyPrefill / tốt nhất | ô ≥ 1.10 | ô ≥ 1.25 |
|---|---|---|---|
| 1 | 1.18 | 2 | 0 |
| 2 | 1.14 | 2 | 0 |
| 4 | 1.07 | 0 | 0 |
| 8 | 1.18 | 8 | 0 |
| 12 | 1.36 | 10 | 3 |
| 24 | 1.35 | 13 | 5 |
| 48 | 2.31 | 28 | 18 |

- **g = 48:** cả độ sâu trong một lần chạy, tức không pipeline. Đây là bài toán gốc ở PROPOSAL §2.2 (r_d / r_u). Lợi ích lớn của HyPrefill **chỉ tồn tại khi không có pipeline theo chiều sâu**, và đó là lý do so với Sarathi thì thấy lợi, so với Layered thì không (§4.3).
- **g ≤ 8:** chưa ô nào đạt 1.25. Layered với `N_lg = ⌈L/512⌉` thuộc vùng này (g ≈ 3 cho L = 8K).
- **`h_fire`:** 0 → 0.2 ms không đổi kết quả ở g ∈ {1, 4}, vì trong các ô này GPU chặn trước CPU.

## 6. MoE: phần khấu hao đến từ đâu

### 6.1 Quét `fused_experts` với routing đặt tay (`moe_diag/`)

p50 ms, không gồm router và shared expert:

| E_t \ T | 64 | 256 | 512 | 768 | 1024 | 2048 | 8192 |
|---|---|---|---|---|---|---|---|
| 16 | 0.057 | 0.100 | 0.180 | **0.246** | 0.187 | 0.252 | 0.800 |
| 128 | 0.132 | 0.175 | 0.226 | **0.281** | 0.232 | 0.299 | 0.814 |
| 256 | 0.221 | 0.234 | 0.320 | **0.332** | 0.325 | 0.396 | 0.919 |
| 512 | 0.395 | 0.410 | 0.435 | 0.447 | 0.450 | 0.513 | 1.135 |

- **T nhỏ:** bị chặn bởi đọc weight. 512 expert (1.5 GB) mất khoảng 0.40 ms, tức khoảng 3.9 TB/s.
- **T ≥ 4096:** bị chặn bởi tính toán (khoảng 320 TFLOP/s ở E_t = 16, T = 8192).
- **Không đơn điệu:** T = 640–768 đắt hơn T = 1024 ở mọi E_t. Nguyên nhân là vLLM chọn cấu hình theo key M gần nhất trong `E=512,N=256,device_name=NVIDIA_H200.json`. Key 512 (cho M ∈ 385–768) dùng `BLOCK_SIZE_M = 16, BLOCK_SIZE_N = 256, num_stages = 2`; key 1024 dùng `BLOCK_SIZE_M = 32`. Bảng được tune với router ngẫu nhiên: 512 token × 10 / 512 expert ≈ 10 token mỗi expert. Routing thật tập trung hơn (D = 32, c = 512: khoảng 200 expert, ≈ 27 token mỗi expert), nên tile 16 hàng lãng phí.

### 6.2 KT1 dưới cấu hình tốt nhất (`moe_config_cf/`)

Cùng các ô và cùng lượt rút với KT1. Đo dưới cấu hình mặc định và dưới từng cấu hình phân biệt trong bảng tune (`override_config`):

| D | inc(512) mặc định → tốt nhất | tỉ lệ 2048 / 4×512 | f |
|---|---|---|---|
| 8 | 0.211 → 0.150 ms | 0.41 → 0.57 | 23.9 → 14.1% |
| 32 | 0.168 → 0.120 ms | 0.45 → **0.63** | 19.4 → **10.5%** |
| 64 | 0.149 → 0.094 ms | 0.48 → 0.74 | 16.8 → 6.1% |

- Theo tiêu chí, KT1 ở D = 32 thành **vùng chưa quyết** (tỉ lệ 0.63 > 0.6), dù f vẫn ≥ 10%.
- Cấu hình tốt nhất ở c = 512–2048 hầu hết là key M1536 (`BLOCK_SIZE_M = 64`).
- Lấy min trên khoảng 15 cấu hình có thiên lệch xuống nhỏ (nhiễu đo ≈ 1%), không đổi kết luận.

**Hệ quả.** Phần lớn lý do "MoE muốn chunk lớn" mà KT1 đo được là do bảng cấu hình chọn tile kém ở M trung bình, không phải do dùng chung expert. Nếu tune `fused_moe` theo routing thật, mọi policy dùng chunk nhỏ đều rẻ đi, và khoảng cách giữa chunk nhỏ và chunk lớn hẹp lại. KT2 với MoE đã tune: KT2-a = 1.14 (vùng giữa), KT2-b = 1.51 (GO; mặc định 1.875).

## 7. Giới hạn của chẩn đoán

- Mô hình KT2 là trạng thái ổn định, một request append mỗi lúc. Nó không có burst, TTFT, nhiều request prefill đồng thời, bộ nhớ activation giữa các tầng, hay tương tác thật giữa decode và prefill trong một lần gọi FA varlen. Mô hình gộp decode/prefill đã được kiểm ở c ≥ 2048 (`plan/01` §2.5), chưa kiểm ở c nhỏ.
- Chỉ có Qwen3-Next TP2. Kimi-Linear (KDA, MLA) và Qwen3.8-Flash-Next (sparse attention, indexer) có hình dạng chi phí attention khác. Với sparse attention, chi phí mỗi token gần như không phụ thuộc t, và ràng buộc của attention có thể là bộ nhớ workspace của indexer chứ không phải thời gian (G1c). Đây là chỗ duy nhất luận điểm gốc có thể đứng theo một dạng khác, nhưng G1c đang nghiêng FAIL (vLLM PR #57105).
- `h_fire` và g là tham số giả định, chưa đo trên bản cài thật.

## 8. Hướng tiếp theo (để quyết định, chưa làm)

1. **Theo tiêu chí, KT2-a là vùng giữa (1.18, hoặc 1.14 khi MoE tune).** Vùng đó chỉ có ở P ≲ 3 ms. Đề xuất coi headline "HyPrefill / Layered" là không đứng được trong vùng có ý nghĩa vận hành, và chuyển sang KT2-b, kèm báo cáo trung thực về vùng P nhỏ. Quyết định này cần ghi vào PROPOSAL §5 kèm lý do.
2. **KT2-b (GO, 1.51–1.875):** pipeline theo nhóm layer cho model hybrid trong vLLM, nhận biết CPU. Đây là Layered Prefill đưa vào vLLM cho GDN hybrid. Độ mới phải đến từ phần riêng của hybrid (state GDN giữa các tầng, chi phí cố định mỗi lần gọi GDN, CPU) và từ bản cài thật đạt gain đo được.
3. **Phát hiện phụ có giá trị riêng:** tune `fused_moe` theo routing thật. Đo được inc(512) giảm 29–37%; cả lần gọi `moe_mixed` ở c = 512 rẻ hơn 17–20%, ở c = 256 và 1024 rẻ hơn 5–9%, còn ở c ≥ 2048 gần như không đổi. Việc này độc lập với HyPrefill, áp dụng cho mọi policy và mọi baseline, nên phải làm trước khi so sánh end-to-end (`docs/03_MEASUREMENT.md` §7: tune baseline tốt nhất). Có thể thành một PR upstream hoặc một mục trong phương án lui đo đạc (`docs/09` §5).
4. **Kiểm độ bền trên Flash-Next / Kimi-Linear** chỉ đáng làm nếu muốn giữ luận điểm gốc ở dạng ràng buộc bộ nhớ của attention (G1c).
