# Design: chạy prefill theo tầng trong vLLM 0.30 (Layered k = 1, HyPrefill k ≥ 2)

Viết 10/10/2026, trước khi code (`plan/05`). Dựa trên đọc mã vLLM v0.30.0 (`third_party/vllm`); đường dẫn bên dưới tương đối với thư mục đó. Model mục tiêu: Qwen3-Next-80B-A3B TP2 (48 layer: 3 GDN + 1 attention lặp 12 lần, MoE sau mỗi mixer).

## 1. Mục tiêu và phạm vi

Một request prefill được phép đi qua các layer ở những iteration khác nhau, và mỗi nhóm operator nhận số token riêng mỗi lần gọi:
- **Chunked** (baseline): mọi token mới đi hết 48 layer trong một iteration (vLLM hiện tại).
- **Layered (k = 1):** cùng một chunk n cho mọi operator, nhưng chunk dừng sau một số layer và đi tiếp ở iteration sau.
- **HyPrefill bản tĩnh (k ≥ 2):** attention nhận chunk c, GDN và MoE nhận chunk k·c; activation chờ giữa các iteration.

Không đổi kernel nào (GDN vẫn `FLA_CHUNK_SIZE` = 64). Không làm trong prototype đầu: prefix cache, spec decode, cascade attention, KV connector, SP-MoE, nhiều request prefill cùng lúc ở các tầng khác nhau (một request prefill đang chạy dở mỗi lúc, phần còn lại xếp hàng).

## 2. Đơn vị lập lịch: tầng (stage)

Chia chuỗi sublayer `[G M G M G M A M] × 12` thành **tầng**, mỗi tầng là một đoạn liên tiếp mà mọi sublayer nhận cùng một tập token:
- Layered: tầng = nhóm `48 / N_lg` layer liên tiếp.
- HyPrefill: ranh giới tầng đặt quanh mỗi attention: `[G M G M G M]` (k·c token), `[A]` gồm norm, qkv, attention, o_proj (c token), `[M G M G M G M]`… Các op "dense" của attention đi theo chunk c, của GDN theo k·c.

Mỗi iteration: **token decode đi qua mọi tầng** (như bây giờ), còn token prefill chỉ có mặt ở các tầng được lập lịch cho iteration đó. Tập token của một lần gọi tầng s = decode ∪ prefill đang ở tầng s. Như vậy decode và prefill vẫn đi chung một lần gọi GEMM/MoE (giữ phần dùng chung weight mà KT1 đo được).

Lịch tầng (tầng nào chạy chunk nào ở iteration nào) do một bộ chọn tham lam theo budget, như `bench/kt2_sim_check.py`: duyệt tầng từ sâu đến nông, chạy tầng nếu đủ token chờ và chi phí dự đoán (bảng cost) còn vừa budget B − D.

## 3. Thay đổi theo thành phần

**Scheduler** (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/request.py`):
- Thêm vào `Request` bản đồ tiến độ theo tầng `stage_done[s]` (số token đã qua tầng s). `num_computed_tokens` chỉ tăng khi token qua **tầng cuối**, vì nó điều khiển vị trí, việc lấy mẫu và commit block (`kv_cache_manager.py:595`).
- Cấp block KV cho cả chunk ngay khi chunk vào tầng đầu, không giải phóng đến khi qua tầng cuối. Preempt (`scheduler.py:1497`) phải xoá cả bản đồ tầng và buffer.
- Request được giữ cờ `is_prefill_chunk` đến tầng cuối (cần cho async scheduling, `async_scheduler.py:28`).
- `SchedulerOutput` thêm trường `stage_plan: dict[req_id, list[(stage, token_lo, token_hi)]]`.

**Model runner:** chọn runner V1 (`gpu_model_runner.py`, ép bằng `VLLM_USE_V2_MODEL_RUNNER=0`) cho prototype vì đường đi đã được đọc kỹ. Chunked baseline đo trên **cả V1 và V2** để chắc không chọn baseline yếu hơn mặc định. Mỗi iteration, thay một lần `self.model(...)` bằng vòng lặp qua tầng:
- dựng tập token của tầng (gather từ input decode và từ buffer activation),
- dựng metadata riêng: attention có `query_start_loc`, `seq_lens`, `slot_mapping` theo vị trí token của tầng; GDN có `has_initial_state`, `prefill_state_indices`, `seq_lens` theo tiến độ của **tầng đó** (`gdn_attn.py:209,400`). Dict `attn_metadata[layer_name]` cho phép mỗi layer một metadata khác nhau,
- mở `set_forward_context` riêng cho lần gọi tầng (MoE dùng `forward_context.num_tokens`),
- scatter output về buffer (hidden, residual) và về decode.
Logits chỉ tính cho token decode và token prefill vừa qua tầng cuối.

**Model** (`vllm/model_executor/models/qwen3_next.py`): `Qwen3NextModel.forward` đã có vòng lặp layer và nhận `IntermediateTensors(hidden_states, residual)` (`:688–728`), nhưng khoảng layer là hằng lúc dựng và bị torch.compile trải phẳng. Thêm một module con có thể compile cho từng loại tầng, chạy một đoạn sublayer với tập token cho trước, dùng lại hợp đồng dữ liệu (hidden, residual) của pipeline parallel. Lớp decoder hiện gộp mixer và MoE trong một `forward` (`:548`); cần tách thành hai bước gọi được riêng để ranh giới tầng rơi giữa attention và MoE.

**State GDN:** mỗi layer có conv state và ssm state riêng theo slot của request (`qwen_gdn_linear_attn.py:1318`); `causal_conv1d_fn` và `chunk_gated_delta_rule` đọc state cũ, ghi state mới cho đúng layer đó. Vì vậy chạy so le giữa các layer là đúng, với điều kiện: (1) `mamba_cache_mode = 'none'` (chế độ 'align' copy state cho mọi layer cùng lúc, `gpu_model_runner.py:4340`), (2) slot không đổi trong suốt prefill.

**KV attention:** KV của layer j được ghi trong forward của layer j, slot tính từ (block table, vị trí) (`gpu_model_runner.py:2204`). Chạy layer j cho vị trí [p, p + c) ở iteration sau là đúng nếu block đã cấp và metadata của lần gọi đó có `seq_lens = p + c`.

## 4. Buffer activation

Mỗi token chờ giữa hai tầng giữ (hidden, residual) bf16: 2 × 2048 × 2 B = 8 KiB mỗi GPU. Với chunk lớn nhất k·c = 8192 và tối đa hai ranh giới có token chờ cùng lúc: khoảng 128 MiB mỗi GPU, không đáng kể so với KV (≈ 60 GiB). Buffer cấp trước, tính vào lượt profile bộ nhớ của vLLM.

## 5. CUDA graph và CPU: rủi ro lớn nhất

Đã đo (`results/step01/2026-10-09_cudagraph_check/`): step prefill chạy eager tốn ~85–90 ms CPU (khoảng 1860 lần phát kernel); với piecewise graph, c = 512 chỉ còn 25 ms/step. Graph piecewise được capture theo đúng một số token mỗi lần forward, và tách tại `unified_attention_with_output` và `qwen_gdn_attention_core` (`compilation.py:772`). Tầng chạy eager sẽ xoá mọi lợi ích.

Cách làm: mỗi loại tầng là một module compile riêng, có graph piecewise capture cho tập kích thước `D + {0, c, k·c}` với D theo bậc thang kích thước decode của vLLM. Số graph tăng theo (số loại tầng × số kích thước); kiểm bộ nhớ graph pool. Nếu không capture được ở M2, báo cáo HyPrefill eager như một kết quả âm có điều kiện, không so với chunked có graph.

## 6. M1: bộ chạy thử bằng kernel thật (trước khi sửa vLLM)

`bench/hyprefill_emulator.py`, một GPU, shape TP2 mỗi GPU, weight ngẫu nhiên, all-reduce cộng theo bảng (một GPU không có all-reduce thật):
- Sublayer dựng từ chính các hàm trong `bench/op_cost.py`: FA3 paged (KV thật cho t), GDN FlashInfer có `initial_state`, `moe_mixed` với routing thật, op dense.
- Ba lịch (chunked, Layered, HyPrefill) do cùng bộ chọn tham lam theo bảng cost quyết, rồi **chạy thật** từng iteration: decode D token qua 48 layer cộng phần prefill theo lịch.
- Đo mỗi iteration: thời gian GPU (CUDA event quanh iteration) và wall (CPU phát lệnh), chạy hai chế độ: eager, và CUDA graph theo tầng (capture từng loại tầng theo kích thước).
- Kết quả: R thật dưới budget B, tỉ lệ iteration vượt B, so với dự đoán của mô phỏng. Quyết theo tiêu chí M1 ở PROPOSAL §5.

## 7. Kiểm đúng

- Token-level: greedy trên 20 prompt, Layered và HyPrefill phải sinh đúng token như chunked (`plan/05`).
- Bất biến: sau tầng cuối, state GDN mỗi layer và KV mỗi layer của request giống hệt chạy chunked (so tensor trên vài request ngắn).
- Hiệu năng: chunked trong bản fork phải bằng wheel trên §2.5 (±3%) trước khi so.

## 8. Thứ tự làm

M1 (bộ chạy thử) → fork vLLM build từ source, chạy chunked bằng wheel → tách decoder layer thành mixer/MoE, module tầng compile được → scheduler + runner cho Layered (k = 1), kiểm đúng → HyPrefill bản tĩnh, kiểm đúng → graph theo tầng → đo end-to-end (bước 8).
