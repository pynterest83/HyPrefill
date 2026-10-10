# Nhật ký thí nghiệm HyPrefill

Mỗi ngày 5 dòng. Viết trong ngày, không viết bù cuối tuần.
Mẫu: **Làm gì** · **Số liệu chính** · **Bất ngờ** · **Bị chặn** · **Mai làm gì**

---

## Nhật ký

### 2026-09-28
- Làm gì: dựng env `hyprefill` (vllm 0.30.0, torch 2.13.0+cu130), tải xong 3 model bước 0, viết `bench/op_cost.py` + `bench/analyze_step01.py`, chạy lượt đo bước 1 (FA3 + GDN FlashInfer/triton) cho Qwen3-Next TP2 và Qwen3.8-27B TP1 trên GPU 4. **Clock chưa khoá** → số tham khảo, không đưa vào paper.
- Số liệu chính: GDN không phụ thuộc t (xu hướng t≥128K so với t≤16K ≤ 5%, một điểm lạc c=4096 t=16K). GDN FlashInfer Qwen3-Next TP2: cost = 53.5 µs + 1.05 µs·⌈c/64⌉ (chi phí cố định trội tới c ≈ 3.2K). FA3 Qwen3-Next TP2 ở t=256K: ~3.0 µs/token/layer, gần phẳng theo c ≥ 256.
- Bất ngờ: (1) đo eager một lần gọi bị giới hạn bởi CPU (wrapper GDN ~0.2 ms CPU so với ~0.04 ms GPU) và tạo ra phụ thuộc t giả, phải chuyển sang đo replay CUDA graph. (2) Triton FLA rẻ hơn FlashInfer ở c nhỏ (a = 32 µs so với 53 µs) nhưng đắt hơn 3× mỗi chunk; điểm cắt ~640 token. (3) FA3 có bậc thang theo tile 128: c=129 đắt hơn c=128 rõ (Qwen3.8-27B t=256K: 15.8 so với 9.7 µs/token).
- Bị chặn: root trong container không khoá được clock GPU (cần admin ở host). GPU dùng chung, lúc có lúc không.
- Mai: nhờ admin khoá clock GPU 4–7; đo phần Dense (projection, norm, gate); mục 2.4 (tách intra/inter) và 2.5 (so tổng cost với forward thật).

### 2026-09-29
- Làm gì: đo lại bước 1 bằng cách đo mới (graph K lần + xoá L2), thêm Dense, §2.4, §2.5 (vLLM thật + profile); demo Layered quét tải 1.3–3.5 req/s; bước 2: MoE, decode, oracle đầu tiên; G1c đọc mã indexer; chốt benchmark (PROPOSAL §4.4).
- Số liệu chính: bảng cost lặp lại median 0.2–0.9%; GDN lệch theo t ≤ 0.6%; §2.5 phần GPU +2%, cả iteration −6..−10% ở c ≥ 2048 nhưng −22..−35% ở c = 512; demo: chunked và layered cùng sụp ở 3.0 req/s (throughput 2.67 / 2.61); oracle Qwen3-Next TP2 ở SLO 50 ms: r_d / r_u = 1.27–2.0 (t ≤ 64K).
- Bất ngờ: vLLM prefill chunk nhỏ bị giới hạn bởi CPU; mô hình số phép tính intra/inter không khớp; indexer QSA trong vLLM 0.30 tự chia query để buffer logits ≤ 512 MB (G1c có thể trượt); Layered không thắng trên H200 với cấu hình của fork.
- Bị chặn: chưa đo được G1c (đang tải Flash-Next, mạng lỗi); cần sim có pipeline + goodput mới quyết G1a/G1b.
- Mai: dừng lại tổng hợp, bàn hướng với advisor trước khi chạy tiếp.

### 2026-09-29 (chiều, chuẩn đo 1980 MHz)
- Làm gì: dọn dữ liệu cũ; đo lại bước 1 trên GPU 6, 7 (mỗi card một lượt), all-reduce TP2 trên GPU 4, 5; bắt đầu kiểm chứng Layered trên GPU 4, 5.
- Số liệu chính: lặp lại giữa hai card median 0.17–0.19%, 100% dòng không chạm trần < 3% (mục bước 0 đạt); GDN không phụ thuộc t (≤ 0.3%); §2.5 −4…−9% ở c ≥ 1024, −15…−32% ở c = 512; all-reduce TP2 39 µs ở 2048 token.
- Bất ngờ: ở c = 512, t = 16K GPU chờ CPU 44% mỗi step; phần GPU của engine dài hơn bảng 2–10%; GDN triton lệch giữa hai card (nghi autotune).
- Bị chặn: các dòng t = 0 của §2.5 đo sai cách (trừ nhầm phần đọc weight); bản đầu của `--sweep` sai, đã sửa và chạy lại.
- Mai: sửa §2.5 t = 0, mô hình hoá phần CPU; bảng bước 2 (MoE, decode) để chạy §2.5 cho Qwen3-Next và §2.5 có decode song song.

### 2026-10-06 (quét lại độ mới + tiền đề, không đo)
- Làm gì: ba checker độc lập quét paper (khoảng 60 truy vấn arXiv, 18 OpenReview ICLR'27), upstream (vLLM/SGLang/Dynamo 01/09–06/10) và xu hướng kiến trúc; đối chiếu tiền đề với `results/step01–02`. Tổng hợp ở `docs/09_RESCAN_2026-10-06.md`; cập nhật PROPOSAL §1.3, §2.4, §2.5.3, §3, §5, §6.
- Số liệu chính: novelty còn giữ (không có kết quả HIGH; gần nhất là CascadeEP 2609.33252, MEDIUM-LOW). Routing thật Qwen3-Next: decode D=32 chạm 151/512 expert; hợp với c=512 là 289, với c=2048 là 340; prefill c=256 đứng riêng chạm 207 (không phải ~505). GDN gom k chunk tiết kiệm ≤ 3.4% budget. Điểm ước lượng trước khi đo giảm từ 7.5–8 xuống khoảng 5.
- Bất ngờ: oracle 1.27–2.0 của 29/09 chỉ tái hiện được với routing ngẫu nhiên. vLLM PR #57105 (merge 27/09) đặt trước workspace logits indexer → G1c nghiêng FAIL. SGLang PR #42411 (PDMux, 03/10) đưa layerwise prefill + decode overlap cho GLM-5.3-Flash vào upstream.
- Bị chặn: chưa có số đo thời gian `moe_mixed` với routing thật; ước lượng 6–7% tiết kiệm MoE còn là suy luận.
- Mai: chạy KT1 (`moe_mixed --routing`, D ∈ {32, 64}, c ∈ {256, 512, 2048}) rồi KT2 (simulator với routing thật); đo peak memory indexer cho G1c. Tiêu chí GO/KILL đã viết ở PROPOSAL §5 trước khi chạy.

### 2026-09-30 → 2026-10-07 (kiểm chứng Layered trên H200)
- Làm gì: demo Layered 9 mức tải ở 1980 MHz (mặc định của tác giả); năng lượng mỗi token; dump routing MoE thật (Qwen3-Next, arXiv); profile nsys của fork, chunked và layered (`bench/step00_layered_profile.sh`).
- Số liệu chính: goodput chunked ~2.7, layered ~2.6 req/s; layered ít năng lượng hơn 9.9–12.7%; trong cùng cửa sổ 60 s, kernel MoE −25%, attention −22%, tổng GPU −11%, nhưng `prepare` ×3.4 và `sample` ×2.4 (tổng), GPU bận 75% → 67%.
- Bất ngờ: cơ chế của Layered có tác dụng trên H200 nhưng fork bị giới hạn bởi CPU nên không thành goodput; routing thật tập trung hơn router ngẫu nhiên (decode 32 → 151 expert, chunk 8192 → 373).
- Bị chặn: chưa kiểm riêng giả thuyết "CPU ăn mất phần GPU tiết kiệm" (cần giảm phần CPU của layered hoặc chạy trên H100).
- Tiếp: bước 1 còn mở (mô hình phần CPU, §2.5 cho Qwen3-Next TP2), bước 2 (`moe_mixed` với routing thật, G1c trên model thật).

### 2026-10-07 (phản biện bản quét lại, KT2 sớm)
- Làm gì: đối chiếu `docs/09` với số đo (`docs/10_REBUTTAL_2026-10-07.md`); sửa tiêu chí KT1/KT2 trước khi chạy (PROPOSAL §5); viết `bench/kt2_capacity.py` (dung lượng trạng thái ổn định, ba policy, all-reduce, CPU); thêm `bench/cgroup_cpu.py` vào mọi script đo.
- Số liệu chính: KT2 với MoE ước lượng, h0 = 10 ms: HyPrefill / max(đồng nhất, Layered) tối đa 1.04 (KILL KT2-a), pipeline / đồng nhất tối đa 1.71 ở t ≥ 64K (GO KT2-b).
- Bất ngờ: bản sim cũ cho phép một sublayer chạy nhiều lần mỗi iteration và tính MoE decode tách riêng; bản đầu của kt2 tính "một lần chạy" là cả 12 layer attention (ép Layered dùng chunk nhỏ), đã sửa trước khi chạy cả lưới.
- Bị chặn: GPU 4–7 bị server vLLM TP4 của dự án khác chiếm (cùng tài khoản); dump routing 192 request hỏng ở `init_device`; clock từng mất khoá sau khi pod được cấp lại tài nguyên.
- Tiếp: KT1 (`moe_mixed --routing`, 3 lần lấy mẫu, D ∈ {8, 32, 64}) và dump 192 request khi GPU trống và khoá 1980 MHz; chạy lại KT2 với `--kt1`.

### 2026-10-08 (chuyển sang hyprefill-dev-0, KT1, KT2 với MoE đo thật)
- Làm gì: dựng máy mới (`AGENTS.md`, `results/env/2026-10-08_hyprefill-dev-0/`), clock khoá 1980 MHz, GPU 4–7 trống; dump lại routing 192 request arXiv (Qwen3-Next TP2); KT1 `moe_mixed --routing`, 3 lần rút ghép cặp theo D (`bench/run_kt1.sh`, `bench/kt1_eval.py`); chạy lại KT2 với `--kt1`.
- Số liệu chính: KT1 **GO**: inc(D, 2048) / 4·inc(D, 512) = 0.41 / 0.45 / 0.48 và f = 23.9 / 19.6 / 16.9% ở D = 8 / 32 / 64 (3 lần rút: 0.40–0.52). KT2 (h0 = 10 ms): KT2-a HyPrefill / max(đồng nhất, Layered) tối đa 1.04 → **KILL**; KT2-b pipeline / đồng nhất tối đa 1.875 → **GO** (h0 = 0: 1.07 / 2.29).
- Bất ngờ: tiết kiệm MoE khi gom chunk lớn hơn nhiều so với ước lượng 06/10 (6–7%), nhưng chunk 2048 chỉ chạm thêm ~20% expert so với 512 → phần lớn tiết kiệm là hiệu suất kernel `fused_moe` khi mỗi expert có nhiều token, không chỉ phần đọc weight (suy luận, chưa đo tách). Layered k = 1 với chunk lớn lấy gần hết phần đó.
- Bị chặn: không. Lỗi chia cho 0 ở c = 0 của `op_cost.py` đã sửa; thư mục của lượt hỏng đã xoá.
- Tiếp: phân tích vì sao tách chunk theo operator chỉ thêm ≤ 4% (ràng buộc nào chặn Layered), rồi quyết hướng theo PROPOSAL §5 (KT2-b).

### 2026-10-08 (chiều, chẩn đoán KT1/KT2: `docs/12_DIAGNOSIS_KT1_KT2_2026-10-08.md`)
- Làm gì: sửa lỗi NaN của KT2-a (ô chunk đồng nhất không khả thi bị bỏ); quét `fused_experts` theo số expert × số token (`bench/moe_diag.py`); đo KT1 dưới mọi cấu hình tune của `fused_moe` (`bench/moe_config_cf.py`); bản đồ HyPrefill / Layered theo (t, P), độ nhạy theo số layer mỗi lần chạy g và CPU mỗi lần chạy (`bench/kt2_diag.py`).
- Số liệu chính: KT2-a đã sửa = 1.18 → **vùng giữa** (không phải KILL), nhưng chỉ ở P = 0.94 ms, Layered 39 token/iteration; ở P thực tế 15–40 ms HyPrefill / Layered = 1.02–1.04. ≥ 1.25 chỉ khi P ≲ 1–3 ms hoặc g ≥ 12 layer mỗi lần chạy. KT1 với cấu hình MoE tốt nhất: tỉ lệ 0.63, f = 10.5% ở D = 32 → vùng chưa quyết; KT2 khi đó: KT2-a 1.14, KT2-b 1.51 (GO).
- Bất ngờ: một nửa lợi ích khấu hao MoE của KT1 là do bảng cấu hình vLLM cho H200 (key M = 512: `BLOCK_SIZE_M = 16`, tune với router ngẫu nhiên); `moe_mixed` ở c = 512 rẻ hơn 17–20% với cấu hình khác, và M = 640–768 đắt hơn M = 1024.
- Bị chặn: không. Quyết định hướng (bỏ headline HyPrefill / Layered, đi KT2-b) chờ duyệt, chưa sửa PROPOSAL.
- Tiếp: ghi quyết định vào PROPOSAL §5; tune `fused_moe` theo routing thật cho mọi policy trước khi so end-to-end; nếu đi KT2-b thì thiết kế bản cài pipeline theo nhóm layer trong vLLM.

### 2026-10-09 (đo lại các phép đo từng bị nghẽn CPU, `bench/run_cpu_recheck.sh`)
- Làm gì: governor CPU đã đổi: NUMA 2–3 (CPU 48–95, cạnh GPU 4–7) là `performance`, NUMA 0–1 là `powersave`; quota cgroup vẫn 128 core, không bị bóp. Đo lại `host_ms` (Qwen3-Next TP2, GPU 6), §2.5 Qwen3.8-27B TP1 (như 29/09), §2.5 Qwen3-Next TP2 (lần đầu), profile nsys fork Layered (như 30/09); chạy lại KT2 với `host_ms` mới; bắt đầu demo Layered quét tải 1.3–4.0 req/s.
- Số liệu chính: thời gian GPU khớp máy cũ ±5% (clock tụt vì SW power cap như cũ). `host_ms`: FA −32%, fa_decode −27%, gdn_decode −24%; GDN ổn định 0.272–0.278 ms (máy cũ dao động 0.22–0.38). §2.5 Qwen3.8-27B c = 512: CPU/idle mỗi step 34 → 9 ms, GPU bận 56 → 83%. Fork ở 2.5 req/s (nsys): máy cũ bão hoà (2.28 req/s, TTFT 7.8 s), máy mới theo kịp (2.44 req/s, TTFT 1.35 s); Layered vẫn ít GPU hơn 11% nhưng TTFT không tốt hơn. KT2 với `host_ms` mới: không đổi (1.181 / 1.875).
- Bất ngờ: §2.5 của Qwen3-Next TP2 không đo được bằng cách cũ: lần chạy có profiler chậm hơn lần đo wall tới 25%, GPU bận ra 110–123% wall. Đã chuyển ra `~/hyprefill_data/invalid_runs/`; `profile_vllm_step.py` giờ tính GPU bận bằng hợp khoảng (không cộng trùng stream) và cảnh báo khi bận > wall.
- Bị chặn: không biết governor của CPU 48–95 trước 09/10 (ảnh chụp 08/10 không ghi); `record_env.sh` giờ ghi governor từng NUMA node.
- Tiếp: §2.5 cho Qwen3-Next TP2 cần cách đo khác (nsys trên chính request được tính wall).
- Demo quét tải (xong 07:27, `results/step00/layered_demo/2026-10-09/`): chunked bão hoà ~3.0 req/s (máy cũ ~2.64). **Layered giờ thắng gần bão hoà**: TTFT layered/chunked 0.97 (2.5), 0.86 (2.8), 0.82 (2.9), 0.65 (3.0), 0.42 (3.2 req/s); SLO đạt ở 3.2 req/s 82% so với 33%; throughput bão hoà 3.04–3.11 so với 3.00; năng lượng −10…−14%. Máy cũ: Layered tệ hơn chunked tới 1.9× ở 2.8 req/s. Vậy giả thuyết 30/09 đúng: CPU cũ ăn mất phần GPU Layered tiết kiệm. Lưới tải còn thô (cả hai goodput ≥ 3.0 ở ngưỡng 90%); cần quét mịn 3.0–3.3.
- G1c (chiều): indexer QSA cấp phát tỉ lệ c·t nhưng bị chặn ở ~1 GiB (giới hạn 512 MB, tốn ≤ 6% thời gian; bỏ giới hạn thì crash ở c = 32768, t = 128K); model thật Flash-Next-FP8 TP2: KV cache −1.6% ở chunk 8192, −15% ở 32768, do activation chung chứ không riêng attention → **G1c trượt**, cửa cuối của luận điểm gốc đóng. Expert decode D = 128 chạm 280/512. `plan/02` đã điền checklist và KẾT QUẢ.

### 2026-10-09 (tối, kiểm lại "HyPrefill không thắng": `docs/13_AUDIT_KT2_2026-10-09.md`)
- Làm gì: §2.5 lần đầu cho Qwen3-Next TP2 (nsys, `bench/validate_forward_moe.py`); kiểm CUDA graph; mô phỏng từng iteration thay công thức KT2 (`bench/kt2_sim_check.py`), có budget CPU, hai thứ tự xếp lịch, MoE cấu hình tốt nhất; quét mịn demo 3.05–3.3 req/s.
- Số liệu chính: bảng cost khớp forward thật −3 … −7% ở các ô GPU là ràng buộc. Mô phỏng: HyPrefill / Layered tới 1.31 (1.36 với MoE tốt nhất), 5/46 ô ≥ 1.25, 9–10 ô ≥ 1.10, ở B = 25 ms (vài ô 50 ms), t ≥ 128K; B = 100 ms: ≤ 1.06. Công thức KT2 trên cùng lưới: 0 ô ≥ 1.25. Demo fork: goodput layered / chunked ~1.03.
- Bất ngờ: công thức trạng thái ổn định lạc quan cho Layered (xếp việc vào budget như chất lỏng); chunk lớn bị "cục", Layered đạt 59–93% throughput công thức hứa. Ngược lại, KT2 tính attention ở context cố định, có lợi cho HyPrefill (đã sửa, ≤ 3%). vLLM 0.30 chạy eager mọi step lớn hơn cỡ CUDA graph lớn nhất (mặc định 512): ~90 ms CPU/step.
- Bị chặn: lượt kiểm forward đầu đặt `max_num_seqs = 16` nên mọi step eager (đã sửa script, chưa chạy lại); §2.5 Qwen3-Next có prefix cache vẫn 91 ms/step ở c = 512, chưa rõ vì sao.
- Tiếp: quyết có dùng mô phỏng từng iteration để quyết KT2-a không (ghi PROPOSAL §5); nếu có, luận điểm mới là độ mịn khi xếp việc vào budget ở TBT chặt, context dài, và kiểm thật ở bước 5.

### 2026-10-10 (đổi trình tự plan: cài HyPrefill sớm)
- Làm gì: rà lại tiến độ so với plan: đã lệch sang nhánh kill test KT1/KT2 (mô hình) thay vì đi 01 → 02 → 03; đề xuất bỏ headline ở `docs/12` trái với `plan/02` §3 (G1 không quyết bằng mô hình) → rút lại. Quyết định cùng người dùng: cài HyPrefill sớm (bước 5), quyết G1a/G1b bằng số end-to-end; thêm M1 (bộ chạy thử bằng kernel thật) và chế độ HyPrefill bản tĩnh vào bước 5; simulator bước 3–4 thành công cụ hỗ trợ. Sửa PROPOSAL §5, `plan/02`, `plan/03`, `plan/04`, `plan/05`, `plan/07`.
- Số liệu chính: không đo.
- Bất ngờ: hai mô hình trên cùng số đầu vào cho kết luận ngược nhau (1.04 so với 1.31–1.36), nên không mô hình nào đủ để quyết.
- Bị chặn: tiêu chí M1 (HyPrefill / Layered ≥ 1.15 ở ô có Layered R ≥ 100 token/iteration → đi tiếp; < 1.05 mọi ô → dừng nhánh) đang chờ duyệt, phải chốt trước khi chạy M1.
- Tiếp: đọc mã vLLM 0.30 (scheduler, model runner, Qwen3-Next, state GDN), viết `docs/06_DESIGN.md`, rồi M1.

### 2026-10-10 (chiều, M1)
- Làm gì: viết `docs/06_DESIGN.md`; chốt tiêu chí M1 chặt hơn (≥ 1.30 ở 1 ô và ≥ 1.15 ở 3 ô, so với baseline tốt nhất từng ô) trước khi viết code; viết `bench/hyprefill_emulator.py` (stack 48 layer Qwen3-Next từ kernel vLLM, weight ngẫu nhiên, routing thật, chi phí lập lịch hiệu chỉnh từng sublayer trên chính stack, CUDA graph mỗi kiểu iteration) và `bench/m1_eval.py`; chạy lưới M1 trên GPU 4–7.
- Số liệu chính: **M1 ở giữa**: tối đa 1.21 (B = 25 ms, D = 8, t = 256K, Δ = 8192), 2 ô ≥ 1.15, 6 ô ≥ 1.10, 0 ô ≥ 1.30; B = 100 ms ≤ 1.06. Mô phỏng từng iteration từng dự đoán 1.25–1.36 ở các ô này.
- Bất ngờ: kết luận nhạy với biên budget: lượt đầu (an toàn 1.02) và lượt v2 (1.04) đều cho "≥ 1.30" ở ô 256K chỉ vì cấu hình tốt nhất của Layered vượt B 0.2–0.6% và bị loại; tune biên công bằng thì còn 1.15–1.21. Bảng cost bước 1/2 lệch từng loại sublayer nhiều hơn tổng (MoE +37…+120% so với KT1, attention −7…−15%) vì routing khác theo layer.
- Bị chặn: không về kỹ thuật. Theo tiêu chí, đi tiếp M2 nhưng khó đạt A*; cần bàn với advisor/người dùng trước khi đầu tư vào bản cài vLLM.
- Tiếp: quyết đi M2 hay không; nếu đi, nhắm vùng TBT chặt (25–50 ms), context ≥ 128K, append lớn (Δ ≥ 8192), nơi HyPrefill thắng 1.10–1.21.

---

## Ý tưởng để dành (không code trong lúc chạy kế hoạch)

-

## Quyết định lớn và lý do

| Ngày | Quyết định | Lý do | Dữ liệu đỡ lưng |
|---|---|---|---|
| 2026-09-28 | Đo bằng kernel vLLM 0.30 (FA3, GDN FlashInfer), không dùng flash-attn/fla riêng | Bảng cost phải cùng kernel với engine sẽ đánh giá; flash-attn 2.8.3 không build được với torch mới | `AGENTS.md` mục Server |
| 2026-09-29 | Trace workload chính: `semianalysisai/cc-traces-weka-062126-256k`. SLO: TBT 50 ms headline + 25 / 100 ms + mức 5 × bước decode; TTFT = 5 × TTFT không tải P90 (con số chốt trước khi so policy); độ nhạy × 0.5 / 1 / 2 | Trace agentic thật, có hash prefix, đúng workload append-prefill; quy tắc SLO theo cách Layered, SLOWeave, DistServe đặt | PROPOSAL §4.2, §4.4; `plan/04` |
| 2026-09-29 | Mọi policy trong cùng vLLM 0.30; Layered = cấu hình k = 1 của hạ tầng chạy theo nhóm layer dùng chung với HyPrefill; fork nanovllm chỉ để kiểm chứng bản Layered, bảng tham khảo khác engine, và phương án lui ở cổng cứng | Công bằng (cùng engine, cùng đường code) và áp dụng được vào vLLM; tác giả Layered cũng chỉ so khác engine với vLLM như tham khảo | `plan/05`, `08`, `09`; PROPOSAL §4.3–4.4, §5 |
| 2026-09-29 | Bật prefix cache cho workload append-prefill (bản cũ: tắt trong eval chính) | Workload chính dựa vào context cũ đã nằm trong cache; khi bật, vLLM cắt chunk theo block Mamba (`mamba_cache_mode = "align"`), là ràng buộc thật cho chunk size | `plan/05` §2.5, `plan/09` §3 |
| 2026-09-29 | Rà lại kế hoạch sau lượt 1830: chuẩn đo mới (1980 MHz, KV paged, state GDN theo config); tiêu chí lặp lại theo median/95%; thêm all-reduce, MoE batch trộn, phần CPU của iteration; G1a/G1b quyết ở bước 3 bằng mô phỏng đã khớp hệ thật; bỏ công thức intra/inter khỏi đường găng. Các phát hiện bất lợi (G1c, MoE dùng chung với decode, Layered trên H200) **giữ là giả thuyết**, đo lại theo chuẩn mới rồi mới quyết | Lượt 1830 cho thấy cách đo và proxy cũ có thể sai; chưa đủ để bỏ trụ nào của luận điểm | `plan/00`–`03`, PROPOSAL §1.3, §2.2, §5 |
| 2026-09-29 | Chốt benchmark trước khi có số G1/G2: headline = goodput (≥ 90% request đạt TTFT ≤ SLO_TTFT và mọi TBT ≤ SLO_TBT), so với baseline tốt nhất từng ô, số headline trên vLLM 0.30 | Theo cách Layered Prefill, SLOWeave, DistServe claim; trước đó metric của G1/G2 chưa định nghĩa ("goodput hoặc TTFT P99") | PROPOSAL §4.4; demo bước 0: theo TTFT trung bình thì layered chậm hơn chunked ~20% ở tải nhẹ, nhưng theo tỉ lệ đạt SLO thì cả hai 100% tới 2.5 req/s, chunked sụp ở 3.0 req/s |
