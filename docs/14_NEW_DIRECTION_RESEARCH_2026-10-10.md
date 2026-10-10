# Hướng mới: serving model hybrid cho workload agentic, triển khai aggregate (research 10/10/2026)

Tài liệu này tổng hợp research về hướng mới sau khi HyPrefill (chunk theo operator) cho kết quả yếu trên trace agentic. Nguồn: bốn lượt tra cứu song song (paper; upstream vLLM qua GitHub API; SGLang, TensorRT-LLM, LMCache, Dynamo và blog production; serving workload agentic), đối chiếu với số đo của dự án.

**Mức kiểm chứng.** Số liệu của paper lấy từ trang abstract hoặc bản HTML trên arXiv, không đọc toàn văn PDF. Số PR và issue của vLLM được tra trực tiếp qua `gh`. Mục ghi *(chưa kiểm)* chỉ thấy qua đoạn trích tìm kiếm. Nhiều paper 2026 mới là preprint một phiên bản, đăng trong hai tháng gần đây, chưa qua bình duyệt.

## 0. Tóm tắt

1. **Dừng HyPrefill làm luận điểm chính.** Kết quả đo:
   - Prompt dài tổng hợp (M1): +10–21%.
   - Trace coding agent, mức request (mô phỏng sơ bộ): goodput ≈ Layered (1.00× ở TBT 50 ms, 0.97× ở TBT 25 ms).

   Lý do: nhờ prefix cache, phần append của đa số request nhỏ. Prefill chỉ chiếm khoảng 10% thời gian GPU. Trong mô phỏng, iteration bị chặn bởi CPU. Các nghiên cứu độc lập cùng chỉ ra decode chiếm 91–98.6% thời gian LLM trên workload agent (IISWC'26).
2. **Giao điểm "agentic + hybrid + aggregate" có điểm đau thật, chưa ai giải trọn.** Workload agentic sống nhờ prefix cache (hit 85–98%). Nhưng với model hybrid, state GDN/Mamba chỉ tái dùng được **tại các mốc đã lưu**. Mất một mốc thì phải tính lại từ mốc trước, dù KV attention vẫn còn. Mọi issue lớn về prefix cache hybrid của vLLM đều nêu chat nhiều lượt và trace agent là trường hợp gãy. Không công trình quản lý KV cho agent nào (Continuum, KVFlow, CacheWise, LMCache…) mô hình hoá state hồi quy.
3. **Đề xuất:** hướng chính là **quản lý vòng đời mốc state cho model hybrid dưới workload agentic nhiều lượt**: đặt mốc gắn với ranh giới lượt và chunk, evict nhất quán giữa KV và mốc, phân tầng GPU/CPU kèm mô hình chi phí tính lại so với tải lại, giữ mốc theo phiên và tool call, cho kết quả chính xác. Hướng phụ là chi phí CPU của hybrid. Trước khi chốt, làm **nghiên cứu đặc trưng hoá 1–2 tuần trên vLLM thật**, với tiêu chí GO/KILL viết trước (§5).
4. **Rủi ro lớn nhất là cạnh tranh nhanh.** Tháng 8–9/2026 đã có Tail-Replay, DASC, Sparse Prefix Caching, Unified Radix Cache của SGLang và nhiều RFC vLLM. Độ mới phải nằm ở **kết hợp** đặt mốc + evict + phân tầng + nhận biết phiên agent, có bằng chứng đo trên trace agent thật, chứ không ở từng mảnh riêng.

## 1. Đặc trưng workload agentic (đã có nhiều số liệu độc lập)

| Nguồn | Điểm chính |
|---|---|
| TraceLab (arXiv 2606.30560, UW, 06/2026), ~4 300 phiên Claude Code/Codex | Prefix trung vị 126K (P90 467K), append 857 (P90 5.3K), output 252. Cache hit 95.7%, nhưng chỉ 84.4% ở bước do người dùng gửi; miss tập trung sau khoảng nghỉ của người > 5 phút. TTL 1 phút: hit 85.4%, prefill khuếch đại 18.9×; TTL 1 giờ: 98.6% nhưng tốn 5× bộ nhớ. Tool call trung vị 0.3 s, P90 13.6 s. Khuyến nghị: định tuyến theo độ dài append, evict theo độ trễ tool |
| Copilot traces (arXiv 2608.00101, Azure/UIUC) | 761 triệu lời gọi. Hit ~90% trong một lượt, ~55% qua ranh giới lượt (đổi model, nén context) |
| IISWC'26 (arXiv 2605.26297), vLLM 0.20, 2×H100 | Hit 84.6–99.5%; nhiều agent đồng thời làm hit rơi về 17–25%, throughput −66%. **Decode chiếm 91–98.6% thời gian LLM, prefill 1.4–9%**. Một trong hai model (Qwen3.6-27B) có thể là hybrid GDN nhưng bài coi như KV thuần *(chưa kiểm)* |
| cc-traces-weka (dự án đo, `results/step04/`) | Prefix cache phục vụ 96.2% token; append trung vị 1 600, p99 49K; append ≥ 16K chiếm 44% công prefill |

**Hệ quả:** trên workload agent đã có prefix cache, thắt cổ chai là **bộ nhớ giữ context giữa các lượt**, **tính lại sau khi bị evict**, decode ở context dài và CPU, không phải hiệu quả kernel prefill.

## 2. Hiện trạng

### 2.1 Nghiên cứu về serving model hybrid

- **Prefix cache cho state hồi quy:**
  - **Marconi** (MLSys'25, arXiv 2411.19379): chỉ giữ mốc ở cạnh cây radix; evict theo FLOP tiết kiệm được trên mỗi byte. Hit tăng tới 34.4×, TTFT −71%. Đây là bài duy nhất kết hợp hybrid với trace kiểu agent (SWE-Bench).
  - **Sparse Prefix Caching** (2605.05219): đặt mốc bằng quy hoạch động; tự ghi "kết hợp với evict kiểu Marconi" là việc tương lai.
  - **Tail-Replay** (2608.30310): bỏ mốc, tính gần đúng state bằng chạy lại 5–10% đuôi. Giữ 92.8–99.9% chất lượng, TTFT 9–14×. **Không chính xác tuyệt đối.**
  - **DASC** (2608.30386, Meituan): nén mốc theo tầm nhớ của từng head/channel, 2.63×, TTFT −42.6% ở cùng budget bộ nhớ.
  - **LinearKV** (2608.11231) và **HYPIC** (2607.01299): cache không phụ thuộc vị trí cho RAG/agent.
  - Case study GLM-5.3-Flash với vLLM và LMCache (2609.15030): lỗi lệch một token khi khôi phục; tải từ CPU giảm TTFT 46–64% so với tính lại.
- **Bộ nhớ:** Jenga (SOSP'25) cấp phát dị thể; AVMP (prototype nhỏ); PrfaaS (prefill xuyên datacenter cho hybrid).
- **Decode và độ chính xác:**
  - KVBuffer, DeltaLog: giảm I/O state khi decode.
  - LeapQuant, DAMP, Low-bit states: lượng tử hoá state. Các bài chưa thống nhất FP8 có an toàn không.
  - Spec decoding với state: STree, Bole, LumoTree.
- **Chưa có paper:**
  - chính sách offload state xuống CPU/SSD;
  - đặc trưng hoá chi phí CPU và launch của hybrid ở quy mô datacenter;
  - lập lịch prefill theo loại phép tính (chính ý tưởng HyPrefill).

### 2.2 Nghiên cứu quản lý KV và lập lịch cho agent (đều không tính tới hybrid)

- **Giữ cache quanh tool call và phiên:** Continuum (TTL quanh tool call, JCT > 8×), KVFlow (NeurIPS'25), CacheScout, CacheWise, TOPAS.
- **Lưu trữ phân tầng:** CachedAttention (ATC'24), Pensieve (EuroSys'25), IMPRESS (FAST'25), LMCache (MLSys'26).
- **Lập lịch:** Autellix, Parrot, Astraea, SMetric, AgentServe, SmoothAgent.
- Tất cả giả định KV cắt được ở bất kỳ vị trí nào.

### 2.3 Upstream vLLM (tra qua GitHub API)

- **Prefix cache hybrid:**
  - chỉ còn chế độ `align` (`all` bị gỡ ở #58997, 01/10/2026);
  - block 528–2096 token;
  - mỗi request giữ một mốc.
- **Các vấn đề đang mở:**
  - **#45238:** mốc rơi vào token riêng của request thì reuse về 0%, kể cả block attention khớp hoàn toàn (TTFT 433 so với 905 ms).
  - **#40696:** prompt ngắn hơn một block thì hit 0%.
  - **RFC #57111:** attention bị evict từ đuôi, mốc state bị evict từ đầu, nên dưới áp lực bộ nhớ nhẹ một request nhiều lượt có thể mất đoạn đuôi nhỏ mà phải tính lại gần như toàn bộ context.
  - **RFC #55697:** mốc do ứng dụng chỉ định (QPS 2.1×).
  - **#60008:** chế độ align làm mất 11–16% throughput dù không có hit (thêm 28 launch eager mỗi step decode, kernel align ngoài CUDA graph).
- **Đã merge:** partial hit (#45939, #46384), retention interval (#45845: mốc dày chiếm 80% pool), mốc nội bộ khi prefill (#52789 Kimi, #57329 Mamba2). Bản cho GDN (#60050, #60659) còn mở.
- **Offload:** có hỗ trợ state Mamba nhưng nhiều lỗi; #52773 (hit lệch nhau giữa các nhóm bị bỏ), #46455, #58653, #50454 (crash). LMCache với hybrid không cho kết quả giống hệt bit.
- **CPU:**
  - Lõi GDN chạy eager dưới piecewise graph.
  - #27222 (Qwen3-Next tốn CPU hơn GPU trên B200; tăng cỡ capture 512 → 1024 nâng throughput 13.9K → 21K tok/s) bị đóng vì quá hạn mà chưa sửa.
  - #50780: profiler CUDA graph làm KV pool nhỏ đi 18–24% với model GDN.
- **Roadmap Q3/2026 (#48168):** lấy agentic làm chủ đề chính (session hint, prefetch, evict, offload nhiều tầng), nhưng không mục nào nói riêng về hybrid.
- **Không có RFC nào** về chunk theo operator hay theo nhóm layer.

### 2.4 Các engine khác

- **SGLang:** đi xa nhất.
  - Unified Radix Cache (08/2026): một cây chung cho FULL/SWA/MAMBA.
  - HiCache L1/L2/L3 qua Mooncake: L3 hit 96.8%, 67.1K so với 15.5K tok/s chỉ dùng L1.
  - Evict theo phiên: device hit 5% → 34% trên SWE-bench.
  - Hạn chế: mốc chỉ đặt ở ranh giới chunk prefill và theo chu kỳ khi decode; evict theo phiên chưa tới L3; RFC #40865 ghi workload agent nhiều lượt nhỏ chịu thiệt nhất vì hit bị kéo lùi về mốc.
- **TensorRT-LLM:** snapshot định kỳ, tắt mặc định; quick start Qwen3-Next còn tắt luôn block reuse.
- **TokenSpeed:** slot làm việc + slot mốc; benchmark agent hit > 90%.
- **Dynamo, llm-d:** router không biết mốc state còn hay mất, nên ước lượng quá mức khả năng reuse.
- **MiniMax quay về full attention cho M2**, lý do gồm prefix cache với traffic chat hit cao, state nhạy độ chính xác thấp, và spec decoding.

## 3. Bản đồ vấn đề

| Vấn đề | Riêng hybrid? | Riêng agentic? | Đã có ai làm | Khoảng trống | Bằng chứng của dự án |
|---|---|---|---|---|---|
| **Mốc state: đặt ở đâu, giữ bao lâu, evict thế nào** | Có (chỉ reuse tại mốc; mất mốc phải tính lại cả đoạn) | Có (nhiều lượt, fork sub-agent, khoảng nghỉ tool/người) | Marconi (evict), Sparse PC (đặt), DASC (nén), SGLang URC, vLLM RFC #57111/#55697 | Chưa ai làm đồng thời đặt mốc + evict + phân tầng + nhận biết phiên trên trace agent thật | Mỗi mốc (state của cả 36 layer GDN) ~19.8 MB mỗi GPU, gấp 3 KV attention của một block 544 token (~6.7 MB); đọc mã `docs/06_DESIGN.md` §9 |
| **Offload state xuống CPU/SSD** | Có (state lớn cố định; tải lại hay tính lại?) | Có (khoảng nghỉ dài) | LMCache (trang mờ), SGLang HiCache, case study GLM | **Chưa có paper** về chính sách; vLLM nhiều lỗi | — |
| **Chi phí CPU và launch** | Phần lớn có (GDN eager, align ngoài graph) | Một phần (append nhỏ → step ngắn → CPU lộ ra) | Sửa lẻ (#22594, #52789); #27222 đóng chưa sửa | **Chưa có paper đặc trưng hoá** | `host_ms` GDN 0.275 ms × 36; step vượt cỡ graph chạy eager ~90 ms; mô phỏng: 60–92% iteration bị chặn bởi CPU |
| Thứ tự evict KV và mốc ngược nhau | Có | Có | RFC #57111 (mở) | Chưa có lời giải tổng quát | — |
| Router không biết mốc | Có | Có (định tuyến theo phiên) | llm-d WIP (chỉ SWA) | Mở | Ngoài phạm vi một instance |
| Lập lịch prefill theo loại phép tính (HyPrefill) | Có | Không (lợi ở append lớn) | Không ai | Có khoảng trống nhưng lợi nhỏ trên agentic | M1 1.1–1.2×; mức request ≈ 1.0× |
| Độ chính xác state | Có | Không | LeapQuant, DAMP, low-bit (08–09/2026) | Đông, nhanh | — |
| Spec decoding + prefix cache | Có | Không | Bole, STree, RFC #52817 | Phần lớn là sửa lỗi đúng/sai | — |

## 4. Các hướng, xếp hạng

**H1 (đề xuất chính): quản lý vòng đời mốc state cho hybrid dưới workload agentic.**

Hệ thống trong vLLM gồm bốn phần ăn khớp nhau:
1. **Đặt mốc** gắn với ranh giới lượt, điểm fork sub-agent và chunk prefill. Đồng thiết kế với lịch prefill: phần còn dùng được của ý tưởng commit theo nhóm ở `docs/06_DESIGN.md` §9.
2. **Evict nhất quán giữa KV attention và mốc,** theo giá trị thật: số token phải tính lại nếu mất (giải đúng bài toán của #57111).
3. **Phân tầng GPU → CPU (→ SSD),** với mô hình chi phí tải lại state lớn cố định so với tính lại từ mốc trước.
4. **Giữ mốc theo phiên:** TTL quanh tool call và khoảng nghỉ người, như Continuum nhưng cho state hồi quy.

Hướng này chính xác tuyệt đối (khác Tail-Replay và DASC). Đo bằng TTFT/goodput trên trace agent thật, dưới áp lực bộ nhớ và nhiều phiên đồng thời.

- *Điểm mạnh:* đúng giao điểm agentic × hybrid × aggregate; điểm đau có trong issue upstream và trong lý do MiniMax bỏ hybrid; các paper hiện có tự ghi phần kết hợp là việc tương lai.
- *Rủi ro:* lĩnh vực đông và nhanh (nhiều preprint 08–09/2026; SGLang đã có phân tầng và evict theo phiên); upstream có thể làm trước một phần. Cần chứng minh lợi ích **lớn** so với SGLang URC/HiCache và vLLM align + retention + offload, không chỉ so với mặc định.

**H2 (hướng phụ, có thể là một phần của H1 hoặc bài riêng): chi phí CPU của hybrid trong serving aggregate.**
- Bằng chứng: số đo của dự án cộng các issue chưa sửa (#27222, #60008, #50780).
- Lời giải có thể: đưa đường đi GDN của batch trộn vào graph, gộp metadata, giảm chi phí align.
- *Rủi ro:* độ mới học thuật vừa phải, dễ bị coi là kỹ thuật thuần.

**H3 (an toàn, có thể làm song song): đặc trưng hoá serving hybrid ở quy mô datacenter trên trace agentic.**
- Chưa có bài nào như vậy: các bài đặc trưng hoá agent đều không tách hybrid, bài đặc trưng hoá hybrid chỉ có GPU biên.
- Phù hợp IISWC, SIGMETRICS, MLSys; cũng là phương án lui đo đạc (PROPOSAL §7).
- Là phần động lực bắt buộc của H1.

**Không khuyến nghị làm hướng chính:**
- Lượng tử hoá state (đông, nhiều bài 08–09/2026).
- Spec decoding với hybrid (đông; phần lớn là sửa lỗi đúng/sai).
- Router nhận biết mốc (cần nhiều instance, lệch khỏi khung aggregate một instance).
- HyPrefill (lợi nhỏ trên agentic).

## 5. Bước tiếp: nghiên cứu đặc trưng hoá (1–2 tuần), tiêu chí viết trước

**Thiết lập:**
- vLLM 0.30, Qwen3-Next-80B-A3B TP2 trên 2×H200 (GPU 4–7), bật prefix cache (align + retention mặc định), bộ nhớ thật.
- Phát lại cc-traces-weka theo đúng nhịp phiên, gồm khoảng nghỉ tool/người và sub-agent; quét số phiên đồng thời.
- So với một model thuần attention trên cùng trace (Qwen3-30B-A3B TP2) để tách phần riêng hybrid.
- Thêm CPU offload (OffloadingConnector) như một cấu hình.

**Đo** (thêm đếm vào vLLM nơi cần):
- TTFT/TBT phân rã: chờ, prefill, decode, CPU.
- Hit theo từng nhóm KV (attention và từng nhóm GDN).
- **Số token phải tính lại do mất mốc trong khi KV attention vẫn còn.**
- Số lần evict theo loại, bộ nhớ dành cho mốc.
- Phần step chạy eager, CPU mỗi step.
- Goodput theo PROPOSAL §4.4.

**Tiêu chí (đề xuất, chờ duyệt trước khi chạy):**
- **H1 GO:** ở mức đồng thời mà KV pool đầy ≥ 80%, token tính lại do mất mốc (dù KV attention còn) ≥ 20% tổng token prefill, **và** TTFT P90 của hybrid tăng ≥ 1.5× so với khi không mất mốc (mô phỏng bằng cách giữ mốc vô hạn).
- **H2 GO:** CPU chiếm ≥ 30% thời gian step ở mức tải goodput.
- **Cả hai trượt:** viết bài H3 (đặc trưng hoá) và tìm thắt cổ chai khác từ dữ liệu đo.

## 6. Nguồn chính

- Paper: Marconi https://arxiv.org/abs/2411.19379 · Sparse Prefix Caching https://arxiv.org/abs/2605.05219 · Tail-Replay https://arxiv.org/abs/2608.30310 · DASC https://arxiv.org/abs/2608.30386 · LinearKV https://arxiv.org/abs/2608.11231 · HYPIC https://arxiv.org/abs/2607.01299 · Jenga https://arxiv.org/abs/2503.18292 · TraceLab https://arxiv.org/abs/2606.30560 · Copilot traces https://arxiv.org/abs/2608.00101 · IISWC'26 agentic https://arxiv.org/abs/2605.26297 · Continuum https://arxiv.org/abs/2511.02230 · KVFlow https://arxiv.org/abs/2507.07400 · CacheWise https://arxiv.org/abs/2606.16824 · LMCache https://arxiv.org/abs/2510.09665 · KVBuffer https://arxiv.org/abs/2605.19049 · LeapQuant https://arxiv.org/abs/2609.38166 · DAMP https://arxiv.org/abs/2608.27513
- vLLM: #45238, #40696, #57111, #55697, #60008, #45845, #45939, #46384, #52789, #57329, #60050, #60659, #52773, #46455, #27222, #50780, #37121, #55196, #48168, #58997
- SGLang và các engine khác: Unified Radix Cache https://www.lmsys.org/blog/2026-08-11-unified-radix-cache · Hybrid Models Meet SGLang https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ · TensorRT-LLM KV cache docs · TokenSpeed (PyTorch blog 2026-05-28) · LMCache hybrid https://docs.lmcache.ai/mp/hybrid_models.html · MiniMax M2 https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model · vLLM AgentX blog https://vllm.ai/blog/2026-09-08-vllm-agentx
- Ghi chép thô của bốn lượt tra cứu nằm ngoài repo (scratchpad của phiên).
