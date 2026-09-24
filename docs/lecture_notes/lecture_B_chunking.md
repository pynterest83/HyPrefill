# Bài giảng B — Ba công trình nền tảng cho HyPrefill (chunked prefill theo nhóm operator)

Ghi chú chuẩn bị cho nghiên cứu sinh trước khi bắt đầu HyPrefill: per-operator-group chunk sizes trong chunked prefill cho các mô hình hybrid (full attention + Gated DeltaNet/linear attention + MoE), mục tiêu giảm TTFT tại P99 TBT cố định.

---

## 1. Layered Prefill (arXiv 2510.08055, MLSys 2026 Oral)
### "From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill"

### Bối cảnh và vấn đề

Layered Prefill xuất phát từ một quan sát rất cụ thể: chunked prefill (kiểu Sarathi-Serve — cắt prefill thành các chunk token cố định rồi trộn với decode trong cùng iteration để giữ TBT thấp) hoạt động tốt với mô hình dense, nhưng "phá" tính thưa (sparsity) của MoE. Theo abstract, chunked prefill "incurs substantial overhead in Mixture-of-Experts (MoE) models: redundant expert weight loads increase memory traffic by up to 39% and inflate energy consumption." Cơ chế gốc rễ được đặt tên là **sparsity erosion**: "chunking inflates expert coverage while suppressing reuse, eroding the very sparsity benefits that make MoE efficient." Vì mỗi request được chia làm nhiều chunk, và mỗi chunk lại được gộp vào một hybrid-batch (prefill + decode) rất lớn (batch size thường >256), nên hầu hết các expert đều bị "chạm" tới mỗi iteration dù chỉ để phục vụ vài token — expert weight phải load lại nhiều lần cho cùng một request.

Số liệu minh chứng: Bảng 1 (Qwen3-30B-A3B, dữ liệu ShareGPT) cho thấy expert coverage tăng theo batch size — dưới 50% ở batch=16, dưới 70% ở batch=64, và lên tới 86.3% ở batch=128 — nghĩa là batch càng lớn (điều mà chunked prefill tạo ra khi trộn nhiều request/chunk), càng nhiều expert bị kích hoạt nhưng mỗi expert chỉ nhận rất ít token/expert (dưới ridge point ~100-300 Op/B của accelerator hiện đại), khiến việc thực thi rơi vào **memory-bound** thay vì compute-bound — cái giá phải trả là băng thông load trọng số expert, không phải FLOPs.

Con số "39% redundant expert traffic" được đo trực tiếp: Bảng 7 chạy 100 request thật trên Qwen, đo tổng dung lượng trọng số expert phải load (GB/tổng traffic) khi dùng chunked prefill (chunk=512) so với layered prefill. Trên tập arXiv (prompt dài, trung bình 9,194 token): traffic giảm từ 35.6 TB xuống 21.7 TB (−39%). Trên ShareGPT (prompt ngắn, trung bình 2,340 token) mức giảm chỉ 12% (28.5 TB → 25.1 TB) — vì prompt ngắn vốn đã ít bị chia chunk. Ngoài ra Bảng 2 minh họa rõ trade-off cố hữu: tăng chunk 512→2048 giảm expert load 955→304 GB/req (−68%) và energy 60.2→32.4 mJ/tok (−46%), nhưng P99 TBT lại tăng 48.4ms→129ms — vi phạm SLO. Đây chính là "chunk-size dilemma" mà cả ba bài trong lecture này đều cố giải theo cách khác nhau.

### Cơ chế

Thay vì chia theo **token** (chunk theo chiều ngang của sequence), Layered Prefill chia theo **layer** (chunk theo chiều dọc của model). Số nhóm layer được định nghĩa:

  N_lg(L) = max(1, ⌈L/512⌉)

với L là độ dài prompt — công thức này chỉ nhằm cân bằng khối lượng công việc prefill mỗi iteration xấp xỉ với baseline chunk=512, để so sánh công bằng.

Nguyên tắc lập lịch cốt lõi: "in each iteration, only one designated layer group performs both decode and prefill for newly admitted requests, while all other groups perform decode only." Tức là: model được chia dọc thành N_lg nhóm layer liên tiếp; ở iteration 1, nhóm layer 1 vừa chạy decode (cho token đang sinh) vừa chạy prefill (cho toàn bộ prompt, không chunk theo token) cho request mới; các nhóm layer còn lại chỉ chạy decode. Ở iteration 2, nhóm layer 2 đảm nhận prefill, v.v., cho tới khi N_lg iterations trôi qua và toàn bộ stack đã prefill xong cho request đó — trong khi decode của các request khác không hề bị gián đoạn suốt quá trình.

Hệ quả thuật toán quan trọng: "each input prompt traverses the prefill path of each layer exactly once, as opposed to chunked prefill, which forces every layer to process the prompt once per chunk." Đây chính là lý do sparsity được bảo toàn — mỗi expert trong mỗi layer chỉ thấy toàn bộ prompt một lần duy nhất (batch prefill lớn, dùng đúng expert cần dùng), thay vì thấy nhiều lần nhỏ lẻ qua nhiều chunk.

Về bộ nhớ cho activation trung gian: bài báo **không đặc tả rõ** cơ chế stash/offload activation giữa các nhóm layer khi một request đang "lửng lơ" giữa nhóm đã prefill và nhóm chưa prefill — chỉ ngầm định rằng decode ở tất cả các nhóm layer tiếp tục diễn ra bình thường, và KV cache tích lũy tuần tự qua các iteration như thường lệ. Đây là một khoảng trống kỹ thuật đáng chú ý.

### Kết quả số

Phần cứng: 2×H100-80GB (NVLink, tensor parallel) là setup chính; kiểm chứng mở rộng trên A100 (2-8 GPU) và H100×8. Model: Qwen3-30B-A3B (128 experts, top-8) và GPT-OSS-20B (32 experts, top-4); dữ liệu ShareGPT và arXiv. Baseline: chunked prefill kiểu Sarathi-Serve (chunk=512), prefill/decode disaggregation, và dense baseline (Qwen3-8B, không MoE).

Kết quả tiêu biểu (Bảng 6, Qwen/arXiv, 1.3 req/s): Mean TTFT giảm 56% (2.80s→1.24s), P99 TTFT giảm 53%, Mean TBT giảm 35%, P99 TBT giảm 27%. Bảng 3 (SLO attainment 90%): layered prefill nâng capacity chịu tải thêm 13-17% tùy cấu hình. Bảng 8: energy/token giảm 20-22% tại rate SLO-compliant. Bảng 9 (kiểm chứng rộng): H100×2/GPT-OSS-120B TTFT giảm 65%; H100×8/Qwen3-235B giảm 47%. Bảng 11 khảo sát trade-off N_lg: N_lg nhỏ (2) cho TTFT thấp nhất nhưng P99 TBT rất cao (138ms); N_lg lớn (16) đảo ngược lại (TTFT 1.27s, P99 TBT 35.7ms) — đúng kiểu trade-off mà công thức N_lg(L) cố định (512 token/nhóm) không tự động tối ưu theo SLO.

Ablation quan trọng nhất: trên dense model (Qwen3-8B, không MoE), layered prefill **thua** chunked prefill (TTFT 1.45s vs 0.955s) — xác nhận rõ ràng phương pháp chỉ có lợi khi có MoE sparsity để khai thác.

### Điểm yếu / giới hạn

Bài báo tự thừa nhận: "layered prefill is most effective for MoE models where expert activation is sparse and strict TBT SLOs require small chunk sizes" — với dense model nó phản tác dụng. **Không có bất kỳ đề cập nào tới SSM, Mamba, Gated DeltaNet, hay linear attention** — phạm vi hoàn toàn giới hạn ở full-attention + MoE decoder truyền thống. Việc quản lý bộ nhớ activation trung gian giữa các nhóm layer chưa được đặc tả chi tiết (rủi ro khi nhiều request "lửng lơ" ở các nhóm khác nhau cùng lúc). N_lg(L) là công thức tĩnh (dựa trên độ dài prompt, chia đều 512 token/nhóm) chứ không tối ưu động theo SLO thực tế — Bảng 11 cho thấy lựa chọn N_lg tối ưu phụ thuộc mạnh vào target TBT. Tác giả cũng để ngỏ multi-GPU phức tạp hơn và trường hợp layer count không chia hết cho group count là future work.

### Rút ra cho HyPrefill

Đây là baseline **quan trọng nhất** cần replicate: cùng hardware class (H100), cùng model họ MoE (Qwen3-30B-A3B là lựa chọn tốt để so sánh trực tiếp), và nhất thiết phải tái tạo Bảng 6 (TTFT/TBT) và Bảng 7 (expert traffic) làm điểm neo. Luận điểm "chunk theo token phá sparsity của MoE" là lý lẽ HyPrefill nên kế thừa và mở rộng: HyPrefill lý luận thêm rằng chunk theo token cũng phá tính hiệu quả của **Gated DeltaNet/linear attention** (state theo chunk, chunk nhỏ → nhiều overhead recompute state) — một hướng mà Layered Prefill hoàn toàn bỏ ngỏ. Khác biệt cốt lõi: Layered Prefill chọn **một** kích thước "chunk" (theo layer-group, cố định N_lg(L)) áp dụng đồng nhất, trong khi HyPrefill đề xuất **chunk size khác nhau cho từng nhóm operator** (full-attn, GDN, MoE-FFN) trong cùng một layer/iteration — tức là chi tiết hơn một bậc so với chia theo layer. Reviewer đã đọc Layered Prefill chắc chắn sẽ hỏi: (a) HyPrefill có quản lý activation trung gian giữa các nhóm operator tốt hơn Layered Prefill không (vì đây là điểm họ bỏ ngỏ)? (b) Nếu chỉ dùng full-attn+MoE (không có GDN), HyPrefill có suy biến về đúng Layered Prefill không, và có đánh bại nó trên chính benchmark của họ không?

### Hai câu hỏi tự kiểm tra

1. Tại sao batch size lớn hơn lại làm tăng expert coverage nhưng KHÔNG tăng compute-intensity mỗi expert, dẫn tới memory-bound?
   *Đáp:* Batch lớn khiến nhiều request/chunk khác nhau được gộp trong 1 iteration, do đó nhiều expert khác nhau được các token khác nhau route tới (coverage cao), nhưng tổng token của mỗi request nhỏ nên mỗi expert vẫn chỉ nhận trung bình rất ít token (~128), dưới ridge point, nên thời gian bị chi phối bởi việc load trọng số expert (memory-bound) chứ không phải bởi FLOPs.

2. Vì sao Layered Prefill lại thua chunked prefill trên mô hình dense (Qwen3-8B)?
   *Đáp:* Vì lợi ích của Layered Prefill đến từ việc tránh load lại expert weight nhiều lần — dense model không có expert routing/sparsity nên không có "redundant expert traffic" để tiết kiệm; ngược lại việc chia theo layer-group làm tăng số iteration cần thiết để hoàn tất prefill toàn bộ stack, gây overhead thuần túy.

---

## 2. SLOWeave (arXiv 2609.07883)
### "Deadline-Aware Adaptive Prefill Chunking for Efficient LLM Serving"

### Bối cảnh và vấn đề

SLOWeave tấn công vào chính điểm yếu của mọi baseline chọn **chunk size cố định**: kích thước chunk tối ưu là hàm của nhiều biến số thay đổi liên tục theo thời gian thực — batch size hiện tại, độ dài prompt, mục tiêu TPOT (time-per-output-token), hàm chi phí phần cứng, và cường độ arrival của request. Bài báo viết rõ: "C⋆=C⋆(n,L,D,T,λ)" — chunk tối ưu C* phụ thuộc vào n (batch size), L (độ dài prompt), D (target TPOT), T (hàm chi phí), λ (áp lực arrival). Một chunk size tĩnh (tune offline) chỉ "trúng" đúng một điểm trong không gian đa chiều này; khi điều kiện runtime thay đổi (traffic tăng đột biến, prompt dài hơn dự kiến...), chunk cố định lập tức trở nên dưới tối ưu hoặc vi phạm SLO.

Về trực giác: "A small chunk limits interruption time but needs many iterations, each with scheduling and kernel-launch overhead. A large chunk amortizes overhead and reaches the first token quickly in isolation, but may violate the time-per-output-token (TPOT) objective of every active request." Motivation minh họa bằng timeline khái niệm (Hình 2): prefill nguyên khối (full prefill) "vượt" deadline decode; chunk cố định thì "stranding slack in every window" (lãng phí phần budget dư ra mỗi cửa sổ); còn SLOWeave thì "expands or contracts c_t to fill the available budget as decode load changes" — co giãn theo tải thực tế.

Số liệu định lượng ở phần kết quả chính (Bảng 3, TPOT mục tiêu 25ms): full-prefill trên traffic hỗn hợp chỉ đạt 6.9% SLO attainment (cả TTFT lẫn TPOT); trên long-context gần như sụp đổ hoàn toàn (~0.1% goodput đạt SLO). Fixed-1024 (baseline cố định tốt nhất) đạt 59.4% SLO attainment trên mixed traffic, còn SLOWeave đạt 79.4%. Ở mục tiêu TPOT ngặt hơn (10ms), khoảng cách giãn mạnh: SLOWeave đạt 35.7 req/s trên mixed traffic so với 10.7 req/s của baseline cố định tốt nhất — cải thiện 3.3×.

### Cơ chế

Mô hình chi phí iteration: T(n, c), với n = |A_t| là số request đang decode hoạt động, c là số token prefill được admit trong iteration đó. Điểm mấu chốt về tính tổng quát: "The function can be a lookup table profiled by the runtime, a fitted regressor, or a conservative analytical model. SLOWeave requires only monotonicity in c, not a particular functional form." — tức là T chỉ cần đơn điệu tăng theo c, không cần biết dạng hàm cụ thể. Mô hình synthetic mặc định dùng trong bài:

  T(n,c) = 0.35 + 𝟙[n>0]·(0.90+0.055n) + 𝟙[c>0]·(0.40+0.006c)  (ms)

Mô hình deadline: mỗi request i đang decode có thời điểm hoàn tất token gần nhất ℓ_i và mục tiêu TPOT D, suy ra deadline token kế tiếp d_i = ℓ_i + D. Tại thời điểm lập lịch s_t, ngân sách iteration là:

  B_t = max(0, min_{i∈A_t} (d_i − s_t))

— tức là "khe hở" (slack) sớm nhất trong tất cả các deadline decode đang hoạt động.

Công thức chọn chunk (Equation 1):

  c*_t = max{ c ∈ ℤ≥0 : c ≤ min(C_max, L_t), T(|A_t|, c) ≤ B_t }

với L_t là độ dài còn lại của prefill đang chờ sớm nhất, C_max là giới hạn triển khai. Nhờ tính đơn điệu của T theo c, việc tìm c*_t có thể thực hiện bằng **binary search theo log-time** (Algorithm 1): nếu không có prefill đang chờ, trả về 0; nếu không có decode nào đang hoạt động, trả về L (prefill toàn bộ); ngược lại binary search trong [0, L], kiểm tra T̂(|A|, mid) + δ(|A|, mid) ≤ B tại mỗi bước, hội tụ về chunk an toàn lớn nhất — độ phức tạp O(log C_max) lần gọi cost model.

Mệnh đề 1 (an toàn & tối đa): nếu cost model chính xác và T đơn điệu theo c, chunk được chọn "(i) hoàn tất trước mọi deadline decode kế tiếp đang hoạt động, và (ii) xử lý ít nhất bằng bất kỳ chunk an toàn-deadline nào khác" — tức vừa an toàn vừa tối ưu (maximal) trong lớp các lựa chọn an toàn. Với mô hình học được (không chính xác tuyệt đối), bài báo thêm biên sai số δ(n,c) và kiểm tra T̂(n,c) + δ(n,c) ≤ B_t; mệnh đề vẫn đúng nếu sai số một phía bị chặn bởi δ.

### Kết quả số

Phần cứng: 8×A100-80GB và 8×H100-80GB (single-node); model 8B dense (1 GPU), 70B dense (tensor-parallel 4-8 GPU), và một biến thể 8B tối ưu long-context. Baseline: cấu hình mặc định của runtime, full-prompt prefill, fixed chunk {64, 256, 1024}, chunking kiểu Sarathi đã tune, và SLOWeave (C_max=4096).

Bảng 2 (GPU thật, TPOT SLO=25ms): A100/8B mixed traffic — P99 TTFT giảm từ 1780ms (default) / 1510ms (fixed-1024) xuống 1190ms (SLOWeave); goodput tăng từ 38.4 / 52.7 lên 67.1 req/s. H100/70B-TP8 long-context: P99 TTFT 5940→3960ms, goodput 22.1→40.8 req/s. Bảng 3 (simulator, TPOT=25ms): mixed workload SLOWeave đạt 59.0 req/s vs 42.4 req/s baseline mạnh nhất (+39%); long-context +38%; bursty +9%. Bảng 4 (tại các TPOT khác nhau): ở TPOT=10ms, cải thiện lên tới 3.35× (mixed), 2.45× (long), 2.39× (bursty); ở TPOT=50ms gần như ngang bằng full prefill (chunking ít cần thiết khi SLO lỏng). SLO attainment tại TPOT=25ms: mixed 79.4% (SLOWeave) vs 59.4% (fixed-1024); long 66.2% vs 51.4%; bursty 88.6% vs 83.0%.

### Điểm yếu / giới hạn

Tác giả tự nêu Limitations rõ ràng: "Our evaluation covers a finite set of models, accelerators, workload traces, and SLO targets." Về mô hình chi phí: "Real iteration cost can be non-smooth because of kernel boundaries, tensor-parallel communication, memory pressure, and prefix-cache hits" — tức mô hình T(n,c) đơn điệu/trơn có thể không khớp thực tế trong mọi trường hợp. Về multi-tenant: "starvation and admission control deserve separate study." Quan trọng nhất cho HyPrefill: **SLOWeave chọn một chunk size c_t* duy nhất áp dụng cho toàn bộ prefill của iteration đó — không có khái niệm chunk khác nhau theo layer hay theo loại operator.** Bài báo cũng hoàn toàn không đề cập MoE, SSM/hybrid attention, hay Gated DeltaNet — chỉ đánh giá trên decoder dense chuẩn (8B, 70B). Ngoài ra: "We also model no preemption cost and no KV transfer" — bỏ qua chi phí khi có prefill/decode disaggregation hoặc preemption.

### Rút ra cho HyPrefill

SLOWeave cho HyPrefill một công cụ toán học rất đáng vay mượn: **lý luận đơn điệu + binary search log-time** để chọn chunk theo deadline, thay vì heuristic tĩnh. HyPrefill nên tái sử dụng chính xác form B_t = min slack còn lại và ràng buộc T(...) ≤ B_t, nhưng mở rộng nó thành **vector chunk (c_attn, c_GDN, c_MoE)** cho từng nhóm operator, với hàm chi phí T tách theo operator (vì full-attn, GDN, và MoE-FFN có profile chi phí rất khác nhau theo chunk size — GDN cần chunk đủ lớn để amortize overhead recurrence, còn MoE cần chunk đủ lớn để đạt ridge point cho expert). Cần replicate Bảng 2/3 làm baseline "single global chunk, deadline-aware" — đây là baseline mạnh hơn Layered Prefill về mặt thích nghi runtime, nên là đối trọng trực tiếp nhất của HyPrefill. Reviewer từng đọc SLOWeave chắc chắn hỏi: "Tại sao không đơn giản chạy SLOWeave riêng cho mỗi operator-group thay vì thiết kế mới?" — câu trả lời HyPrefill cần chuẩn bị: các operator trong cùng layer chia sẻ chung một deadline decode (cùng B_t) nhưng có cost model T khác nhau và ràng buộc phụ thuộc lẫn nhau (chunk GDN ảnh hưởng tới state truyền sang layer sau, chunk MoE ảnh hưởng tới expert reuse) — đây là bài toán tối ưu đa biến ràng buộc chung, không tách rời được thành N lần chạy SLOWeave độc lập.

### Hai câu hỏi tự kiểm tra

1. Tại sao SLOWeave chỉ cần T(n,c) đơn điệu theo c mà không cần biết dạng hàm cụ thể để search hiệu quả?
   *Đáp:* Vì binary search chỉ cần tính chất "tăng c thì T tăng" để loại bỏ đúng một nửa khoảng tìm kiếm mỗi bước dựa trên so sánh T(mid) với B — không cần đạo hàm hay dạng giải tích, nên hoạt động với lookup table, regressor, hay bất kỳ cost model nào miễn đơn điệu.

2. Vì sao ở TPOT=50ms (SLO lỏng), lợi ích của SLOWeave gần như biến mất so với full prefill?
   *Đáp:* Vì khi ngân sách iteration B_t rộng, ngay cả full prefill (chunk = toàn bộ L) cũng thỏa T(n,L) ≤ B_t trong hầu hết trường hợp, nên không còn xung đột giữa "prefill nhanh" và "không vi phạm deadline decode" — bài toán mà SLOWeave giải quyết trở nên không còn ràng buộc chặt.

---

## 3. COREY (arXiv 2604.10597)
### "Entropy-Guided Runtime Chunk Scheduling for Selective Scan Kernels"

### Bối cảnh và vấn đề

COREY nhắm vào Mamba/selective-scan (SSM), không phải attention hay MoE — nhưng cực kỳ liên quan tới HyPrefill vì Gated DeltaNet cũng là một dạng linear-attention/recurrent-scan có đặc tính chunk-size-sensitive tương tự. Quan sát khởi điểm: "chunk size, fusion boundaries, and per-call kernel-launch frequency can change end-to-end latency by an order of magnitude" dù kết quả toán học (giá trị đầu ra) không đổi — tức đây thuần túy là vấn đề hiệu năng kernel, không phải chất lượng mô hình. Thực hành hiện tại là profiling offline để chọn một chunk size tĩnh cho mỗi tổ hợp model-hardware-workload, nhưng cách này "scales poorly with the deployment matrix and offers no guarantees once the workload distribution shifts" — không tổng quát hóa khi phân phối input thay đổi.

Bằng chứng định lượng: chọn đúng chunk (512) thay vì chunk tĩnh=64 cho tốc độ nhanh hơn 4.41× trên GPU consumer (RTX 3070); trên accelerator datacenter mức cải thiện là 3.90×–4.04×. Điều thú vị và là nền tảng cho ý tưởng "entropy-guided": phân tích activation thật trên checkpoint Mamba-370M cho thấy "inter-layer entropy spans 2.27–3.61 nats" qua 7 layer được lấy mẫu — tức là các layer khác nhau có phân phối activation (và do đó "độ khó" cho scan kernel) khác nhau, gợi ý rằng có thể dùng entropy của activation làm tín hiệu để chọn chunk size runtime thay vì offline. Tác giả tự gọi đây là "concept-and-feasibility contribution", giới hạn hoàn toàn ở họ Mamba-1.x.

### Cơ chế

Đo entropy Shannon bằng histogram cố định K bin trên tensor activation Z:

  Ĥ(Z) = −Σ_{k=1}^{K} p_k log(p_k + ε)

với p_k là xác suất bin k, ε>0 để ổn định số học. Chuẩn hóa theo số bin để bất biến với độ phân giải histogram:

  H̃ = Ĥ / log K ∈ [0,1]

Giá trị chuẩn hóa r := H̃ trở thành tín hiệu lập lịch, độc lập với K (kiểm chứng với K ∈ {32,...,1024}). Tham số hiệu chỉnh then chốt: **H_ref = log K** — với K=256 bin dùng trong checkpoint hook thực tế, H_ref = log 256 ≈ 5.55 nats. Bài báo lưu ý một phiên bản cũ dùng H_ref = 8.0 cố định đã đánh giá thấp "ngân sách entropy", làm thiên lệch lựa chọn về phía chunk nhỏ hơn mức cần thiết — công thức H_ref = log K sửa lỗi này bằng cách gắn ngưỡng tham chiếu trực tiếp vào entropy tối đa lý thuyết của histogram K-bin (phân phối đều).

Công thức ánh xạ entropy → chunk size (dạng lũy thừa 2):

  C_COREY = clip( 2^round(log2(C_min + r·(C_max − C_min))), C_min, C_max )

với C_min=32, C_max=512 — nội suy tuyến tính trong không gian r, sau đó làm tròn về log2 gần nhất để ra chunk dạng lũy thừa 2 (thân thiện kernel). Bộ lập lịch chạy **inline trong prefill**, trên hidden state sau convolution, tính entropy rồi định tuyến chunk đã chọn tới một scan kernel đã patch (recurrence-preserving) mà không đổi kết quả toán học.

### Kết quả số

Ba tầng đánh giá rõ ràng: (1) **Tier-1 Prototype**: cost model Python xác định trên activation synthetic, chỉ mang tính chẩn đoán, không chạy trên GPU thật; (2) **Tier-2a**: hook inline trên checkpoint thật, đo overhead tính entropy trong critical path của prefill; (3) **Tier-2b/2c**: benchmark kernel thật + end-to-end đã routing, trên GPU thật (patched scan kernel).

Phần cứng: RTX 3070 (consumer, WSL2), server 4-GPU, H800 PCIe (datacenter), cùng TPU v4-8/T4/RTX 3090 để đối chiếu. Model: Mamba-370M (chính), Mamba-1.4B, Mamba-2.8B (phạm vi hạn chế). Baseline: chunk tĩnh {32,64,128,256,512}, offline static-oracle, per-timestep không tối ưu, proxy dựa trên moment (variance, kurtosis), random scheduler, bảng tra theo sequence-length đã học.

Ở mức **kernel**: COREY (chunk=512, do entropy chọn) đạt speedup 4.41× so với static-64 trên RTX 3070, khớp chính xác oracle offline; trên datacenter 3.90–4.04×. Overhead tính entropy (Bảng 4): +8.3% nếu instrument toàn bộ 48 layer trên consumer GPU, giảm còn ≈2.1% nếu chỉ sample mỗi 4 layer; trên datacenter accelerator chỉ +2.3%. Chi phí mỗi lần gọi scheduler (cô lập): 1.10±0.16 ms.

**Kết quả âm tính (negative), trọng tâm của bài** — Bảng 5, end-to-end đã routing, prompt 976 token + 32 token sinh ra, trên H800: static-512 (oracle) = 891.51±10.17 ms (baseline nhanh nhất); full-histogram COREY = 970.26±27.36 ms (**chậm hơn 8.8%**); sampled-histogram COREY = 932.69±20.70 ms (chậm hơn 4.6%); guarded sampled-histogram = 903.03±25.35 ms (chậm hơn 1.3%); learned seq-len table = 897.63±7.40 ms (chậm hơn 0.7%). Kết luận tác giả: "no entropy-guided variant beats the best static chunk" — **không biến thể entropy-guided nào đánh bại được chunk tĩnh tốt nhất** trên workload đo được. Bảng 6-7 (mixed-regime, 160 prompt / 8 regime trên consumer GPU) còn cho thấy static-512 toàn cục (317.1ms) gần như ngang bằng per-regime oracle (316.7ms, chỉ nhanh hơn 0.14%) — tức là kể cả oracle hoàn hảo cũng không có nhiều để thắng trên workload thực tế này.

Bảng 14: 80 prompt LongBench có entropy tập trung hẹp (3.87–4.14 nats, percentile 5-95), tất cả đều rơi vào cùng một "bucket" chunk=256 — nghĩa là trong thực tế, tín hiệu entropy không đủ đa dạng để tạo khác biệt hữu ích. Overhead accounting (mục 5.2) giải thích: chi phí không nằm ở "chọn sai chunk" (decision quality cao — guarded/learned đều ra đúng chunk như oracle) mà ở **decision cost**: dựng histogram, đồng bộ hóa (synchronization boundary) quanh việc thu thập feature runtime, và dispatch overhead của scheduler — các chi phí này phải trả "even when the chosen chunk eventually matches static-512". Theorem 2 (Hadamard entropy majorization) còn bị **bác bỏ thực nghiệm**: entropy sau biến đổi Hadamard giảm ở 160/160 trường hợp thử (ΔH trung bình = −1.40±0.37 nats), trái với dự đoán lý thuyết ban đầu.

### Điểm yếu / giới hạn

Đây là bài học lớn nhất COREY để lại: **overhead của việc ước lượng runtime (entropy) có thể lớn hơn lợi ích** mà việc chọn chunk "khôn ngoan hơn" mang lại, đặc biệt khi (a) phân phối entropy của workload thực tế khá hẹp/tập trung (nên oracle cũng chẳng hơn static bao nhiêu), và (b) chi phí đồng bộ hóa + dispatch của cơ chế đo lường runtime là cố định, không giảm theo lợi ích tiềm năng. Tác giả nhấn mạnh nên coi entropy-guided routing "as a conditional mechanism rather than a universal default." Phạm vi bị giới hạn chặt ở Mamba-1.x — "Mamba-2's structured state-space duality scan has different hardware characteristics and is not addressed." Không đề cập hybrid attention+SSM hay MoE.

### Rút ra cho HyPrefill

Đây là bài học **cảnh báo phương pháp luận** quan trọng nhất cho HyPrefill: nếu HyPrefill dùng bất kỳ tín hiệu runtime nào (entropy, độ dài prompt, tải hệ thống...) để quyết định chunk size cho từng operator-group, phải đo và báo cáo **overhead của chính cơ chế quyết định đó** một cách tách bạch khỏi lợi ích thuật toán — theo đúng khung "decision quality vs. decision cost" của COREY. HyPrefill nên chủ động replicate kiểu bảng overhead (Bảng 4-5 của COREY) để chứng minh cơ chế chọn per-operator-group chunk của mình có overhead runtime thấp hơn ngưỡng nguy hiểm mà COREY gặp phải (8.8% slowdown). Vì HyPrefill nhắm vào TTFT/TBT ở mức hệ thống lớn hơn nhiều so với per-scan-call latency, rủi ro overhead có thể nhỏ hơn tương đối, nhưng cần chứng minh bằng số chứ không giả định. Khác biệt cốt lõi: COREY dùng tín hiệu **nội dung** (entropy của activation) để chọn chunk cho **một loại kernel duy nhất** (selective scan); HyPrefill dùng tín hiệu **cấu trúc + SLO** (loại operator, deadline decode còn lại) để chọn chunk cho **nhiều loại operator khác nhau trong cùng layer** — không dựa vào phân phối nội dung activation nên tránh được vấn đề "entropy quá tập trung, không đáng đo" mà COREY gặp. Reviewer từng đọc COREY chắc chắn sẽ hỏi: "HyPrefill có đo overhead của cơ chế quyết định chunk theo per-operator-group không, và liệu overhead đó có nuốt hết lợi ích TTFT không, giống như COREY?" — cần chuẩn bị số liệu overhead rõ ràng để trả lời trước.

### Hai câu hỏi tự kiểm tra

1. Vì sao decision quality cao (guarded/learned variant chọn đúng chunk như oracle) mà end-to-end vẫn chậm hơn static-512?
   *Đáp:* Vì độ trễ tăng thêm không đến từ việc chọn sai chunk, mà từ chi phí cố định của chính cơ chế đo lường/quyết định runtime (dựng histogram, đồng bộ hóa, dispatch overhead) — chi phí này vẫn phải trả dù kết quả cuối cùng trùng với chunk tĩnh tốt nhất.

2. Tại sao entropy tập trung hẹp (3.87–4.14 nats) trên LongBench lại là vấn đề cho ý tưởng entropy-guided chunking?
   *Đáp:* Vì khi entropy của hầu hết prompt thực tế rơi vào cùng một khoảng hẹp, công thức ánh xạ entropy→chunk sẽ luôn trả về cùng một chunk size (một "bucket" duy nhất, chunk=256) cho gần như mọi input — nghĩa là tín hiệu entropy không mang đủ thông tin phân biệt để biện minh cho chi phí đo lường runtime liên tục.

---

## Tổng kết liên kết ba bài cho HyPrefill

| | Đơn vị chunk | Tín hiệu điều khiển | Đảm bảo | Kiến trúc | Kết quả chính |
|---|---|---|---|---|---|
| Layered Prefill | layer-group (toàn model) | độ dài prompt (N_lg(L) tĩnh) | không có ràng buộc SLO tường minh, chỉ thực nghiệm | MoE + full attention | TTFT −56%, expert traffic −39% |
| SLOWeave | toàn bộ prefill/iteration (1 chunk) | deadline decode B_t, cost model T(n,c) | chứng minh được (Prop. 1): an toàn + tối đa | dense decoder | Goodput +39% (mixed), SLO attainment 79.4% vs 59.4% |
| COREY | selective-scan kernel call | entropy activation H̃ | không chứng minh, chỉ thực nghiệm; overhead vượt lợi ích | Mamba-1.x (SSM) | Kernel: 4.41×; End-to-end: **chậm hơn** 0.7-8.8% |

HyPrefill nằm ở giao điểm: mượn **đơn vị chunk chi tiết hơn layer** (per-operator-group, không phải per-layer như Layered Prefill), mượn **khung deadline-aware + đảm bảo toán học** từ SLOWeave (mở rộng thành vector chunk đa operator), và mượn **kỷ luật đo overhead runtime** từ bài học thất bại của COREY. Ba câu hỏi phản biện lớn nhất khi trình bày HyPrefill: (1) so với Layered Prefill, activation trung gian giữa các operator-group được quản lý ra sao; (2) so với SLOWeave, vì sao bài toán không tách được thành N instance SLOWeave độc lập; (3) so với COREY, overhead của cơ chế chọn chunk per-operator-group được đo và kiểm soát ra sao để không lặp lại kết quả âm tính.
