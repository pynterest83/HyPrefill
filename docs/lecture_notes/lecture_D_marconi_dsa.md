# Bài giảng D — Marconi và DeepSeek Sparse Attention (DSA): nền tảng cho HyPrefill

---

# Phần 1. Marconi: Prefix Caching for the Era of Hybrid LLMs
**arXiv 2411.19379 (MLSys 2025)**

## 1. Bối cảnh và vấn đề

Prefix caching (radix-tree KV cache sharing kiểu vLLM/SGLang) là kỹ thuật chuẩn để tránh tính lại phần prompt trùng lặp giữa các request (system prompt, few-shot, hội thoại nhiều lượt, agent loop). Với Transformer thuần, kỹ thuật này hoạt động tốt vì KV cache có ba tính chất thuận lợi: (i) kích thước tỉ lệ tuyến tính với số token, (ii) có thể "cắt lát" (tensor slicing) theo chiều sequence để lấy đúng KV của một tiền tố bất kỳ 1…p từ một chuỗi dài hơn 1…q, và (iii) việc cache thêm một tiền tố mới gần như không tốn thêm gì ngoài bộ nhớ.

Với các mô hình hybrid (SSM/Mamba xen kẽ Attention), state của lớp SSM phá vỡ cả ba giả định trên. Bài báo chỉ rõ ba tính chất của SSM state:
1. "SSM states are constant-sized regardless of how many tokens they represent" — state không phình theo độ dài chuỗi.
2. "SSM states are updated in place, so a sequence's states cannot be rolled back to represent its prefixes" — đây là điểm cốt lõi: vì state được cập nhật đè lên chính nó theo kiểu hồi quy, một khi đã update tới token q thì không thể "lùi" state về đúng trạng thái tại token p < q bằng phép slicing như KV cache.
3. "SSM states are orders of magnitude larger than the KVs of a single token" — một state đơn có thể lớn hơn nhiều so với KV của một token.

Hệ quả: muốn tái sử dụng một tiền tố bất kỳ, hệ thống buộc phải **chủ động checkpoint (chụp lại) state tại đúng vị trí đó trong lúc prefill**, vì không thể suy ra ngược từ state ở vị trí sau. Nếu checkpoint tại mọi vị trí block (fine-grained, giống cách radix tree làm với KV) thì gặp đúng hai vấn đề mà bài báo đo thực nghiệm:

- **Cache bị dùng lãng phí (underutilization):** với block size 32, 25.0% các block KV được request tương lai tái sử dụng, nhưng chỉ 0.4% SSM state được tái sử dụng — chênh lệch 65.3×. Nghĩa là phần lớn state được chụp lại không bao giờ được ai dùng lại.
- **Bộ nhớ nổ nhanh:** với block size 16, state của một lớp SSM lớn gấp 4× so với KV của một lớp Attention trong cùng block; với model 7B, một chuỗi 10K token tốn 17.4 GB để lưu toàn bộ state — gấp 3.3× so với một Transformer cùng quy mô.

Tóm lại: muốn tối đa hoá cơ hội tái sử dụng thì phải checkpoint mịn, nhưng checkpoint mịn lại tạo ra rất nhiều entry to mà ít khi được hit, làm cache bị "thrashing" (đầy chỗ bởi rác). Đây chính là bài toán Marconi giải quyết.

## 2. Cơ chế

Marconi quản lý KV và SSM state trong **cùng một radix tree** (không tách rời không gian cache), nhưng bổ sung hai chính sách thông minh:

**(a) Admission policy — "speculative insertion".** Thay vì checkpoint state ở mọi vị trí, trước khi prefill một sequence, Marconi **thử chèn (speculative insert)** các token đầu vào vào radix tree hiện có để xem việc chèn này có tạo ra node phân nhánh (branching point) mới hay không. Chỉ khi có phân nhánh — tức phần tiền tố đó thực sự được chia sẻ giữa nhiều request — Marconi mới chụp và cache state tại điểm đó trong lượt prefill. Bài báo phân loại hai kiểu tiền tố đáng cache: (1) tiền tố thuần input (system prompt, instruction, few-shot), và (2) tiền tố input+output (lịch sử hội thoại, agent trajectory). Cách lấy state chính xác tại điểm chia nhánh: nếu kiến trúc có "chunked state passing" trong prefill thì chỉ cần vật chất hoá state của chunk kề áp chót; nếu không, Marconi làm **hai lượt prefill** để lấy đúng state tại vị trí tiền tố.

**(b) Eviction policy — FLOP-aware, thay cho LRU thuần.** Vấn đề với LRU: với Attention, kích thước KV tỉ lệ với độ dài chuỗi và do đó là proxy tốt cho lượng compute tiết kiệm được khi hit; nhưng SSM state có kích thước cố định bất kể tiết kiệm được bao nhiêu compute. Marconi định nghĩa:

```
flop_efficiency(n) = (tổng FLOPs được cứu qua các layer tại node n) / (bộ nhớ mà state ở node n chiếm dụng)
```

và điểm ưu tiên giữ lại một node n trong cache:

```
S(n) = recency(n) + α · flop_efficiency(n)
```

trong đó cả hai đại lượng được chuẩn hoá về (0,1). Hệ số α được **tự động hiệu chỉnh**: khi khởi động, α = 0 (thuần LRU) cho tới lần evict đầu tiên; sau một giai đoạn "bootstrap" (5–15× số request trước lần evict đầu), Marconi chạy grid-search bất đồng bộ để chọn α tốt nhất dựa trên việc replay các request đã quan sát. Ngoài ra, khi tính ứng viên để evict, Marconi xét cả node có ≤1 con (không chỉ node lá) vì node có nhiều con đại diện cho tiền tố dùng chung — không nên evict; và khi cache hit, chỉ cập nhật timestamp của đúng node được truy cập (không lan truyền lên toàn bộ tổ tiên như hệ thống khác).

## 3. Kết quả số

Thực nghiệm trên mô hình hybrid 7B ({4,24,28} layer Attention/SSM/MLP) và Jamba-1.5-Mini (12B active/52B total, state dim 128, 4× A100-40GB), trên ba trace thực: LMSys (hội thoại đa lượt, output dài), ShareGPT (hội thoại đa lượt, output ngắn), SWE-Bench (agentic, SWE-Agent trên GitHub issue thật). So với baseline vLLM+ (fine-grained checkpoint mọi block, block size 32) và SGLang+ (radix tree + LRU thuần, không FLOP-aware):

- Token hit rate trung bình cao hơn vLLM+ **4.5×, 7.3×, 34.4×** trên LMSys/ShareGPT/SWE-Bench tương ứng.
- P95 TTFT giảm **36.1%, 71.1%, 46.8%** (tương đương 275.4 ms, 103.3 ms, 617.0 ms).
- So với SGLang+ (đo riêng lợi ích của eviction FLOP-aware), hit rate cải thiện **19.0–219.7%**, rõ nhất trên SWE-Bench (P95: +219.7%; hit rate 16.4% → 32.7%, tức +99.4%; FLOP saved +90.3%).
- Phân tích theo độ dài chuỗi trên SWE-Bench: với chuỗi < 7K token, Marconi thấp hơn SGLang+ tới 3.0% hit rate, nhưng với chuỗi > 7K token thì vượt lên tới +25.5% — đúng như thiết kế: FLOP-aware eviction đánh đổi hit rate của chuỗi ngắn lấy hit rate của chuỗi dài (vì chuỗi dài "đáng" giữ hơn về compute).
- Ablation: thay đổi tỉ lệ Attention:SSM 1:2/1:4/1:8, thay đổi state dimension 16→128 (hit rate improvement so vLLM+ tăng từ 5.7× lên 35.4×), thay đổi kích thước cache (60–140GB, cải thiện 10–68.3% tuỳ mức độ tranh chấp — lợi ích rõ nhất khi cache ở mức tranh chấp vừa phải), thay đổi tốc độ đến của request.

## 4. Điểm yếu / giới hạn

Nhóm tác giả tự thừa nhận một số đánh đổi: (i) vì phải đợi lần xuất hiện thứ hai của một tiền tố mới xác nhận có phân nhánh, Marconi **bỏ lỡ cơ hội tái sử dụng ngay ở lần xuất hiện thứ hai** của một tiền tố "thuần input" (dùng chính lần đó để checkpoint) — tuy tác giả cho là ảnh hưởng không đáng kể vì các tiền tố này thường được chia sẻ bởi rất nhiều request sau đó; (ii) admission có chọn lọc làm giảm độ phủ, chỉ tối đa hai SSM state được admit mỗi sequence, nên giới hạn khả năng tái sử dụng tiền tố tuỳ ý; (iii) chính sách FLOP-aware chủ động đánh đổi hit rate của chuỗi ngắn để lấy hit rate chuỗi dài (thấy rõ ở mức -3.0% cho chuỗi <7K), dù tác giả cho rằng độ suy giảm latency tuyệt đối là nhỏ (~2.1 ms). Bài báo không có mục "future work" tường minh, và phạm vi thực nghiệm giới hạn ở một họ kiến trúc hybrid Attention+Mamba/Mamba2, chưa kiểm chứng trên các biến thể hybrid khác (dù tác giả khẳng định phương pháp tổng quát hoá được cho "all Hybrid models with recurrent layers").

## 5. Rút ra cho HyPrefill

**Cách đóng khung một systems paper về hybrid serving:** Marconi đi theo mạch: (1) định lượng thực nghiệm một pathology cụ thể của thành phần mới (bảng 65.3× lệch giữa reuse rate của KV và SSM state, và hệ số 3.3–4× phình bộ nhớ) → (2) từ đó suy ra *tại sao* chính sách "một-cỡ-cho-tất-cả" (ở đây là "cache mọi thứ giống nhau") thất bại → (3) thiết kế cơ chế tách biệt theo *loại toán tử* (ở đây: admission khác nhau cho SSM vs Attention, eviction có trọng số theo FLOP/byte đặc thù từng loại layer) → (4) cài đặt trên hệ thống serving thật (mở rộng vLLM/SGLang) → (5) đánh giá bằng baseline mạnh (không so với baseline ngây thơ) trên trace thực. HyPrefill nên áp dụng đúng mạch này: định lượng trước bằng số liệu tại sao chunk size đồng nhất giữa full-attention/GDN/MoE là dưới tối ưu, rồi mới đề xuất chunk size khác nhau theo nhóm toán tử.

**Phương pháp luận đánh giá đáng học theo:** (a) dùng trace thực đa dạng về hình thái (hội thoại dài/ngắn, agentic) thay vì chỉ benchmark tổng hợp; (b) đo bằng metric phục vụ thực tế (TTFT theo percentile P5/P50/P95, hit rate, throughput) chứ không chỉ FLOPs lý thuyết; (c) luôn **phân tích theo bucket** (ở đây theo độ dài chuỗi) vì trung bình có thể che giấu hiệu ứng hai chiều (Marconi thua trên chuỗi ngắn nhưng thắng đậm trên chuỗi dài); (d) ablation đa trục: tỉ lệ layer, dimension trạng thái, áp lực bộ nhớ cache, tốc độ đến của request — để chứng minh cơ chế robust chứ không chỉ hợp với một cấu hình. HyPrefill nên tái sử dụng chính khung ablation này (thay "tỉ lệ Attention:SSM" bằng "tỉ lệ full-attn:GDN:MoE layer", thay "state dimension" bằng "chunk size mỗi nhóm toán tử").

## 6. Câu hỏi tự kiểm tra

**Q1.** Tại sao LRU thuần không đủ để làm eviction policy cho cache hybrid, và Marconi thay bằng gì?
*Đáp:* Vì kích thước KV tỉ lệ với độ dài chuỗi (proxy tốt cho compute tiết kiệm được khi hit) nhưng SSM state có kích thước cố định bất kể tiết kiệm bao nhiêu compute, nên "mới nhất" không phản ánh đúng giá trị giữ lại. Marconi dùng điểm số kết hợp S(n) = recency(n) + α·flop_efficiency(n), với flop_efficiency = FLOP cứu được / byte bộ nhớ chiếm dụng, và α được tự động hiệu chỉnh qua grid-search sau giai đoạn bootstrap.

**Q2.** "Speculative insertion" là gì và tại sao cần thiết?
*Đáp:* Trước khi prefill, Marconi thử chèn (không thật) chuỗi input vào radix tree để kiểm tra xem có tạo node phân nhánh mới (nghĩa là tiền tố này thực sự dùng chung với request khác) hay không; chỉ khi có phân nhánh mới thực sự checkpoint SSM state tại điểm đó trong lúc prefill thật. Cơ chế này cần thiết vì SSM state rất lớn và tỉ lệ hit thực tế ở mức chi tiết cực thấp (0.4%) — nếu checkpoint mọi vị trí sẽ làm cache đầy rác vô ích.

---

# Phần 2. DeepSeek-V3.2 — DeepSeek Sparse Attention (DSA), "lightning indexer"
**arXiv 2512.02556**

## 1. Bối cảnh và vấn đề

DeepSeek-V3.2-Exp được phát triển từ DeepSeek-V3.1-Terminus bằng cách thay cơ chế attention dày đặc (dense, dùng MLA) bằng DeepSeek Sparse Attention (DSA): mỗi query token chỉ attend tới một tập con k token được chọn động, thay vì toàn bộ t token trước đó. Mục tiêu là giảm chi phí huấn luyện và suy luận dài ngữ cảnh, vì attention dense có độ phức tạp O(L²) theo độ dài chuỗi L — với 128K context điều này trở nên rất đắt cả về compute lẫn KV-cache. Vấn đề kỹ thuật cốt lõi là: **làm sao biết token nào đáng attend tới** mà không phải tính toàn bộ attention dày đặc trước (nếu phải tính dense trước để biết chọn ai thì sparse hoá vô nghĩa). DSA giải quyết bằng một "lightning indexer" — một mô-đun rẻ, riêng biệt, có nhiệm vụ chấm điểm mức độ liên quan giữa query hiện tại và mọi key quá khứ, để chọn ra top-k trước khi tính attention thật.

## 2. Cơ chế

**Công thức chỉ số (index score):**

```
I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
```

trong đó H^I là số "đầu" (head) của indexer, q^I_{t,j} ∈ ℝ^{d^I} là vector query của indexer ứng với head j tại vị trí truy vấn t, k^I_s ∈ ℝ^{d^I} là vector key của indexer tại vị trí quá khứ s, và w^I_{t,j} ∈ ℝ là trọng số đầu học được, phụ thuộc token truy vấn. ReLU được chọn (thay vì softmax) vì hiệu quả throughput. Indexer có **số đầu nhỏ và được cài đặt ở FP8** để tối đa hoá tốc độ — bài báo không công bố tường minh giá trị chính xác của H^I và d^I trong phần văn bản trích xuất được (cả từ bản HTML lẫn PDF); đây là điểm cần tra thêm trong config chính thức (config.json / inference code) nếu cần con số cụ thể cho HyPrefill.

**Chọn top-k và tính attention thật:** với mỗi query t, chọn ra k = **2048** vị trí s có I_{t,s} lớn nhất, rồi chỉ tính attention thật trên tập này:

```
u_t = Attn(h_t, { c_s | I_{t,s} ∈ Top-k(I_{t,:}) })
```

Nhờ đó, độ phức tạp attention chính giảm từ O(L²) xuống **O(L·k)** với k ≪ L. Tuy nhiên bài báo nói rõ: "the lightning indexer still has a complexity of O(L²)" — vì để biết top-k, vẫn phải tính điểm số I_{t,s} cho **mọi cặp (t,s)**, chỉ có điều phép tính này rẻ hơn nhiều (ít đầu, chiều nhỏ, FP8) so với attention chính (MLA, chiều lớn hơn, nhiều đầu hơn, độ chính xác cao hơn).

**Suy ra công thức chi phí theo chunk (áp dụng cho chunked prefill):** xét một chunk có c token truy vấn mới, xử lý tại vị trí ngữ cảnh hiện tại t (tức trước chunk này đã có t token đã prefill):
- Chi phí indexer cho chunk này: mỗi trong c query phải tính điểm với toàn bộ t key trước đó, qua H^I đầu, mỗi phép dot-product tốn O(d^I):
```
cost_indexer(c, t) = O(c · t · H^I · d^I)
```
- Chi phí attention thưa (sau khi đã chọn top-k) cho chunk này: mỗi trong c query chỉ attend k=2048 entry, với chiều biểu diễn d (của MLA):
```
cost_sparse(c) = O(c · k · d)
```
Điểm mấu chốt: `cost_indexer` **tăng tuyến tính theo t** (vị trí hiện tại trong chuỗi) ngoài việc tăng theo c, trong khi `cost_sparse` **không phụ thuộc t** — chỉ phụ thuộc c (và hằng số k, d cố định). Cộng dồn qua toàn bộ L/c chunk của một chuỗi độ dài L (t chạy từ 0 đến L), tổng chi phí indexer là O(L²·H^I·d^I) — khớp với O(L²) mà bài báo nêu cho toàn chuỗi — còn tổng chi phí attention thưa là O(L·k·d) — khớp O(Lk).

**Bộ nhớ:** để chọn top-k, indexer phải giữ toàn bộ ma trận điểm số I_{t,:} của chunk hiện tại trước khi rút gọn còn k, tức một buffer kích thước **c · t** phần tử (số hàng = c query trong chunk, số cột = t key trong ngữ cảnh đã có) cho mỗi layer, thường ở FP32. Đây là chi phí bộ nhớ *tạm thời* nhưng **tăng theo t** dù c không đổi — nghĩa là càng về sau trong một prefill dài, buffer của mỗi chunk càng phình to, dù bản thân attention thưa sau top-k lại rất nhỏ gọn (chỉ O(c·k)).

**Huấn luyện indexer — hai giai đoạn:**
1. *Dense warm-up:* giữ nguyên attention dense, đóng băng toàn bộ tham số mô hình trừ indexer; loss là KL-divergence giữa phân phối attention thật (gộp qua các đầu, chuẩn hoá L1) và softmax của điểm indexer: `ℒ_I = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:}))`. Cấu hình: learning rate 10⁻³, 1000 bước, 16 chuỗi × 128K token/bước ⇒ tổng 2.1B token.
2. *Sparse training:* tiếp tục tối ưu indexer nhưng chỉ so khớp trên tập token đã được chọn: `ℒ_I = Σ_t D_KL(p_{t,S_t} ‖ Softmax(I_{t,S_t}))`, và **tách indexer khỏi đồ thị tính toán chính** để tối ưu riêng. Cấu hình: learning rate 7.3×10⁻⁶, 15.000 bước, 480 chuỗi × 128K token/bước ⇒ tổng 943.7B token, mỗi query vẫn chọn 2048 key-value token.

## 3. Kết quả số

Bài báo xác nhận DeepSeek-V3.2-Exp **không suy giảm hiệu năng đáng kể** so với DeepSeek-V3.1-Terminus trên cả tác vụ ngắn và dài ngữ cảnh; đánh giá sở thích người dùng qua ChatbotArena cho Elo score gần như tương đương (tính đến 10/11/2025). Trên các benchmark dài ngữ cảnh, DSA thậm chí nhỉnh hơn: AA-LCR cao hơn 4 điểm ở chế độ reasoning; Fiction.liveBench vượt trội nhất quán qua nhiều metric. Về chi phí suy luận, Hình 3 của bài báo minh hoạ "token cost" (ước tính từ việc benchmark dịch vụ thật trên cụm H800, đơn giá thuê GPU 2 USD/giờ) của V3.1-Terminus so với V3.2 theo vị trí token trong chuỗi, cho cả pha prefill và decode; văn bản mô tả DSA đạt "significant end-to-end speedup in long-context scenarios" nhưng **không nêu con số USD/triệu-token cụ thể tại 32K hay 128K trong phần văn bản trích xuất được** (giá trị nằm trong hình vẽ, không được số hoá trong bản HTML/PDF chuyển đổi) — đây là điểm cần lấy trực tiếp từ hình gốc hoặc mã nguồn benchmark nếu cần trích dẫn số chính xác. Bài báo cũng ghi nhận với ngữ cảnh 128K, "20%+ of the test cases exceed this limit", cho thấy việc mở rộng hỗ trợ ngữ cảnh dài hơn vẫn đang là vấn đề thực tế.

## 4. Điểm yếu / giới hạn

- Indexer vẫn là O(L²) toàn cục — DSA không loại bỏ chi phí bậc hai, chỉ **dịch chuyển** nó sang một phép toán rẻ hơn (ít đầu, FP8, ReLU thay softmax). Với ngữ cảnh cực dài, chi phí này vẫn có thể trở thành nút thắt, đặc biệt về **bộ nhớ tạm** (buffer điểm số kích thước c·t mỗi layer) chứ không chỉ compute.
- Bài báo không công bố tường minh H^I, d^I và không định lượng bằng số liệu cụ thể (USD, ms) mức tiết kiệm tại các mốc ngữ cảnh cụ thể trong phần văn bản — thông tin định lượng nằm trong hình vẽ không trích xuất được qua HTML/PDF-to-text.
- Việc phải "tách indexer khỏi đồ thị tính toán chính" trong giai đoạn sparse training cho thấy việc co-train indexer với model chính không ổn định/không hiệu quả nếu để gradient lan thẳng — một ràng buộc kỹ thuật khi mở rộng ý tưởng sang kiến trúc khác.
- Ở chế độ short-context, cài đặt dùng "masked MHA để giả lập DSA" (theo mô tả trong kết quả fetch), nghĩa là lợi ích thực sự của sparse hoá chỉ rõ rệt ở long-context — với short-context có thể không có lợi (thậm chí thêm overhead) do chưa đạt ngưỡng để bù chi phí indexer.

## 5. Rút ra cho HyPrefill

DSA cho thấy một layer "hybrid" mới — sparse attention với indexer — thực ra có **hai chế độ chi phí khác hẳn nhau về hình dạng (slope) theo t**, và đây chính là chỗ per-operator-group chunk sizing của HyPrefill nên khai thác:

- **Slope của indexer tăng tuyến tính theo t** (cost_indexer(c,t) = O(c·t·H^I·d^I)): giống hệt full-attention dense — chi phí một chunk phụ thuộc vào toàn bộ ngữ cảnh đã tích luỹ, không phải hằng số.
- **Slope của phần attention-thưa-thật-sự gần như hằng số theo t** (cost_sparse(c) = O(c·k·d)): giống GDN/SSM ở chỗ chi phí mỗi chunk không phình theo ngữ cảnh.

Điều này có nghĩa là trong một layer DSA, **hai sub-block cần hai chính sách chunk-size khác nhau**: phần indexer nên được coi như một "nhóm toán tử" ứng xử giống full-attention (chunk nhỏ dần khi t lớn để giữ chi phí/latency mỗi chunk ổn định), còn phần attention-sau-top-k có thể dùng chunk lớn hơn vì chi phí không phụ thuộc t.

Quan trọng hơn, đây **không chỉ là vấn đề compute mà là vấn đề bộ nhớ**: issue vLLM #56457 (về indexer sparse attention, tên trong issue là "QSA/DSA indexer") ghi nhận trực tiếp hiện tượng logits buffer của indexer **tăng theo chunk index i trong lúc chunked prefill**: "chunk i of a long prompt now allocates 3200 × (800·i) × 4B = 10.24 MB × i, a strictly increasing size per chunk" — khớp chính xác với công thức bộ nhớ **c·t mỗi layer** suy ra ở trên (chunk sau có t lớn hơn ⇒ buffer to hơn tuyến tính theo chỉ số chunk). Hệ quả thực tế: bộ cấp phát bộ nhớ CUDA không tái sử dụng được block cũ (vì kích thước liên tục tăng), gây `cudaMalloc` liên tục và tích luỹ tới ~14GB/rank, dẫn tới OOM ở ngữ cảnh dài (254K token, treo tại 166.400 token). Cách khắc phục trong issue là giới hạn cứng kích thước buffer (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, hạ từ 512 xuống 64) hoặc làm tròn kích thước buffer theo bucket cố định để bộ cấp phát tái sử dụng được.

Bài học cụ thể cho HyPrefill: (1) khi thiết kế chunk size cho nhóm toán tử "indexer" trong các hybrid sparse-attention mới (không chỉ DSA mà cả các sparse-attention hybrid tương lai), phải coi bộ nhớ logits-buffer là ràng buộc **memory-bound theo c·t**, không chỉ compute-bound theo c; (2) chunk size hợp lý cho nhóm indexer nên là hàm giảm dần theo t (t càng lớn, c càng nhỏ) để giữ c·t ≤ ngân sách bộ nhớ cố định — tương tự việc đặt trần cho buffer thay vì để nó tăng tuyến tính vô hạn; (3) nên tách rời chính sách chunk-size của "indexer group" khỏi "sparse-attention-compute group" và khỏi "GDN/MoE group" vì ba nhóm có hình dạng chi phí (slope theo t) khác nhau hoàn toàn — đây chính là luận điểm trung tâm mà HyPrefill cần chứng minh bằng số liệu, tương tự cách Marconi chứng minh SSM state và KV cache cần chính sách khác nhau.

## 6. Câu hỏi tự kiểm tra

**Q1.** Vì sao DSA giảm được chi phí attention chính từ O(L²) xuống O(Lk) nhưng bản thân indexer vẫn giữ nguyên O(L²)?
*Đáp:* Để biết k token nào đáng attend (top-k theo I_{t,s}), hệ thống bắt buộc phải tính điểm số cho **mọi** cặp (t,s), tức vẫn là một phép tính toàn-cặp O(L²); chỉ có điều phép tính này (ReLU của dot-product, số đầu H^I nhỏ, chiều d^I nhỏ, FP8) rẻ hơn nhiều so với attention chính (MLA, chiều lớn, độ chính xác cao). Sau khi đã có top-k, attention thật chỉ cần tính trên k entry mỗi query, cho O(L·k) với k ≪ L.

**Q2.** Tại sao buffer điểm số của indexer trong chunked prefill lại là một nút thắt bộ nhớ, và độ lớn của nó phụ thuộc vào những gì?
*Đáp:* Trước khi rút gọn còn top-k, indexer phải giữ toàn bộ ma trận điểm số I_{t,s} của chunk hiện tại, kích thước ~ c (số query trong chunk) × t (số key/ngữ cảnh tích luỹ tới thời điểm đó), mỗi layer. Vì t tăng dần qua các chunk trong một prefill dài, buffer của chunk sau luôn lớn hơn chunk trước dù c không đổi — được xác nhận thực tế trong vLLM issue #56457 (buffer chunk thứ i tăng tuyến tính theo i, ~10.24MB·i), khiến bộ cấp phát bộ nhớ không tái sử dụng được block cũ và dẫn tới OOM ở ngữ cảnh dài nếu không giới hạn cứng kích thước buffer.
