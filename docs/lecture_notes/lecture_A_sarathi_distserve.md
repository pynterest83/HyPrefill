# Giáo án đọc bài: Sarathi-Serve và DistServe (nền tảng cho HyPrefill)

---

## 1. Sarathi-Serve (arXiv 2403.02310, OSDI 2024)

### Bối cảnh và vấn đề

LLM inference có hai pha với đặc tính hoàn toàn khác nhau: **prefill** (xử lý toàn bộ prompt, compute-bound, bão hòa GPU chỉ với khoảng 512 token) và **decode** (sinh từng token, memory-bound, utilization thấp nếu không batch). Batching rất hiệu quả cho decode và cho throughput tổng thể, nhưng khi nhiều request được batch cùng nhau, các iteration prefill và decode bị **interleave**, gây khó khăn để đạt đồng thời throughput cao và latency thấp.

Vấn đề cụ thể mà paper gọi tên là **generation stalls**: vì prefill có thể mất thời gian tùy ý theo độ dài prompt, một scheduler ưu tiên prefill (prefill-prioritizing, như vLLM) sẽ chèn một prefill dài vào giữa các bước decode đang chạy, làm "đóng băng" việc sinh token của các request khác trong một khoảng thời gian dài. Paper minh họa: "Figure 1(a) highlights one of the many generation stalls lasting over several seconds in vLLM." Một ví dụ số cụ thể với Falcon-180B: một prompt 4k token cần ≈1150ms để chạy, trong khi một iteration decode-only với batch size 32 chỉ mất ≈200ms — nếu interleave, bubble sinh ra có thể lên tới ≈950ms. Ngược lại, các hệ thống ưu tiên decode (decode-prioritizing, kiểu FasterTransformer) không nhận request mới cho tới khi toàn bộ batch decode hiện tại hoàn tất, gây throughput thấp và TTFT cao cho request mới. Đây chính là thế lưỡng nan throughput-vs-latency mà Sarathi-Serve muốn giải quyết.

### Cơ chế

Sarathi-Serve kết hợp hai kỹ thuật:

**(1) Chunked-prefills**: "splits a prefill request into near equal sized chunks" thay vì chạy toàn bộ prompt trong một iteration. Cơ sở: một prefill với độ dài vừa phải (~512 token) đã bão hòa compute GPU, trong khi prompt thực tế dài hơn nhiều (median 1730–7059 token trong hai dataset đánh giá). Vì vậy có thể tách prefill dài thành các đơn vị tính toán nhỏ hơn mà vẫn giữ GPU bão hòa. Lưu ý: attention của mỗi chunk phải đọc lại KV-cache của các chunk trước đó, nhưng vì attention prefill vẫn compute-bound ngay cả ở chunk nhỏ nên overhead này ở mức chấp nhận được.

**(2) Stall-free scheduling (hybrid batching)**: Sarathi-Serve trước tiên tính "the budget of maximum number of tokens that can be executed in a batch based on user specified SLO" — **token budget**. Thuật toán scheduler (Algorithm 3) theo thứ tự: (a) đóng gói tất cả token decode đang chạy vào batch tiếp theo trước tiên; (b) thêm phần prefill dở dang của request đang được chunk; (c) nhận request mới trong phần token budget còn lại. Vì decode luôn được ưu tiên đưa vào batch trước, hệ thống "leverages the arithmetic intensity slack in decode iterations to execute prefills without delaying execution" — tức tận dụng phần compute "rảnh" trong decode (decode memory-bound, chưa dùng hết compute) để nhét prefill chunk vào mà không làm decode bị trễ ("decodes never experience a generation stall due to a co-running prefill chunk").

**Xác định token budget**: paper không đưa công thức toán tường minh mà mô tả trade-off — budget nhỏ tốt cho TBT (ít token prefill hơn mỗi iteration → latency thấp hơn), budget lớn giảm overhead chunking. Cách làm thực tế: "one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO." Một chi tiết tinh vi: hiệu ứng **tile-quantization** của GPU — matmul đạt hiệu suất tối đa khi kích thước chia hết cho tile size; ví dụ "using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256." Với pipeline parallelism, chunk size không đều giữa các microbatch còn gây pipeline bubble.

**Định nghĩa chính xác** (Section 2.4): TTFT "measures the latency of generating the first output token from the moment a request arrives in the system"; TBT "measures the interval between the generation of consecutive output tokens of a request, and affects the overall perceived fluidity of the response." SLO P99 TBT được đặt bằng 5× (strict) hoặc 25× (relaxed) thời gian một iteration decode thuần.

### Kết quả số

Đánh giá trên 4 model/cấu hình: Mistral-7B (1×A100), Yi-34B (2×A100, TP2), LLaMA2-70B (8×A40, TP4-PP2), Falcon-180B (8×A100 trên 2 node qua Ethernet thường, TP4-PP2). Dưới SLO nghiêm ngặt, capacity tăng so với vLLM: Mistral-7B 2.6×, Yi-34B 3.7×, LLaMA2-70B 4.3×, Falcon-180B 3.6× (và tới 5.6× so với baseline nhờ loại bỏ pipeline bubble). So với Orca: Yi-34B 4.0×, LLaMA2-70B 6.3×.

Bảng ablation (Yi-34B, dataset openchat_sharegpt4) rất đáng chú ý: hybrid-batching-only cho P50 TTFT 0.53s / P99 TBT 0.68s; chunked-prefill-only cho 1.04s / 0.17s; kết hợp cả hai (Sarathi-Serve) cho 0.76s / 0.14s — chứng minh cần cả hai kỹ thuật cùng lúc để tối ưu đồng thời TTFT và TBT. Overhead của chunking: ở chunk size 512, overhead tối đa ~25%; ở chunk size 2048 cho sequence dài thì gần như không đáng kể.

### Điểm yếu / giới hạn

Token budget/chunk size là một siêu tham số **toàn cục, cố định**, phải tune qua profiling offline cho từng cặp model+hardware, không thích nghi động theo shift của workload. Việc chọn sai chunk size (do tile-quantization) có thể gây phạt hiệu năng khá lớn và khó đoán (32% chỉ vì lệch 1 token). Bản thân paper cũng thừa nhận riêng lẻ mỗi kỹ thuật đều không đủ: chunked-prefill-only làm tăng TTFT ("prefill chunks are slightly inefficient"), hybrid-batching-only làm tăng TBT ("long prefills can still create generation stalls"). Ngoài ra, mô hình chi phí ngầm định là attention Transformer đồng nhất qua các layer; paper không xét kiến trúc hybrid có linear-attention/SSM hay MoE, nơi chi phí theo chunk size có thể khác hẳn về hình dạng (không chỉ là compute-bound tuyến tính).

### Rút ra cho HyPrefill

- **Baseline**: Sarathi-Serve (chunk size/token budget toàn cục, đồng nhất mọi layer) là baseline chính cần đánh bại.
- **Metric**: cặp TTFT + P99 TBT dưới một SLO cố định, cùng phương pháp ablation tách riêng đóng góp từng kỹ thuật.
- **Mượn lập luận**: suy token budget bằng profiling offline theo TBT SLO — HyPrefill mở rộng thành **profiling theo từng operator-group** (full attention, linear attention/Gated DeltaNet, MoE), vì mỗi loại có cost-vs-chunk-size curve khác nhau (attention: chi phí đọc KV-cache tăng theo context length; linear attention/SSM: state cố định kích thước, chi phí gần tuyến tính/hằng số; MoE: phụ thuộc routing/tải expert).
- **Khác biệt cốt lõi**: Sarathi chọn MỘT token budget cho toàn model do giả định chi phí per-layer đồng nhất. HyPrefill phải chọn một **vector kích thước chunk theo từng nhóm toán tử**, và giải bài toán ghép nối joint vì các nhóm chạy trên cùng tập token trong một iteration — vẫn giữ ràng buộc P99 TBT toàn cục nhưng giảm TTFT tốt hơn so với một chunk size chung.

### Câu hỏi tự kiểm tra

**Q1:** Vì sao chunk size 257 lại chậm hơn đáng kể so với 256 dù chỉ chênh 1 token?
**Đáp:** Do hiệu ứng tile-quantization của GPU matmul — kích thước không chia hết cho tile size buộc GPU phải xử lý thêm nguyên một tile dư thừa, khiến thời gian prefill tăng ~32%.

**Q2:** Vì sao chỉ dùng riêng chunked-prefill hoặc riêng hybrid-batching đều không đạt cả TTFT thấp và TBT thấp?
**Đáp:** Hybrid-batching-only vẫn chạy trọn một prefill dài nên có thể gây stall cho decode (TBT cao, 0.68s); chunked-prefill-only không đảm bảo decode luôn được ưu tiên vào batch trước nên request mới phải chờ xen giữa các chunk prefill khác (TTFT cao, 1.04s). Kết hợp cả hai cho kết quả cân bằng nhất (0.76s / 0.14s).

---

## 2. DistServe (arXiv 2401.09670, OSDI 2024)

### Bối cảnh và vấn đề

DistServe xuất phát từ quan sát: khi prefill và decode **colocate** trên cùng GPU/batch (như vLLM, Orca), chúng gây interference lẫn nhau nghiêm trọng. Số liệu minh họa (Figure 1, model 13B, 1 GPU): hệ colocate hiện tại chỉ đạt **~1.6 request/s (rps)** goodput tối đa ở ngưỡng 90% SLO attainment; trong khi nếu chạy riêng, baseline prefill-only đạt 5.6 rps/GPU, baseline decode-only đạt 10 rps/GPU; một hệ disaggregate lý thuyết (2 GPU prefill + 1 GPU decode) đạt trung bình **3.3 rps/GPU — cao hơn 2.1× hệ colocate hiện tại**.

Nguyên nhân gốc rễ (Figure 2): "adding a single prefill job to a batch of decoding requests significantly slows down both processes, leading to a marked increase in TTFT and TPOT." Cụ thể: "A prefill step often takes much longer than a decoding step. When batched together, decoding steps in the batch are delayed by the prefill steps, significantly elongating their TPOT"; ngược lại thêm decode vào batch prefill cũng làm tăng TTFT đáng kể. Paper còn chỉ ra giải pháp né tránh trước đó — chunked-prefill kèm piggyback decode (kiểu Sarathi) — chỉ **giảm bớt chứ không loại bỏ** interference: "This technique alleviates the slowdown of the decoding job caused by the long prefill job, but it does not eliminate it," đồng thời việc chia N chunk làm tăng chi phí đọc lại KV-cache lên O(N²) thay vì O(N) nếu không chunk.

### Cơ chế

Giải pháp: **tách vật lý** prefill và decode ra hai pool GPU riêng, mỗi pool tự do chọn cấu hình song song (tensor/pipeline parallelism) tối ưu riêng cho đặc tính pha của nó, loại bỏ hoàn toàn interference; kết nối hai pool qua truyền KV-cache trên mạng.

**Goodput**: định nghĩa là "the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned" — thước đo throughput đã chuẩn hóa theo ràng buộc SLO và số GPU, phản ánh đúng chi phí-hiệu quả thay vì throughput thô.

**Thuật toán đặt cấu hình (placement)** có hai biến thể theo băng thông cụm máy: (1) **High node-affinity** (Algorithm 1, băng thông liên-node tốt): liệt kê mọi cấu hình song song khả thi cho instance prefill và instance decode riêng biệt, với mỗi cặp cấu hình dùng simulator + binary search trên request rate để tìm goodput tối đa thỏa SLO, rồi nhân bản instance để đạt traffic mục tiêu; độ phức tạp O(N·M²) với N là số node tối đa/instance, M là số GPU/node. (2) **Low node-affinity** (Algorithm 2, băng thông liên-node hạn chế): ràng buộc prefill và decode phải colocate cùng node để tận dụng NVLink băng thông cao, chỉ liệt kê cấu hình song song trong-node rồi mô phỏng tương tự. Thời gian chạy thuật toán dưới 1.3 phút cho cấu hình lớn nhất; sai số simulator so với hệ thực <2%.

**Truyền KV-cache**: theo mô hình **"pull"** thay vì "push" — "decoding instances fetch KV cache from prefill instances as needed, using the GPU memory of prefill instances as a queuing buffer," giúp tránh quá tải khi traffic bùng nổ đột ngột. **Cost model** ví dụ cụ thể: OPT-66B, request 512 token, arrival rate 10 rps → cần truyền 11.3GB/s ≈ 90 Gbps. Với NVLink trong-node (đỉnh 600GB/s giữa 2 A100) hoặc InfiniBand liên-node (800Gbps), hơn 95% request có độ trễ truyền <30ms, đóng góp <0.1% tổng latency với OPT-175B.

### Kết quả số

Testbed: 4 node × 8 GPU A100-80GB SXM (32 GPU), NVLink trong-node, băng thông liên-node chỉ 25Gbps (cố tình chọn cấu hình băng thông thấp để kiểm tra cost model). Model: OPT-13B/66B/175B (FP16). SLO theo ứng dụng: Chatbot (ShareGPT) OPT-13B TTFT 0.25s/TPOT 0.1s, OPT-66B 2.5s/0.15s, OPT-175B 4.0s/0.2s; Code completion (HumanEval) OPT-66B 0.125s/0.2s; Summarization (LongBench) OPT-66B 15s/0.15s. Baseline: vLLM, DeepSpeed-MII, vLLM++ (vLLM với parallelism tối ưu, hiệu năng tương đương vLLM gốc).

Ở ngưỡng 90% SLO attainment: Chatbot OPT-13B tăng 2.0–4.6× request rate so với vLLM; OPT-175B tăng 1.6–7.4× so với DeepSpeed-MII (khớp con số headline "7.4× more requests" trong abstract); Code completion OPT-66B tăng 5.7× so với vLLM, 1.6× so với DeepSpeed-MII; Summarization tăng 4.3× so với vLLM, 1.8× so với DeepSpeed-MII. Về độ chặt SLO (giữ nguyên rate/attainment, siết SLO): Chatbot chặt hơn 1.8–3.2× so với vLLM; Summarization chặt hơn tới **12.6×** so với vLLM (khớp con số headline thứ hai). Ablation: vLLM++ (tối ưu song song) không cải thiện gì so với vLLM gốc — chứng tỏ vấn đề không nằm ở cấu hình song song mà ở chính việc colocate.

### Điểm yếu / giới hạn

Tác giả tự thừa nhận: với ứng dụng offline không nhạy latency, "techniques such as chunked-prefill... may be preferred" hơn là disaggregation. Trong môi trường ít GPU (thậm chí 1 GPU), "the design space for DistServe is significantly limited... simpler architectural choices like non-disaggregated systems may reduce deployment complexity." Prototype chưa cài đặt preemption và fault tolerance. Băng thông mạng cần được cấp phát/đảm bảo (NVLink hoặc InfiniBand) — trong cụm băng thông thấp hơn giả định, KV transfer có thể thành nút thắt cổ chai, buộc phải dùng biến thể Algorithm 2 (colocate trong-node) làm giảm bớt tính linh hoạt của disaggregation. Ngoài ra, việc tìm cấu hình đặt (placement) là **tìm kiếm offline dựa trên simulator cho một workload/rate giả định trước**, chưa rõ khả năng thích nghi khi tỷ lệ prefill:decode hoặc phân bố độ dài input thay đổi đột ngột trong thực tế — đây là điểm một reviewer chắc chắn sẽ chất vấn.

### Rút ra cho HyPrefill

- **Phân biệt hai trường phái**: DistServe tách vật lý prefill/decode (disaggregation); dòng Sarathi/HyPrefill giữ chung một engine, điều phối bằng chunk size (co-location có kiểm soát). HyPrefill kế thừa dòng thứ hai nhưng phải đối mặt lập luận cốt lõi của DistServe: colocate gây interference — với model hybrid, interference này xảy ra **giữa nhiều nhóm toán tử** (attention/SSM/MoE), không chỉ giữa prefill và decode.
- **Metric**: **goodput** (max request rate ở X% SLO attainment mỗi GPU) nên là chỉ số đánh giá đầu bảng của HyPrefill thay vì throughput thô.
- **Mượn lập luận**: phê phán O(N²) KV-cache re-read của chunked-prefill ngây thơ hữu ích để lập luận ngược cho HyPrefill — với Gated DeltaNet/linear attention, state kích thước cố định (không tăng dần như KV-cache), nên chi phí "re-chunk" của nhóm này không chịu overhead O(N²) như attention đầy đủ; đây là lý do định lượng để chọn chunk size lớn hơn cho nhóm linear-attention so với full-attention.
- **Mượn phương pháp**: "liệt kê cấu hình + simulator + binary search + đo lỗi <2%" của Algorithm 1/2 là khuôn mẫu tốt để HyPrefill xây profiler/simulator chọn **vector chunk size theo operator-group** thỏa P99 TBT — nhiều chiều hơn bài toán 1 chiều của Sarathi hay bài toán cấu hình song song của DistServe.
- **Khác biệt**: DistServe đổi *nơi* chạy pha nào; HyPrefill đổi *lượng và loại* công việc gộp vào một chunk trong cùng engine. Hai trục bổ sung nhau — về sau có thể kết hợp, nhưng đóng góp riêng của HyPrefill là chunk size theo nhóm toán tử, trục mà cả hai paper chưa khai thác vì chỉ xét Transformer đồng nhất.

### Câu hỏi tự kiểm tra

**Q1:** Vì sao goodput là thước đo phù hợp hơn throughput thô để so sánh các hệ serving có ràng buộc SLO?
**Đáp:** Throughput thô có thể cao nhưng vi phạm SLO cho phần lớn request nên vô dụng thực tế; goodput đo request rate tối đa mà vẫn giữ tỷ lệ đạt SLO (vd 90%) trên mỗi GPU, phản ánh đúng chi phí/hiệu quả khi bị ràng buộc latency — đúng là thứ khách hàng trả tiền để có.

**Q2:** Vì sao chunked-prefill (kiểu Sarathi) không loại bỏ hoàn toàn interference như DistServe làm được?
**Đáp:** Chunked-prefill vẫn chạy prefill và decode chung một GPU/batch, chỉ giảm kích thước lát cắt prefill xen giữa các bước decode nên decode vẫn bị trễ (ít hơn nhưng không bằng 0); hơn nữa việc chia N chunk làm tăng chi phí đọc lại KV-cache lên O(N²) so với O(N) nếu không chunk. DistServe loại bỏ hoàn toàn interference bằng cách tách hẳn phần cứng chạy hai pha.
