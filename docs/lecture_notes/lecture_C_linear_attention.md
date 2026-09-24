# Bài giảng C — Linear Attention / Gated DeltaNet: vì sao chi phí prefill không phụ thuộc t

Mục tiêu: hiểu vì sao chi phí xử lý một chunk c token mới của Gated DeltaNet (GDN) không phụ thuộc số token t đã xử lý trước, để xây `cost_GDN(c)` cho scheduler của HyPrefill. Phân biệt: **kernel chunk C=64** (hằng số cứng trong kernel FLA, `FLA_CHUNK_SIZE`) vs **scheduler chunk c** (số token HyPrefill gom vào một lượt forward, có thể lẻ, không chia hết cho 64).

---

# Phần 1 — Gated Delta Networks (arXiv 2412.06464, ICLR 2025)

## 1. Bối cảnh

Bốn bước tiến hoá, mỗi bước sửa nhược điểm bước trước:
1. **Softmax attention**: mỗi query so khớp toàn bộ key trước đó → chi phí prefill O(T²), KV-cache tăng tuyến tính theo T.
2. **Linear attention**: thay softmax(QK^T) bằng tích feature map tuyến tính, viết lại thành một **state ma trận** cập nhật tuần tự kiểu RNN → chi phí mỗi bước là hằng số, không phụ thuộc t. Nhược điểm: state cộng dồn vô hạn, không "quên" được, dễ nhiễu/bão hoà.
3. **Gated linear attention (Mamba2, RetNet)**: thêm cổng quên α_t giảm dần ảnh hưởng thông tin cũ. Nhưng decay là *đồng nhất*, không phân biệt cặp key-value nào cần giữ, cặp nào nên ghi đè.
4. **Delta rule (DeltaNet)**: mượn ý "delta learning rule" — thay vì chỉ cộng dồn, state được cập nhật *có chọn lọc*: gỡ thành phần chiếu lên hướng key hiện tại rồi ghi giá trị mới vào đúng khe đó (giống "xoá rồi ghi" theo địa chỉ = key). Cải thiện rõ rệt khả năng associative recall.

**Gated DeltaNet** hợp nhất: cổng quên vô hướng kiểu Mamba2 (xoá sạch bộ nhớ nhanh khi cần) + delta rule chọn lọc (ghi đè chính xác theo key). Đây là block dùng trong các mô hình hybrid (Qwen3-Next, Kimi Linear...) mà HyPrefill nhắm tới.

## 2. Công thức cốt lõi

S_t ∈ ℝ^{d_k×d_v}; q_t,k_t ∈ ℝ^{d_k}, v_t ∈ ℝ^{d_v}.

- Linear attention: **S_t = S_{t-1} + k_t v_t^T, o_t = S_t^T q_t**
- Mamba2 (decay vô hướng): S_t = α_t S_{t-1} + k_t v_t^T
- DeltaNet: **S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t k_t v_t^T**. (I − β_t k_t k_t^T) là Householder tổng quát: chiếu bỏ thành phần của S_{t-1} theo hướng k_t rồi ghi đè giá trị mới — khác cộng dồn thuần của linear attention.
- **Gated DeltaNet**: **S_t = S_{t-1}·α_t(I − β_t k_t k_t^T) + β_t k_t v_t^T**. α_t: quên toàn cục (α→0 xoá sạch state); β_t: ghi đè chọn lọc theo key (β→1 thay hoàn toàn khe nhớ ứng với k_t).

Để song song hoá delta rule trong 1 chunk C token, dùng **WY representation/UT transform** (biểu diễn tích nhiều phép Householder liên tiếp bằng một cặp ma trận nhỏ thay vì nhân tuần tự C ma trận d×d):
- T = [I + strictLower(diag(β)KK^T)]^{-1} diag(β) ∈ ℝ^{C×C}
- W = TK, Ũ = TV (pseudo-key/value đã hấp thụ hiệu ứng ghi-đè tuần tự trong chunk); bản gated thêm mặt nạ decay Γ (tích luỹ α trong chunk) vào KK^T.

## 3. Thuật toán chunkwise

1. Chia T token thành N=T/C chunk (C=64, cố định trong kernel FLA). Q[t],K[t],V[t] ∈ ℝ^{C×d}.
2. Intra-chunk: tính T (C×C), W=TK, Ũ=TV — matmul cỡ C×C và C×d, chi phí **O(C²d)**.
3. Đầu ra trong chunk: O[t] = Q̄[t]S[t]^T + (Q[t]K[t]^T⊙M)(Ũ[t]−W̄[t]S[t]^T), M là mặt nạ nhân quả — vẫn O(C²d).
4. Inter-chunk: S[t+1] = S̄[t] + (Ũ[t]−W̄[t]S[t]^T)^T K̄[t] — nhân (C×d)^T×(C×d)→d×d, chi phí **O(Cd²)**, không phụ thuộc số chunk đã qua, chỉ phụ thuộc C, d.
5. Lặp N lần theo thứ tự (chunk sau cần S[t] chunk trước — tuần tự bắt buộc giữa chunk, song song hoàn toàn bên trong chunk).

**Chi phí mỗi chunk**: O(C²d)+O(Cd²), chỉ phụ thuộc C,d. **Bộ nhớ state**: một ma trận d_k×d_v/head, không tăng theo T. Bài không tự công bố FLOP/kernel chi tiết, chỉ dẫn: "chunkwise algorithm could be similarly adapted... as implemented in Flash Linear Attention" — nghĩa là kernel thực thi (`chunk_gated_delta_rule` trong thư viện FLA) kế thừa thiết kế "hai pha" (materialization) của bài GLA ở Phần 2.

## 4. Vì sao chi phí không phụ thuộc t

Khi scheduler đưa c token mới vào (đã có state đầu S₀ từ t token trước, nhưng S₀ chỉ là ma trận d_k×d_v cố định), thuật toán chia c thành ⌈c/C⌉ chunk, mỗi chunk tốn hằng số O(C²d+Cd²):

**cost_GDN(c | có state từ t token) ≈ (c/C)·[γC²d + δCd²] + overhead_cố_định** — không số hạng nào chứa t.

Lý do: linear attention/GDN nén toàn bộ lịch sử vào state kích thước cố định, nên "đọc" lịch sử tốn O(d²) (một matmul d×d) chứ không phải O(t·d) như quét lại KV-cache. Ngược lại **full attention** cho c token mới trên nền t token cũ tốn **α·c·t** (mỗi trong c query nhân t key cũ) — tăng tuyến tính theo t. Đây là cơ sở định lượng cho việc cấp chunk size khác nhau giữa attention layer và GDN layer trong kiến trúc hybrid.

## 5. Kết quả số

Huấn luyện 400M và 1.3B tham số (100B token, FineWeb-Edu, ngữ cảnh 4K). Kết quả chính:
- Perplexity WikiText: GDN 16.42 vs Mamba2 16.56, DeltaNet 17.71.
- Recall thực tế (2K context): GDN 30.6% vs Mamba2 29.8%, DeltaNet 26.2%; hybrid GDN-H2: 40.1%.
- S-NIAH (4K): GDN vượt trội Mamba2 (vd NIAH-3: 27.6% vs 4.6%).
- LongBench (14 task): GDN 16.6%, GDN-H2 18.4%, vs Mamba2 13.5%, Transformer++ 11.0%.
- Thông lượng train (H100): GDN ≈ DeltaNet, chậm hơn Mamba2 ~2-3K token/giây; Transformer++ nhanh nhất ở ngữ cảnh ngắn nhờ FlashAttention-2 tối ưu sâu, nhưng lợi thế đảo ngược khi ngữ cảnh dài.

## 6. Rút ra cho HyPrefill

- `cost_GDN(c)` nên là **affine theo c**: a + b·⌈c/64⌉ (a: overhead cố định/lần gọi kernel, b: chi phí biên mỗi chunk-64, gộp cả intra C²d và inter Cd²) — **không có số hạng nhân t**.
- Đo micro-benchmark: (1) overhead cố định (gọi với c rất nhỏ, ngoại suy về c=0); (2) tỉ trọng intra vs inter (đổi d_k,d_v,num_heads); (3) xác nhận thực nghiệm cost không đổi khi t thay đổi (cùng c, S₀ sinh từ t=0/1K/10K/100K token, kiểm tra latency bất biến).
- Kernel gọi khi đo: `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, truyền `initial_state` để mô phỏng đúng tình huống chunked-prefill có state.
- Ràng buộc: kernel chunk C giữ nguyên =64 (`FLA_CHUNK_SIZE` trong vLLM) — chi tiết triển khai Triton, không phải biến HyPrefill tối ưu. Biến tối ưu là **scheduler chunk c**, có thể không chia hết 64 và khác chunk size dùng cho attention layer.

## 7. Câu hỏi tự kiểm tra

**Câu 1**: Vì sao cần cả α_t lẫn β_t, không dùng riêng một trong hai?
*Đáp*: α_t cho "quên toàn cục nhanh" (DeltaNet thuần thiếu điều này). β_t cho "ghi đè chọn lọc theo key" (Mamba2 thuần chỉ decay đồng nhất, không phân biệt key). Kết hợp cho khả năng biểu diễn tốt hơn, rõ nhất ở recall/NIAH.

**Câu 2**: Vì sao chi phí xử lý c token mới không phụ thuộc t dù state "chứa" thông tin của t token đó?
*Đáp*: State S có kích thước cố định d_k×d_v bất kể tích luỹ từ bao nhiêu token — không phải danh sách tăng theo t như KV-cache. Đọc S tốn O(d²), không cần quét lại t token cũ, nên chi phí chỉ là hàm của c (và d,C).

---

# Phần 2 — Gated Linear Attention Transformers (arXiv 2312.06635, ICML 2024)

## 1. Bối cảnh

Bài này nối lý thuyết linear attention với huấn luyện hiệu quả trên GPU:
1. **Softmax attention**: O(L²) vì phải vật chất hoá ma trận attention L×L (dù FlashAttention giải quyết tốt I/O, FLOP vẫn bậc hai).
2. **Linear attention dạng hồi quy**: S_t=S_{t-1}+k_t^Tv_t, o_t=q_tS_t — FLOP tuyến tính O(Ld²) nhưng mỗi bước chỉ là phép cộng ma trận nhỏ, arithmetic intensity thấp, không tận dụng tensor core → chậm thực tế dù FLOP thấp.
3. **Dạng song song**: khai triển thành QK^T rồi nhân V, tận dụng tốt tensor core nhưng quay lại O(L²d).
4. Cần dạng vừa tuyến tính FLOP vừa tận dụng matmul → **chunkwise parallel form**: trong chunk dùng công thức song song (matmul C×C), giữa chunk dùng công thức hồi quy (cộng dồn state).

Song song đó, bài thêm **cổng quên** vào linear attention (như Phần 1): tránh state cộng dồn vô hạn gây nhiễu, và dùng decay *phụ thuộc dữ liệu* (data-dependent) thay vì decay cố định như RetNet, cải thiện chất lượng ngôn ngữ.

## 2. Công thức cốt lõi

- Linear attention (Eq.1): S_t=S_{t-1}+k_t^Tv_t, o_t=q_tS_t, S_t∈ℝ^{d×d} (quy ước hàng-vector; tương đương S_t=S_{t-1}+k_tv_t^T, o_t=S_t^Tq_t theo quy ước cột ở Phần 1).
- **GLA (Eq.3)**: **S_t = Diag(α_t)S_{t-1} + k_t^Tv_t**, α_t∈ℝ^{1×d_k} là **vector cổng theo từng kênh (channel-wise)** — khác α_t vô hướng của Mamba2/GDN, cho mỗi chiều key tốc độ quên riêng, biểu diễn phong phú hơn nhưng phức tạp hơn khi song song hoá.
- Tham số hoá gate: α_t = σ(x_tW_α^1W_α^2+b_α)^{1/τ}, hạng thấp W_α^1∈ℝ^{d×16}, W_α^2∈ℝ^{16×d_k}, τ=16 — thêm rất ít tham số (~4d²/layer) mà vẫn phụ thuộc dữ liệu.
- Chunkwise chưa gate (Eq.2): liên-chunk S[i+1]=S[i]+K[i+1]^TV[i+1]; trong-chunk O[i+1]=Q[i+1]S[i]+(Q[i+1]K[i+1]^T⊙M)V[i+1].
- Chunkwise có gate: decay tích luỹ Λ (đầu chunk→j), Γ (j→cuối chunk), γ (toàn chunk): **S[i+1] = (γ_{i+1}^T·1)⊙S[i] + (K[i+1]⊙Γ_{i+1})^TV[i+1]**, phần trong-chunk nhân thêm Λ,Γ vào Q,K trước matmul.

## 3. Thuật toán chunkwise — FlashLinearAttention

Thiết kế gốc mà GDN (Phần 1) kế thừa nguyên vẹn. **Thuật toán 1 — 2 pha (materialization)**:
1. Chia L token thành N=L/C chunk, C=64 (bội số 16, khớp tile tensor core).
2. **Pha 1 (tuần tự, SRAM)**: S=0; với n=1..N nạp K[n],V[n], S←S+K[n]^TV[n], **ghi S[n] ra HBM**.
3. **Pha 2 (song song giữa các chunk)**: nạp Q[n],K[n],V[n],S[n]; O[n]=Q[n]S[n]+(Q[n]K[n]^T⊙M)V[n].

Ý tưởng: Pha 1 rẻ (O(Ld²)) nên chạy tuần tự nhanh; có state khởi đầu mọi chunk rồi thì Pha 2 chạy song song hoàn toàn — đổi ~10-20% bộ nhớ (lưu N state ra HBM) lấy song song hoá cấp-chuỗi tối đa. Bản không-materialize (giữ state trong SRAM) tiết kiệm bộ nhớ hơn nhưng mất song song cấp-chuỗi, phải tuần tự.

**Secondary-level chunking** (bản gate): chia mỗi chunk C thành sub-chunk nhỏ hơn để tránh mất ổn định số do decay quá nhỏ — phần liên-sub-chunk dùng matmul bán chính xác, phần nội-sub-chunk tính trong không gian log ở độ chính xác đầy đủ.

**Chi phí**: intra O(C²d+Cd²)/chunk; tổng O(LCd+Ld²) — so với O(L²d) (song song thuần) và O(Ld²) (hồi quy thuần). Khi C≪L: LCd+Ld² ≪ L²d, nhưng C,d đủ lớn để tensor core hiệu quả (khác hồi quy chỉ nhân vector-ma trận từng bước, tận dụng phần cứng kém).

## 4. Vì sao chi phí không phụ thuộc t

S[i+1]=(γ^T·1)⊙S[i]+(K[i+1]⊙Γ)^TV[i+1]: **S luôn cỡ d×d cố định**, không phụ thuộc số chunk i đã qua. Xử lý chunk mới (c token, chia ⌈c/C⌉ chunk) chỉ tốn: cập nhật state Cd²/chunk (chỉ phụ thuộc d,C, không phụ thuộc t=i·C) + attention nội-chunk C²d/chunk (cục bộ, không phụ thuộc t). Vậy:

**cost(c | có state từ t token) = (c/C)·[γC²d+δCd²] + overhead** — không có t.

So với **full attention**: xử lý c token mới trên nền t token cache tốn α·c·(t+c)≈α·c·t khi t≫c — tăng tuyến tính t, vì attention giữ nguyên toàn bộ t vector key/value và phải quét lại mỗi lần, không nén thành state cố định. Đây là cơ sở lý thuyết trực tiếp cho ý tưởng "per-operator-group chunk size" của HyPrefill.

## 5. Kết quả số

Huấn luyện 340M (15B token) và 1.3B (100B token), SlimPajama, tokenizer Mistral:
- Perplexity WikiText (1.3B): GLA 17.22 vs Transformer++ 16.85, RetNet 18.64 (GLA vượt RetNet rõ nhờ gate phụ thuộc dữ liệu).
- Downstream (LAMBADA, PIQA, HellaSwag, WinoGrande, ARC): GLA, Mamba tương đương Transformer++.
- MQAR (recall tổng hợp): GLA vượt RetNet, Mamba, Hyena, RWKV-4.
- Recall thực tế (FDA, SWDE, SQUAD): mô hình subquadratic thua Transformer, nhưng GLA vượt các mô hình subquadratic khác.
- Ngoại suy: train 8K, test tốt tới 18K token.
- **Tốc độ kernel (H100, batch 32)**: FlashLinearAttention (cả hai biến thể) nhanh hơn rõ rệt FlashAttention-2 trên dải 512–32K token; lợi thế thông lượng của GLA so Mamba/Transformer++ càng rõ khi độ dài huấn luyện >4096 token.

## 6. Rút ra cho HyPrefill

- Bài gốc định nghĩa **C=64** và "hai pha materialization" — nền tảng mà `chunk_gated_delta_rule` và hằng số `FLA_CHUNK_SIZE=64` (vLLM) kế thừa. C=64 là tham số kernel do lý do phần cứng (bội số 16), **không phải biến HyPrefill tối ưu**.
- Micro-benchmark nên tách 3 phần khi fit `cost_GDN(c)`: (1) overhead cố định/lệnh gọi (gần bất biến theo c nhỏ); (2) chi phí Pha 1 (Cd²/chunk, tuyến tính theo ⌈c/64⌉); (3) chi phí Pha 2 (C²d/chunk, tuyến tính theo số chunk nhưng có thể song song giữa chunk).
- c không nhất thiết chia hết 64 → cần đo cả c lẻ (kernel pad chunk cuối), ví dụ c=65 gần tốn bằng c=128 vì ⌈65/64⌉=2.
- Đo với `initial_state` khác None (giả lập state đã tích luỹ nhiều token) để kiểm chứng thực nghiệm: latency bất biến khi thay t, giữ c cố định — bằng chứng trực tiếp cho công thức affine cost_GDN(c)=a+b·c trước khi dùng trong scheduler.

## 7. Câu hỏi tự kiểm tra

**Câu 1**: Thiết kế "materialization" (lưu S[n] ra HBM ở Pha 1) đánh đổi gì lấy gì?
*Đáp*: Đánh đổi bộ nhớ (+~10-20% do lưu N state trung gian ra HBM) lấy khả năng song song hoá cấp-chuỗi ở Pha 2 — mọi chunk tính output đồng thời vì đã có sẵn state khởi đầu riêng, không cần chờ chunk trước.

**Câu 2**: Nếu HyPrefill chọn scheduler chunk c cho layer GDN độc lập với layer attention, điều gì đảm bảo việc này không phá vỡ tính đúng của kết quả so với xử lý nguyên khối?
*Đáp*: State liên-chunk S[i] là tóm tắt chính xác, không mất mát của mọi token trước đó (công thức đóng, không xấp xỉ). Chia chuỗi thành đoạn c bất kỳ và truyền state qua `initial_state` giữa các lệnh gọi cho kết quả toán học **giống hệt** xử lý nguyên khối — chunking chỉ ảnh hưởng phân bổ compute/song song hoá, không ảnh hưởng giá trị output, miễn state truyền đúng thứ tự.
