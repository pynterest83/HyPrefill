# Bước 5 — Design doc và hạ tầng chạy theo nhóm layer trong vLLM

**Mục tiêu:** Viết design doc 2 trang, rồi cài **một cơ chế chạy theo nhóm layer dùng chung** trong vLLM 0.30. Layered Prefill là cấu hình k = 1 của cơ chế này, HyPrefill là cấu hình có chunk riêng theo operator (quyết định 2026-09-29, xem §3).

---

## 1. Đầu ra bắt buộc

- [ ] `docs/06_DESIGN.md` — 2 trang, viết **trước khi code**
- [ ] Fork vLLM tại tag v0.30.0 (`third_party/vllm`), build được từ source trong env `hyprefill`, chạy lại được §2.5 bước 1 với kết quả như bản wheel
- [ ] **Chạy theo nhóm layer:** một request prefill có thể dừng sau nhóm layer g ở iteration i và chạy tiếp từ nhóm g + 1 ở iteration i + 1; activation giữa các nhóm và state GDN được giữ đúng
- [ ] **Chế độ Layered** (k = 1, cùng chunk cho mọi operator, `N_lg` chỉnh được) chạy đúng trên Qwen3-30B-A3B và Qwen3-Next
- [ ] **Kiểm chứng bản Layered của mình với code gốc của tác giả:** trên Qwen3-30B-A3B, cùng máy, cùng trace arXiv, tỉ lệ cải thiện goodput "layered so với chunked" trong vLLM của mình khớp tỉ lệ đó trong fork nanovllm gốc trong ±15% (tương đối). Chưa khớp thì chưa dùng làm baseline
- [ ] Output token-level giống chunked prefill của vLLM trên 20 prompt greedy, cho cả chế độ Layered

## 2. Nội dung design doc

### 2.1 Operator group
Chia model thành các nhóm layer liên tiếp theo loại operator. Với Qwen3-Next (12 × (3 GDN + 1 gated attention)): nhóm FA và nhóm nonFA xen kẽ, nhóm MoE sau mỗi mixer. Nêu rõ cách nhóm khi số layer không chia hết. Layered dùng nhóm theo độ sâu (`N_lg` nhóm liên tiếp), HyPrefill dùng nhóm theo loại operator; cả hai là cách chia khác nhau trên cùng hạ tầng.

### 2.2 Chunk theo group
`c_FA(t)` từ cost model đã kiểm chứng (bước 1–3); `c_nonFA = k · c_FA` với k từ mô phỏng bước 3. Layered: k = 1.

### 2.3 Activation buffer
Output nhóm FA cho chunk i giữ trong buffer đến khi đủ k chunk. Kích thước `k · c_FA · d_model · 2` byte. Tính con số thật cho từng model và đưa vào doc. Buffer phải nằm trong phần bộ nhớ vLLM đã tính (không để engine OOM).

### 2.4 Stagger
Các nhóm nonFA chạy chunk lớn ở iteration khác nhau (offset) để đỉnh mỗi iteration ≤ B. Viết rõ thuật toán chọn offset.

### 2.5 Ràng buộc của vLLM — phần dễ sai nhất
| Ràng buộc | Nguồn | Hệ quả |
|---|---|---|
| Kernel chunk GDN = 64, giống nhau giữa prefill và decode | vLLM RFC #55524, PR #49827 | **Không đổi**; chỉ đổi số token scheduler đưa vào |
| **Prefix cache với model hybrid:** chế độ `mamba_cache_mode = "align"` (mặc định khi bật prefix cache) buộc scheduler cắt chunk prefill theo ranh giới block Mamba (`need_mamba_block_aligned_split` trong `v1/core/sched/scheduler.py`); block size là 544 (Qwen3-Next TP2) hoặc 784 (Qwen3.8-27B TP1) | vLLM 0.30, PR #54076 | Chunk của HyPrefill và của mọi baseline phải tuân theo; ghi rõ ràng buộc này làm c nhỏ bị làm tròn ra sao. **Workload chính (append-prefill) cần prefix cache**, nên đây không phải chi tiết phụ |
| Buffer chunked-scan GDN scale theo token batch | vLLM issue #54775 | Thêm ràng buộc bộ nhớ vào cost model |
| CUDA graph: batch có prefill chỉ graph từng mảnh; attention và lõi GDN chạy eager (`splitting_ops`) | vLLM 0.30 `config/compilation.py` | Chạy theo nhóm layer không được làm hỏng việc graph các mảnh còn lại; đo chi phí CPU thêm (bước 6) |
| Model runner chạy trọn chiều sâu mỗi bước | vLLM V1 `gpu_model_runner` | Phần sửa lớn nhất; xem scheduler có thay được bằng `scheduler_cls` không, phần chạy theo nhóm layer thì phải sửa model runner |

### 2.6 Decode không đổi
Decode batch chạy bình thường mỗi iteration, đi hết chiều sâu. Chỉ phần prefill thay đổi.

### 2.7 Bộ kernel
Không còn là quyết định riêng: mọi chế độ (chunked, Layered, HyPrefill) chạy trong cùng vLLM 0.30, cùng kernel với bảng cost bước 1–2 (FA3, GDN FlashInfer, MoE Triton). Fork nanovllm gốc chỉ dùng để kiểm chứng bản Layered (§1) và cho một bảng tham khảo khác engine; nó dùng kernel riêng nên **không** so trực tiếp với HyPrefill.

## 3. Vì sao làm thẳng trong vLLM, không làm trên fork trước (quyết định 2026-09-29)

| | Fork nanovllm (layered-prefill) | vLLM 0.30 |
|---|---|---|
| Công bằng với Layered | Code gốc của tác giả, nhưng không có GDN nên Layered-cho-hybrid vẫn là bản mình cài | Layered là cấu hình k = 1 của cùng cơ chế với HyPrefill: ablation sạch nhất, không ai nói được là cài baseline kém |
| Áp dụng vào open source | Không | Có, sau này gửi được PR |
| Bảng cost | Phải đo lại bằng kernel của fork | Dùng lại bảng bước 1–2 |
| Công sức | Ít hơn | Nhiều hơn (model runner V1); nhưng HyPrefill cần hạ tầng này trong vLLM đằng nào cũng phải làm |
| Vai trò | **Phương án lui** nếu vLLM chưa chạy được end-to-end ở cổng cứng bước 8; và để kiểm chứng bản Layered | **Hướng chính** |

Cách Layered Prefill tự so với vLLM (tham khảo): phép so chính của họ đều trong engine của họ; bảng so với vLLM 0.10.2 chunked chỉ là tham khảo khác engine ("cấu hình tương tự"). Mình làm tương tự: mọi claim trong một engine (vLLM), cộng một bảng tham khảo khác engine (nanovllm gốc so với vLLM của mình).


---

## KẾT QUẢ

> **Để trống — điền khi làm xong bước này.**

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

### R5. Việc chuyển sang bước sau

-

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
