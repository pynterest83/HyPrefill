# Bước 9 — Củng cố trong vLLM (hoặc phương án lui trên fork)

**Mục tiêu:** Đưa bản vLLM lên mức eval đầy đủ: TP, prefix cache, mọi ràng buộc kernel, và các baseline còn lại (SLOWeave) trong cùng engine.

---

## 1. Đầu ra bắt buộc

- [ ] HyPrefill và Layered chạy trong vLLM trên Qwen3-Next TP2 và Qwen3.8-27B TP1, có prefix cache
- [ ] Không vi phạm ràng buộc kernel (danh sách bên dưới)
- [ ] **SLOWeave cài trong cùng vLLM** (không có code chính thức), dùng cùng bảng cost, tune δ
- [ ] Bảng tham khảo khác engine: fork nanovllm gốc (Layered, chunked) so với vLLM của mình trên Qwen3-30B-A3B, cùng máy, cùng trace. Chỉ để tham khảo, không làm claim

## 2. Ràng buộc bắt buộc tôn trọng

| Ràng buộc | Nguồn | Kiểm tra bằng |
|---|---|---|
| `FLA_CHUNK_SIZE = 64` không đổi | vLLM PR #49827 | grep trong code, không override |
| Chunk size kernel Mamba2 giống nhau prefill và decode | vLLM RFC #55524 | chạy test bit-identical nếu có |
| Chunk boundary align theo block size nhóm Mamba khi bật prefix cache (`mamba_cache_mode = "align"`) | vLLM PR #54076, `v1/core/sched/scheduler.py` | assert trong scheduler |
| Buffer chunked-scan GDN nằm ngoài memory profiling | vLLM issue #54775 | đặt trần thủ công, tránh OOM |

## 3. Chiến lược giảm rủi ro

- **Prefix cache bật cho workload append-prefill**, vì workload này dựa vào việc context cũ đã nằm trong cache (sửa 2026-09-29; bản cũ ghi "tắt prefix cache trong eval chính", mâu thuẫn với workload chính). Hybrid + prefix cache còn bug (SGLang #39342, vLLM #43587): kiểm tính đúng sớm (bước 5) và ghi rõ phiên bản. Workload long-context chạy thêm cấu hình không prefix cache.
- **TP:** Qwen3-Next bf16 (~152 GB) không vừa một H200, chạy TP2 như bảng cost bước 1. Kiểm vLLM 0.30 hỗ trợ TP2 cho batch trộn GDN. Nếu không, dùng bản FP8 (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`) ở TP1 và **đo lại bảng cost** cho đúng shape và dtype đó. Mọi policy chạy cùng TP, cùng dtype.

## 4. Nếu bước 8 đã chuyển sang phương án lui

- **Fork:** thêm GDN vào fork nanovllm (dùng kernel vLLM 0.30 nếu port được sang torch mới, nếu không thì đo lại bảng cost bằng kernel của fork), cài HyPrefill cạnh Layered gốc, so trong fork.
- **Measurement:** mở rộng bảng cost sang model thứ 4 và 5, làm sâu mô phỏng đã kiểm chứng, bắt đầu viết sớm.


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
