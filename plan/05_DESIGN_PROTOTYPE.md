# Tuần 05 — Design doc và khởi động prototype

**Ngày:** 20–26/10/2026
**Mục tiêu:** Viết design doc 2 trang, fork codebase, thêm hỗ trợ layer GDN.

---

## 1. Đầu ra bắt buộc

- [ ] `docs/06_DESIGN.md` — 2 trang, viết **trước khi code**
- [ ] Fork `scale-snu/layered-prefill`, build và chạy lại được
- [ ] Layer GDN thêm vào fork (dùng kernel `fla`), chạy đúng trên Qwen3.8-27B hoặc Qwen3-Next

## 2. Nội dung design doc

### 2.1 Operator group
Chia model thành các nhóm layer liên tiếp theo loại operator. Với Qwen3-Next (12 × (3 GDN + 1 gated attention)): nhóm FA và nhóm nonFA xen kẽ, nhóm MoE sau mỗi mixer. Nêu rõ cách nhóm khi số layer không chia hết.

### 2.2 Chunk theo group
`c_FA(t)` từ cost model closed-form; `c_nonFA = k · c_FA` với k từ bảng oracle.

### 2.3 Activation buffer
Output nhóm FA cho chunk i giữ trong buffer đến khi đủ k chunk. Kích thước `k · c_FA · d_model · 2` byte. Tính con số thật cho từng model và đưa vào doc.

### 2.4 Stagger
Các nhóm nonFA chạy chunk lớn ở iteration khác nhau (offset) để đỉnh mỗi iteration ≤ B. Viết rõ thuật toán chọn offset.

### 2.5 Ràng buộc kernel — phần dễ sai nhất

| Ràng buộc | Nguồn | Hệ quả |
|---|---|---|
| Kernel chunk GDN = 64, giống nhau giữa prefill và decode | vLLM RFC #55524, PR #49827 | **Không đổi**; chỉ đổi số token scheduler đưa vào |
| Boundary align theo block Mamba khi bật prefix cache | vLLM PR #54076 | c phải là bội của block size nhóm Mamba |
| Buffer chunked-scan GDN scale theo token batch | vLLM issue #54775 | Thêm ràng buộc bộ nhớ vào cost model |

### 2.6 Decode không đổi
Decode batch chạy bình thường mỗi iteration. Chỉ phần prefill thay đổi.

## 3. Vì sao fork nanovllm trước, không phải vLLM

| | Fork layered-prefill | vLLM |
|---|---|---|
| Ưu | codebase nhỏ, đã có scheduling theo layer group, sửa 1–2 tuần | model hybrid chạy sẵn, thuyết phục reviewer hơn |
| Nhược | chưa có GDN, phải thêm | scheduler V1 + model runner phức tạp; ràng buộc align |
| Quyết định | **làm trước**, để có số end-to-end sớm (tuần 8) | bắt đầu tuần 9 nếu fork cho số tốt |


---

## KẾT QUẢ

> **Để trống — điền khi làm xong tuần này.**

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

### R5. Việc chuyển sang tuần sau

-

---

## Nhật ký

| Ngày | Việc làm | Kết quả / chặn ở đâu |
|---|---|---|
|  |  |  |
