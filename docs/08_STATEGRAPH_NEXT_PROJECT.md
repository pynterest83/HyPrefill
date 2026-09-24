# StateGraph — dự án kế tiếp sau HyPrefill

Ghi chú ngày 24/09/2026. Không khởi động trước khi HyPrefill nộp xong (~28/01/2027).

## Tên đề tài
StateGraph: Reconstructability-Aware Inference State Management for Large Language Models

## Vì sao hoãn chứ không bỏ
Lý do hoãn là **thời gian**, không phải chất lượng. Phần việc ước 6–10 tuần, cần DeepSeek V4.1 trên 4–8 GPU qua đường SGLang mới, quá chậm cho deadline tháng 1/2027.

Tiền đề đang **mạnh lên** theo thời gian: xu hướng sparse attention thêm nhiều loại state (compressed KV, indexer key, Top-K selection) bên cạnh SSM state và SWA replay state. Blog hạ tầng GLM (17/09/2026) cho thấy hệ production đang quản lý những thứ này một cách tạm bợ (Layer Split, cache quantization hỗn hợp INT8/FP8/BF16).

Nhưng novelty đang **bị bào mòn** bởi chính các hệ thống đang ship: Marconi, Jenga, Cake, AttentionStore, và vLLM hybrid KV cache manager. Đây là bài **tổng hợp và tổng quát hoá**, không phải bài cơ chế mới — phải nói thẳng ngay Intro để tránh bị đánh "incremental over X, Y, Z".

## Venue — quyết định 24/09/2026

**Không nhắm NeurIPS.** Không có tiền lệ đã xác minh nào cho paper quản lý KV/state ở venue ML. Cake (arXiv:2410.03065) không ghi venue trên arXiv; Jenga (arXiv:2503.18292) là preprint. Các tiền lệ chắc chắn đều ở systems: Marconi → MLSys 2025, AttentionStore → USENIX ATC 2024, PagedAttention → SOSP 2023. StateGraph cũng thiếu góc "tính đúng đắn" mà HyPrefill dùng để bán cho reviewer ML.

**Không nhắm SOSP 2027 (~01/04/2027).** Chỉ 9 tuần từ tháng 2, trong khi riêng phần việc đã 6–10 tuần. Thêm nữa tháng 2–4/2027 trùng vòng phản biện ICML của HyPrefill.

| Ưu tiên | Venue | Deadline | Tuần từ 02/2027 |
|---|---|---|---|
| 1 | **ASPLOS 2028 cycle 2** | ~09/2027 | ~30 |
| 1 | **EuroSys 2028 fall** | ~09/2027 | ~30 |
| 2 | **OSDI 2028** | ~12/2027 | ~44 |
| 3 | SOSP 2028 | ~04/2028 | ~60 |

Cả bốn đều CCF-A. Kiểm lại ngày chính xác khi khởi động.

## Hạ tầng tái sử dụng được từ HyPrefill
Kỷ luật đo lường, quyền truy cập H200, model đã tải, khung simulator trace-level, kinh nghiệm xử lý state hybrid. Ước tái sử dụng 30–40%, không phải bắt đầu từ con số 0.

## Việc phải làm đầu tiên khi khởi động
Quét scoop lại toàn bộ. Marconi, Jenga và vLLM hybrid KV manager đều đang tiến; kiểm tra phần nào còn trống trước khi đầu tư.
