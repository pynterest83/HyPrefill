# Kỷ luật đo lường

Áp dụng từ ngày đầu tiên, không đợi đến khi có số thật. Một bảng số không tái lập được là một tuần mất trắng.

## 1. GPU

```bash
# khoá clock trước mỗi phiên đo (H200)
sudo nvidia-smi -lgc 1980,1980
sudo nvidia-smi -pm 1

# trả về mặc định khi xong
sudo nvidia-smi -rgc
```

Kiểm tra nhiệt độ trước khi đo: nếu GPU đang nóng từ phiên trước, số đầu tiên sẽ lệch. Chờ về nhiệt độ nền.

## 2. Đo thời gian

- Dùng `torch.cuda.Event`, **không** dùng `time.time()` quanh lời gọi bất đồng bộ.
- Warmup ≥ 10 lần trước khi tính.
- Lấy median của ≥ 30 lần, ghi cả P50 và P99, và khoảng dao động.
- `torch.cuda.synchronize()` đúng chỗ, không thừa không thiếu.

```python
import torch
def timed(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"p50": ts[len(ts)//2], "p99": ts[int(len(ts)*0.99)],
            "min": ts[0], "max": ts[-1], "n": iters}
```

## 3. Đo bộ nhớ

Với layer indexer của sparse attention, bộ nhớ đỉnh là một ràng buộc thật, phải đo:

```python
torch.cuda.reset_peak_memory_stats()
fn()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
```

Đo cả `max_memory_reserved()` vì allocator có thể không tái dùng block khi kích thước tăng dần (đúng hiện tượng trong vLLM issue #56457).

## 4. Đo năng lượng

```bash
nvidia-smi --query-gpu=index,timestamp,power.draw --format=csv -lms 100 > power.csv &
PID=$!
# chạy thí nghiệm
kill $PID
```

Tích phân theo thời gian, chia cho số token sinh ra, ra J/token.

## 5. Lưu kết quả

```
results/<tuần>/<ngày>/
├── config.json      model, shape, phiên bản thư viện, commit hash, lệnh chạy
├── raw.csv          số thô, một dòng một lần đo
└── summary.csv      đã tổng hợp
```

Quy tắc: **không bao giờ ghi đè**. Chạy lại thì tạo thư mục ngày mới. Dung lượng rẻ hơn nhiều so với một buổi chiều cố nhớ xem số cũ đến từ cấu hình nào.

`config.json` tối thiểu:

```json
{
  "date": "2026-09-22",
  "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "gpu": "H200", "n_gpu": 2, "tp": 2,
  "dtype": "bfloat16",
  "torch": "...", "flash_attn": "...", "fla": "...", "vllm_commit": "...",
  "repo_commit": "...",
  "cmd": "python bench/op_cost.py --op fa --c 64,128,...",
  "clock_locked": true
}
```

## 6. Nhật ký

`LOG.md`, mỗi ngày 5 dòng: làm gì, số liệu chính, điều bất ngờ, chỗ bị chặn, việc mai. Viết trong ngày, không viết bù cuối tuần.

## 7. Trước khi tin một con số

- Chạy lại trên phiên khác, số có lặp lại trong 3% không?
- Có làm sanity check bằng một cách khác không? (ví dụ: tổng cost dự đoán so với một forward pass thật)
- Đơn vị đúng chưa? ms hay µs, GB hay GiB.
- Có so với giới hạn lý thuyết không? Nếu kết quả vượt băng thông HBM hoặc peak FLOPS thì phép đo sai.

## 8. Trước khi đưa một con số vào paper

- Truy được về `results/<tuần>/<ngày>/raw.csv` nào?
- Cấu hình có ghi trong `config.json` không?
- Có khoảng dao động kèm theo không, hay chỉ có một số trần trụi?
- Baseline có được tune công bằng không? (Nếu so với static chunk, phải quét vài mức và lấy mức tốt nhất, không lấy mức tệ nhất.)
