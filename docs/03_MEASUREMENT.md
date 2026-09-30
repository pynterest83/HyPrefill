# Kỷ luật đo lường

Áp dụng từ ngày đầu tiên, không đợi đến khi có số thật. Một bảng số không tái lập được là một tuần mất trắng.

## 1. GPU

```bash
# khoá clock trước mỗi phiên đo (H200), chỉ các GPU dùng để đo
sudo nvidia-smi -i 4,5,6,7 -pm 1
sudo nvidia-smi -i 4,5,6,7 -lgc 1980,1980

# trả về mặc định khi xong
sudo nvidia-smi -i 4,5,6,7 -rgc
```

- Trên `quangch1-dev-0`, root trong container **không** khoá được clock; admin phải khoá ở host hoặc cấp quyền. Script đo phải dừng hẳn nếu khoá thất bại, không được chạy tiếp rồi ghi `clock_locked: true`.
- Xác nhận khoá có hiệu lực: lúc nhàn rỗi `clocks.sm` phải đứng ở 1980 MHz (chưa khoá thì về khoảng 345 MHz).
- Mọi phiên đo cùng một mức clock. Số đo ở hai mức clock khác nhau không so với nhau được. **Mức chuẩn: 1980 MHz** (mức boost tối đa, gần với lúc serve thật; chốt 2026-09-29). Số đo ở 1830 MHz trước đó đã chuyển sang `~/hyprefill_data/clk1830/`.
- Khoá không giữ được clock khi chạm trần công suất (700 W). Đã thấy trên GPU 4 (khoá 1830 MHz): FA3 Qwen3-Next TP2 ở c = 8192, t = 256K chạy ở 1140–1830 MHz, trung bình ~1380 MHz, 700 W; GDN giữ đúng 1830. `bench/op_cost.py` lấy mẫu clock và công suất **trong lúc đo** (NVML, ~10 ms) và ghi `sm_clock_mean_mhz`, `sm_clock_min_mhz`, `power_max_w` theo từng dòng. Các dòng bị giới hạn công suất vẫn dùng được nếu lặp lại được giữa hai lượt đo (plan/00: lệch < 3%); nếu không, khoá ở mức thấp hơn mà GPU giữ được và dùng đúng mức đó cho mọi phiên.
- **GPU phải được dùng riêng trong lúc đo.** Tiến trình của container khác không hiện trong `nvidia-smi`; kiểm tra `memory.used` và `utilization.gpu` trước và trong lúc đo (sau khi để GPU nhàn rỗi ~1 s, vì `utilization.gpu` là trung bình theo cửa sổ), ghi vào từng dòng kết quả. Dòng nào có tải lạ thì bỏ.

Kiểm tra nhiệt độ trước khi đo: nếu GPU đang nóng từ phiên trước, số đầu tiên sẽ lệch. Chờ về nhiệt độ nền.

## 2. Đo thời gian

- Dùng `torch.cuda.Event`, **không** dùng `time.time()` quanh lời gọi bất đồng bộ.
- **Micro-benchmark kernel: đo replay CUDA graph chứa K lần [xoá L2 → op]**, trừ đi graph chỉ có K lần xoá L2, chia K (K sao cho một lần replay khoảng 2 ms, tối đa 50). Lý do: (a) event quanh một lần replay vẫn tính cả ~4 µs CPU phát lệnh graph, đủ làm kernel 40 µs lệch 5–10% giữa hai lượt đo (đã gặp 28/09); (b) xoá L2 (ghi 256 MB > 60 MB L2) đặt op vào trạng thái cache lạnh như khi serve (state GDN, KV từ HBM; layer khác chạy xen giữa); GDN c = 64 lệch 40.5 µs (L2 nóng) so với 46.9 µs (L2 lạnh). Cách đo này lặp lại trong ±0.1 µs với kernel nhỏ. Không đo event quanh một lần gọi eager. Với kernel nhỏ, gọi eager bị giới hạn bởi CPU (wrapper GDN FlashInfer tốn ~0.2 ms CPU so với ~0.04 ms GPU ở c = 64), nên event đo tốc độ CPU; CPU trên máy dùng chung dao động và từng tạo ra một phụ thuộc t giả ở bước 1. Ghi riêng thời gian CPU mỗi lần gọi (`host_ms`), vì vLLM chạy GDN và attention ngoài CUDA graph nên chi phí này có thật khi serve.
- **Đo đúng cách vLLM lưu dữ liệu, không phải cách tiện nhất.** (a) KV của attention dạng **paged**, block size đúng như vLLM tính cho model hybrid (trang attention ≥ trang state Mamba; Qwen3.8-27B TP1: 784, Qwen3-Next TP2: 544, model không hybrid: 16) — `kv_block_size` trong `bench/op_cost.py`, đã đối chiếu với log engine. KV liền mạch rẻ hơn 5–14% và làm bảng cost thấp hơn thực tế. (b) State GDN theo `mamba_ssm_dtype` của config (Qwen3.8: fp32; Qwen3-Next: bf16, wrapper FlashInfer ép sang fp32 mỗi lần gọi).
- Tách chi phí nhiều thành phần nhỏ (bước 6) bằng profiler (`torch.profiler` hoặc `nsys`), không bằng event quanh từng đoạn eager.
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
results/stepNN/<ngày>/
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

## 7. So sánh công bằng với baseline

Mọi con số so sánh HyPrefill với baseline phải thoả **tất cả** các điều sau.

**Cùng phần cứng và cấu hình, chỉ khác scheduler:**
- Cùng GPU cụ thể (ví dụ luôn là GPU 4–5 cho TP = 2), cùng TP, cùng cặp GPU, CPU ghim cùng NUMA node (`taskset`), cùng clock đã khoá, cùng power limit.
- Cùng engine, cùng phiên bản (driver, CUDA, torch, engine commit), cùng backend kernel (attention, GDN, MoE), cùng cấu hình CUDA graph, KV cache, prefix cache (bật hoặc tắt như nhau), `max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`.
- **Chạy lại mọi baseline trên chính máy này.** Không bao giờ đặt số của mình cạnh số trong paper của họ: họ đo trên H100/A100/H800, engine khác, model khác.
- **Cost model phải đo trên đúng bộ kernel của engine đang đánh giá.** Bảng cost đo bằng kernel vLLM 0.30 không dùng được cho prototype chạy kernel khác, và ngược lại.

**Baseline được tune tốt nhất, bằng cùng thông tin HyPrefill có:**
- Static chunk: quét {256, 512, 1024, 2048, 4096, 8192} (bội 128, vì FA3 có bậc thang theo tile 128), lấy mức tốt nhất ở từng SLO.
- Layered Prefill: quét số nhóm layer `N_lg` (paper dùng `⌈L/512⌉`, nhưng Bảng 11 của họ cho thấy `N_lg` đổi TTFT lấy TBT), lấy mức tốt nhất ở từng SLO.
- SLOWeave: dùng chính bảng cost đã đo làm hàm T, tune biên an toàn δ.
- Nếu HyPrefill dùng thông tin mà baseline không được dùng, phần gain đó là "thông tin", không phải "cơ chế"; phải nói rõ hoặc cho baseline cùng thông tin.

**Cùng engine, cùng hạ tầng:** mọi policy trong vLLM 0.30; Layered là cấu hình k = 1 của hạ tầng chạy theo nhóm layer dùng chung với HyPrefill, nên hai bên chỉ khác chính sách chunk. So khác engine (ví dụ nanovllm gốc với vLLM) chỉ được báo như bảng tham khảo.

**Kiểm chứng bản cài lại của baseline:** chạy bản cài lại trên đúng bài của họ (Layered Prefill: Qwen3-30B-A3B, ShareGPT và arXiv) trên máy này, so **tỉ lệ cải thiện** với paper (Bảng 6: TTFT −56%), không so giá trị tuyệt đối. Lệch nhiều thì sửa bản cài lại trước khi dùng làm baseline.

**Cách chạy:**
- Cùng trace, cùng seed, cùng quá trình đến (Poisson, cùng rate), loại bỏ warmup như nhau.
- Chạy **xen kẽ** trong cùng phiên (A-B-A-B…), không chạy hết baseline hôm nay rồi HyPrefill hôm khác. Mỗi cấu hình ≥ 3 lần, median kèm khoảng dao động.
- Tính cả chi phí của chính HyPrefill (buffer, stagger, thời gian ra quyết định).
- Output token-level giống nhau giữa mọi policy (greedy).
- Năng lượng chỉ đo trên GPU dùng riêng: `power.draw` gồm cả tải của người khác.

## 8. Trước khi tin một con số

- Chạy lại trên phiên khác, số có lặp lại trong 3% không?
- Có làm sanity check bằng một cách khác không? (ví dụ: tổng cost dự đoán so với một forward pass thật)
- Đơn vị đúng chưa? ms hay µs, GB hay GiB.
- Có so với giới hạn lý thuyết không? Nếu kết quả vượt băng thông HBM hoặc peak FLOPS thì phép đo sai.

## 9. Trước khi đưa một con số vào paper

- Truy được về `results/stepNN/<ngày>/raw.csv` nào?
- Cấu hình có ghi trong `config.json` không?
- Có khoảng dao động kèm theo không, hay chỉ có một số trần trụi?
- Baseline có được tune công bằng không? (Nếu so với static chunk, phải quét vài mức và lấy mức tốt nhất, không lấy mức tệ nhất.) Có thoả đủ §7 không?
- `config.json` có ghi `clock_locked: true` và mọi dòng có tải lạ bằng 0 không?
