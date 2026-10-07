#!/usr/bin/env python3
"""Step 1 (plan/01): cost of the TP all-reduce vLLM runs after each row-parallel layer (twice per
decoder layer: after attention/GDN o_proj and after the MLP/MoE), per number of tokens n.

Goes through vLLM's own path (`tensor_model_parallel_all_reduce` on vLLM's TP group), so the
backend is the one vLLM picks by size (custom all-reduce for small tensors, NCCL above). Timing
as in bench/op_cost.py: a CUDA graph with K back-to-back all-reduces (captured under vLLM's
`graph_capture`, as the engine does), replay timed with CUDA events, divided by K. Rank 0
writes results/step01/<date>_allreduce_<tag>/.

  CUDA_VISIBLE_DEVICES=4,5 taskset -c 48-71 torchrun --nproc-per-node 2 bench/allreduce_cost.py \
      --hidden 2048 --tag Qwen3-Next-80B-A3B-Instruct_tp2
"""
import argparse, csv, datetime, json, os, pathlib, statistics, subprocess

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--n", default="1,8,16,32,64,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--k", type=int, default=20, help="all-reduces per captured graph")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (graph_capture, init_distributed_environment,
                                                 initialize_model_parallel)
    import cgroup_cpu; cg0 = cgroup_cpu.snapshot()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, "env://", int(os.environ["LOCAL_RANK"]), "nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)

        rows = []
        for n in [int(x) for x in a.n.split(",")]:
            x = torch.randn(n, a.hidden, device=dev, dtype=torch.bfloat16)
            for _ in range(3):
                tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with graph_capture(dev) as ctx:
                with torch.cuda.graph(g, stream=ctx.stream):
                    for _ in range(a.k):
                        y = tensor_model_parallel_all_reduce(x)
            for _ in range(a.warmup):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); g.replay(); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) / a.k)
            ts.sort()
            rows.append(dict(op="allreduce", n=n, bytes=n * a.hidden * 2, p50_ms=statistics.median(ts),
                             p99_ms=ts[min(len(ts) - 1, int(len(ts) * 0.99))], min_ms=ts[0], max_ms=ts[-1]))
            if rank == 0:
                r = rows[-1]
                print(f"n={n:6d} ({r['bytes'] / 2**20:7.2f} MiB): p50 {r['p50_ms'] * 1000:8.2f} us  "
                      f"{r['bytes'] / r['p50_ms'] / 1e6:6.1f} GB/s algo", flush=True)
            del g, x

    if rank == 0:
        name = f"{datetime.date.today()}_allreduce" + (f"_{a.tag}" if a.tag else "")
        base = REPO / "results/step01"; base.mkdir(parents=True, exist_ok=True)
        d, i = base / name, 2
        while True:
            try:
                d.mkdir(); break
            except FileExistsError:
                d = base / f"{name}_run{i}"; i += 1
        with open(d / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,clocks.sm,power.limit", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        json.dump({"hidden": a.hidden, "world": world, "k": a.k, "gpus": gpus,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "lock_mhz": os.environ.get("LOCK_MHZ"), "cgroup_cpu": cgroup_cpu.delta(cg0), "torch": torch.__version__,
                   "date": datetime.datetime.now().isoformat(timespec="seconds")}, open(d / "config.json", "w"), indent=2)
        print(f"wrote {d.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
