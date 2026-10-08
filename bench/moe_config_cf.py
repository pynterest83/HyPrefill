#!/usr/bin/env python3
"""Counterfactual for KT1 (diagnosis, 2026-10-08): how much of the measured MoE amortization
comes from the fused_moe tile config rather than from shared expert reads.

vLLM picks the Triton config of the nearest tuned M (tokens) in
E=512,N=256,device_name=NVIDIA_H200.json. Those configs were tuned with a random router
(M=512: ~10 tokens per expert, BLOCK_SIZE_M=16); real routing concentrates tokens on fewer
experts, and bench/moe_diag.py shows M in 385..768 costing more than M=1024. Here the KT1 cells
(real routing, same paired draws as op_cost.py moe_mixed) are timed under the default config and
under every distinct config of the tuned table (override_config); min over configs = what a
routing-aware tuning could reach.

  CUDA_VISIBLE_DEVICES=4 taskset -c 48-71 python bench/moe_config_cf.py --routing <dump> --clock-locked
Output: results/step02/<date>_moe_config_cf/{config.json, summary.csv}
"""
import argparse, csv, datetime, json, os, pathlib, shlex, subprocess, sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cgroup_cpu  # noqa: E402
import op_cost  # noqa: E402
from op_cost import ClockSampler, gpu_state, gpu_time, load_routing, load_shapes, make_moe_mixed, stats, versions  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "results/step00/configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--draws", default="0,1,2")
    ap.add_argument("--decode-batch", default="8,32,64")
    ap.add_argument("--c", default="0,256,512,1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--clock-locked", action="store_true")
    ap.add_argument("--tag", default="moe_config_cf")
    a = ap.parse_args()
    assert a.warmup >= 10 and a.iters >= 30, "docs/03_MEASUREMENT.md: warmup >= 10, iters >= 30"
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
    dev, dtype = "cuda:0", torch.bfloat16
    shape = load_shapes(a.config, a.tp)["moe"]
    torch.zeros(1, device=dev)
    before = gpu_state()
    busy = [g for g in before if int(g["utilization.gpu"]) > 5 or int(g["memory.used"]) * 10 > int(g["memory.total"])]
    if busy:
        sys.exit(f"GPU busy: {busy}")
    import vllm.model_executor.layers.fused_moe.fused_moe as fm
    fname = get_config_file_name(shape["experts"], shape["inter"], None)
    table = json.load(open(pathlib.Path(fm.__file__).parent / "configs" / fname))
    cands, seen = [("default", None)], set()
    for m in sorted(table, key=int):
        key = json.dumps(table[m], sort_keys=True)
        if key not in seen:
            seen.add(key); cands.append((f"M{m}", table[m]))

    base = REPO / "results" / "step02"
    d, i = base / f"{datetime.date.today()}_{a.tag}", 2
    while True:  # never overwrite
        try:
            d.mkdir(parents=True); break
        except FileExistsError:
            d = base / f"{datetime.date.today()}_{a.tag}_run{i}"; i += 1
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "model_config": a.config, "tp": a.tp,
            "shape": shape, "config_file": fname, "candidates": dict(cands[1:]), "routing": a.routing,
            "gpu": before, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **versions(),
            "repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "--porcelain")),
            "cmd": " ".join(shlex.quote(x) for x in sys.argv), "clock_locked": a.clock_locked,
            "warmup": a.warmup, "iters": a.iters}
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    f = open(d / "summary.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["draw", "decode_batch", "c", "config", "p50_ms", "p99_ms", "min_ms", "max_ms", "n", "experts_touched",
                "sm_clock_mean_mhz", "sm_clock_min_mhz", "power_max_w"])
    sampler = ClockSampler()
    cg0 = cgroup_cpu.snapshot()
    for draw in map(int, a.draws.split(",")):
        op_cost.ROUTING = load_routing(a.routing, draw)
        for bd in map(int, a.decode_batch.split(",")):
            for c in map(int, a.c.split(",")):
                torch.cuda.empty_cache()
                fn = make_moe_mixed(shape, bd, c, dev, dtype)  # same paired draw as op_cost.py
                touched = op_cost.MOE_TOUCHED[0]
                best = None
                for name, cfg in cands:
                    try:
                        with override_config(cfg), sampler:  # override must be active while the graph is captured
                            samples, _, _ = gpu_time(fn, a.warmup, a.iters)
                    except Exception as e:  # a config can be invalid for some M (e.g. shared memory)
                        print(f"draw {draw} bd={bd} c={c} {name}: {type(e).__name__}"); continue
                    s = stats(samples)
                    clk_mean, clk_min, pw = sampler.summary()
                    w.writerow([draw, bd, c, name, f"{s['p50']:.5f}", f"{s['p99']:.5f}", f"{s['min']:.5f}",
                                f"{s['max']:.5f}", s["n"], touched, f"{clk_mean:.0f}", clk_min, f"{pw:.0f}"])
                    best = min(best or (9e9, ""), (s["p50"], name))
                    if name == "default":
                        dflt = s["p50"]
                f.flush()
                print(f"draw {draw} bd={bd:2d} c={c:5d} experts={touched}  default {dflt:.4f} ms  best {best[0]:.4f} ({best[1]})", flush=True)
    meta["cgroup_cpu"] = cgroup_cpu.delta(cg0); cgroup_cpu.warn(meta["cgroup_cpu"])
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()
