import sys, time, json, glob, os, statistics as st
import numpy as np
from vllm import LLM, SamplingParams


def main():
    mode, c = sys.argv[1], int(sys.argv[2])
    model = sorted(glob.glob(os.path.expanduser("~/hf_cache/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/*/")))[0]
    kw = dict(model=model, tensor_parallel_size=2, max_num_batched_tokens=c, max_model_len=16384 + c + 64, enable_prefix_caching=False)
    if mode == "capture":
        kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]}
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config.compilation_config
    print("CAPTURE", cfg.cudagraph_capture_sizes, "mode", cfg.cudagraph_mode, flush=True)
    rng = np.random.default_rng(0)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    p = lambda: [{"prompt_token_ids": rng.integers(1000, 100000, 16384).tolist()}]
    llm.generate(p(), sp, use_tqdm=False)
    w = []
    for _ in range(3):
        q = p(); t0 = time.perf_counter(); llm.generate(q, sp, use_tqdm=False); w.append((time.perf_counter() - t0) * 1e3)
    steps = -(-16384 // c)
    print("RESULT", json.dumps(dict(mode=mode, c=c, wall_ms=st.median(w), steps=steps, wall_per_step_ms=st.median(w) / steps)), flush=True)


if __name__ == '__main__':
    main()
