#!/usr/bin/env python3
"""Step 2 (plan/02, Việc 4): dump the real per-token expert routing of a MoE model with vLLM's
`enable_return_routed_experts`, for prompts and generated tokens, so that the overlap between
the experts a decode batch touches and the experts a prefill chunk touches can be measured
offline (bench/moe_overlap.py) on real routing instead of a random router.

Output (raw, not committed): ~/hyprefill_data/step02/moe_routing/<date>_<model>/
  req_<i>.npz with `experts` [seq_len, layers, topk] (int16), `prompt_len`
  meta.json with model, TP, dataset, lengths, vLLM version

  CUDA_VISIBLE_DEVICES=4,5 taskset -c 48-71 python bench/moe_routing_dump.py \
      --model <path to Qwen3-Next snapshot> --tp 2 --dataset arxiv --n 64 --max-prompt 16384
"""
import argparse, datetime, json, os, pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent


def prompts(dataset, n, tokenizer, max_prompt, seed):
    import datasets, random
    rng = random.Random(seed)
    if dataset == "arxiv":
        ds = datasets.load_dataset("ccdv/arxiv-summarization", split="test")
        texts = [r["article"] for r in ds]
    elif dataset == "sharegpt":
        ds = datasets.load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                                   data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")
        texts = [" ".join(t["value"] for t in r["conversations"]) for r in ds if r["conversations"]]
    else:
        raise ValueError(dataset)
    rng.shuffle(texts)
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"][:max_prompt]
        if len(ids) >= 512:  # prefill chunks need real consecutive tokens
            out.append(ids)
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dataset", choices=["arxiv", "sharegpt"], default="arxiv")
    ap.add_argument("--n", type=int, default=64, help="number of requests")
    ap.add_argument("--max-prompt", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=128, help="generated tokens per request (decode routing)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    tok = AutoTokenizer.from_pretrained(a.model)
    reqs = prompts(a.dataset, a.n, tok, a.max_prompt, a.seed)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enable_return_routed_experts=True,
              max_model_len=a.max_prompt + a.gen + 16, enable_prefix_caching=False, seed=a.seed)
    sp = SamplingParams(max_tokens=a.gen, min_tokens=a.gen, temperature=0.0)
    outs = llm.generate([{"prompt_token_ids": r} for r in reqs], sp)

    name = pathlib.Path(a.model.rstrip("/")).parts[-3] if "snapshots" in a.model else pathlib.Path(a.model).name
    out = pathlib.Path.home() / "hyprefill_data/step02/moe_routing" / f"{datetime.date.today()}_{name}_{a.dataset}"
    out.mkdir(parents=True, exist_ok=False)
    lens = []
    for i, (r, o) in enumerate(zip(reqs, outs)):
        ex = o.outputs[0].routed_experts
        assert ex is not None, "vLLM returned no routing; is enable_return_routed_experts supported for this model?"
        np.savez_compressed(out / f"req_{i}.npz", experts=ex.astype(np.int16), prompt_len=len(r))
        lens.append([len(r), int(ex.shape[0])])
    json.dump({"model": a.model, "tp": a.tp, "dataset": a.dataset, "n": len(reqs), "gen": a.gen,
               "max_prompt": a.max_prompt, "seed": a.seed, "vllm": vllm.__version__,
               "lens_prompt_total": lens, "date": datetime.datetime.now().isoformat(timespec="seconds")},
              open(out / "meta.json", "w"), indent=1)
    print(f"wrote {out} ({len(reqs)} requests)")


if __name__ == "__main__":
    main()
