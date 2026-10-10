#!/usr/bin/env python3
"""Workload characterization of semianalysisai/cc-traces-weka-062126-256k (plan/04): what a serving
engine with prefix caching would actually have to prefill per request.

Per request: context = `in` tokens; new tokens = blocks (64 tokens, `hash_ids`) not seen earlier in the
same trace (hash_id_scope = local), i.e. the append a perfect per-session prefix cache cannot serve;
cached = context - new. Reports distributions (overall, main agent vs sub-agent), the share of prefill
work by context and append bucket, output lengths and inter-arrival times within a trace.

  python bench/trace_stats.py [--trace <traces.jsonl>]
Output: results/step04/<date>_trace_stats/{summary.json, buckets.csv, requests_sample.csv}
"""
import argparse, collections, datetime, glob, json, os, pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
    default = (glob.glob(f"{hf}/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl") or [None])[0]
    ap.add_argument("--trace", default=default)
    a = ap.parse_args()
    rows = []
    types = collections.Counter()
    with open(a.trace) as f:
        for line in f:
            tr = json.loads(line)
            bs = tr.get("block_size", 64)
            seen = set()
            last_t = None
            flat = []  # main-agent requests and the inner requests of sub-agent groups, in time order
            for r in tr["requests"]:
                if r.get("type") == "subagent":
                    flat += [dict(x, type="sub") for x in r.get("requests", []) if "in" in x]
                elif "in" in r:
                    flat.append(dict(r, type="main"))
            flat.sort(key=lambda r: r["t"])
            for r in flat:
                h = r.get("hash_ids", [])
                new_blocks = sum(1 for x in h if x not in seen)
                seen.update(h)
                types[r.get("type")] += 1
                new = min(r["in"], new_blocks * bs)
                rows.append(dict(trace=tr["id"], t=r["t"], type=r.get("type"), model=r.get("model"), ctx=r["in"],
                                 new=new, cached=r["in"] - new, out=r["out"], ttft=r.get("ttft"),
                                 gap=(r["t"] - last_t) if last_t is not None else np.nan))
                last_t = r["t"]
    d = pd.DataFrame(rows)
    out = REPO / "results/step04" / f"{datetime.date.today()}_trace_stats"
    out.mkdir(parents=True, exist_ok=False)
    q = lambda s: {f"p{p}": float(np.nanpercentile(s, p)) for p in (10, 25, 50, 75, 90, 99)} | {"mean": float(np.nanmean(s))}
    summary = dict(trace=a.trace, requests=len(d), traces=int(d.trace.nunique()), types=dict(types),
                   context=q(d.ctx), new_tokens=q(d.new), cached_frac=float(d.cached.sum() / d.ctx.sum()),
                   out=q(d.out), gap_s=q(d.gap.dropna()),
                   prefill_tokens_with_cache=int(d.new.sum()), prefill_tokens_without_cache=int(d.ctx.sum()))
    for ty, g in d.groupby("type"):
        summary[f"type_{ty}"] = dict(n=len(g), context=q(g.ctx), new_tokens=q(g.new), out=q(g.out))
    # where the prefill work (new tokens) sits: by context of the request and by append size
    cb = [0, 16384, 32768, 65536, 131072, 196608, 262144]
    nb = [0, 512, 2048, 4096, 8192, 16384, 32768, 65536, 262144]
    d["ctx_bucket"] = pd.cut(d.ctx, cb, right=True)
    d["new_bucket"] = pd.cut(d.new, nb, right=True)
    b = d.groupby(["ctx_bucket", "new_bucket"], observed=True).agg(requests=("new", "size"), new_tokens=("new", "sum")).reset_index()
    b["share_of_prefill"] = b.new_tokens / d.new.sum()
    b.to_csv(out / "buckets.csv", index=False, float_format="%.4f")
    by_ctx = d.groupby("ctx_bucket", observed=True).new.sum() / d.new.sum()
    by_new = d.groupby("new_bucket", observed=True).new.sum() / d.new.sum()
    summary["prefill_share_by_context"] = {str(k): float(v) for k, v in by_ctx.items()}
    summary["prefill_share_by_append"] = {str(k): float(v) for k, v in by_new.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    d.sample(min(5000, len(d)), random_state=0).drop(columns=["ctx_bucket", "new_bucket"]).to_csv(out / "requests_sample.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("type_")}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
