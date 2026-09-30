#!/usr/bin/env python3
"""Step 1 analysis: the checks in plan/01_COST_FA_GDN.md §3 and draft Figure 1.

  python bench/analyze_step01.py results/step01/<run> [<run> ...] [--fig figures/x.png]
  python bench/analyze_step01.py --repeat results/step01/<runA> results/step01/<runB>
  python bench/analyze_step01.py --merge results/step01/<runs of one config...> --out <table.csv>

For each run directory (from bench/op_cost.py) prints:
  - FA: us/token vs c (one row per t) -> expect ~flat in c, rising with t
  - GDN: spread of p50 across t at fixed c -> must stay < 5% (the key check)
  - GDN fit cost(c) = a + b * ceil(c / 64) -> fixed per-call cost a vs per-chunk cost b
"""
import argparse, json, math, pathlib

import numpy as np
import pandas as pd


def load(run):
    run = pathlib.Path(run)
    df = pd.read_csv(run / "summary.csv")
    meta = json.loads((run / "config.json").read_text())
    return df, meta


def fa_table(df):
    fa = df[df.op == "fa"]
    return fa.pivot(index="t", columns="c", values="us_per_token_p50")


def gdn_t_spread(df):
    g = df[df.op == "gdn"]
    rows = []
    for c, grp in g.groupby("c"):
        p = grp.p50_ms
        rows.append({"c": c, "p50_min_ms": p.min(), "p50_max_ms": p.max(),
                     "spread_pct": 100 * (p.max() - p.min()) / p.min()})
    return pd.DataFrame(rows).set_index("c")


def gdn_t_trend(df):
    """Systematic t-dependence, robust to single outliers: median p50 over t >= 128K vs over
    t <= 16K at each c, plus rows more than 10% off the median across t (outliers)."""
    g = df[df.op == "gdn"]
    rows, outliers = [], []
    for c, grp in g.groupby("c"):
        med = grp.p50_ms.median()
        lo, hi = grp[grp.t <= 16384].p50_ms.median(), grp[grp.t >= 131072].p50_ms.median()
        rows.append({"c": c, "trend_pct": 100 * (hi - lo) / lo})
        for _, r in grp.iterrows():
            if abs(r.p50_ms - med) / med > 0.10:
                outliers.append(f"c={c} t={r.t}: {r.p50_ms * 1000:.1f} us vs median {med * 1000:.1f}")
    return pd.DataFrame(rows).set_index("c"), outliers


def gdn_fit(df):
    """Least-squares fit of p50 = a + b * ceil(c/64), pooled over t (GDN is t-independent)."""
    g = df[df.op == "gdn"]
    x = np.ceil(g.c / 64.0).to_numpy()
    y = g.p50_ms.to_numpy()
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2


def repeatability(run_a, run_b):
    """plan/00 check: two identical runs agree within 3% (p50 GPU time, row by row)."""
    a, _ = load(run_a)
    b, _ = load(run_b)
    m = a.merge(b, on=["op", "c", "t"], suffixes=("_a", "_b"))
    m["diff_pct"] = 100 * (m.p50_ms_b - m.p50_ms_a).abs() / m.p50_ms_a
    clk = [x for x in ("sm_clock_mean_mhz_a", "sm_clock_mean_mhz_b") if x in m]
    print(f"\n######## repeatability {pathlib.Path(run_a).name} vs {pathlib.Path(run_b).name}")
    for op, g in m.groupby("op"):
        within = (g.diff_pct < 3).mean() * 100
        print(f"{op}: {len(g)} rows, median diff {g.diff_pct.median():.2f}%, "
              f"p95 {g.diff_pct.quantile(0.95):.2f}%, max {g.diff_pct.max():.2f}%, within 3%: {within:.0f}%")
    worst = m.sort_values("diff_pct", ascending=False).head(8)
    cols = ["op", "c", "t", "p50_ms_a", "p50_ms_b", "diff_pct"] + clk
    print("worst rows:")
    print(worst[cols].to_string(index=False))
    ok = (m.diff_pct < 3).all()
    print(f"-> {'PASS' if ok else 'FAIL'}: every row within 3%" if ok else
          f"-> FAIL: {(m.diff_pct >= 3).sum()} of {len(m)} rows differ by >= 3%")


def merge(runs, out):
    """Cost table for the oracle: per (op, c, t) the median p50 over repeated runs, with the
    spread across runs. Median, because power-capped rows (FA and big GEMMs at 700 W) vary
    with the clock the GPU happened to hold in each run."""
    dfs = []
    for i, r in enumerate(runs):
        df, meta = load(r)
        assert meta.get("clock_locked"), f"{r} was not measured with locked clocks"
        dfs.append(df.assign(run=i))
    d = pd.concat(dfs)
    g = d.groupby(["op", "layers", "c", "t"])
    t = g.agg(p50_ms=("p50_ms", "median"), min_run_ms=("p50_ms", "min"), max_run_ms=("p50_ms", "max"),
              n_runs=("p50_ms", "size"), clock_mean_mhz=("sm_clock_mean_mhz", "median"),
              clock_min_mhz=("sm_clock_min_mhz", "min"), host_ms=("host_ms", "median")).reset_index()
    t["spread_pct"] = 100 * (t.max_run_ms - t.min_run_ms) / t.p50_ms
    t["us_per_token"] = 1000 * t.p50_ms / t.c
    t["power_capped"] = t.clock_min_mhz < t.clock_mean_mhz.max()  # below the lock (highest clock seen)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False, float_format="%.6g")
    print(f"{out}: {len(t)} rows from {len(runs)} runs; spread median {t.spread_pct.median():.2f}%, "
          f"max {t.spread_pct.max():.1f}%; power-capped rows {int(t.power_capped.sum())}")


def figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 3.6 * len(runs)), squeeze=False)
    for row, (name, df, meta) in enumerate(runs):
        for col, op in enumerate(["fa", "gdn"]):
            ax = axes[row][col]
            d = df[df.op == op]
            if d.empty:
                ax.set_visible(False); continue
            cmap = plt.get_cmap("viridis")
            ts = sorted(d.t.unique())
            for i, t in enumerate(ts):
                s = d[(d.t == t) & (d.c % 64 == 0)].sort_values("c")
                ax.plot(s.c, s.us_per_token_p50, marker="o", ms=3, lw=1.5,
                        color=cmap(i / max(1, len(ts) - 1)), label=f"t={t // 1024}K" if t else "t=0")
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("chunk c (tokens)"); ax.set_ylabel("µs / token / layer (p50)")
            lock = "locked" if meta.get("clock_locked") else "NOT locked"
            ax.set_title(f"{name}: {op.upper()} (TP={meta['shapes'].get('tp', 1)}, clock {lock})", fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nfigure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fig", type=pathlib.Path)
    ap.add_argument("--repeat", action="store_true", help="compare exactly two runs row by row")
    ap.add_argument("--merge", action="store_true", help="median cost table over repeated runs")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    if a.merge:
        merge(a.runs, a.out)
        return
    if a.repeat:
        assert len(a.runs) == 2, "--repeat takes two runs"
        pd.set_option("display.width", 200, "display.precision", 4)
        repeatability(*a.runs)
        return
    pd.set_option("display.width", 200, "display.precision", 3)
    loaded = []
    for r in a.runs:
        df, meta = load(r)
        name = pathlib.Path(r).name
        loaded.append((name, df, meta))
        print(f"\n######## {name}  (clock_locked={meta.get('clock_locked')}, "
              f"gdn_backend={meta.get('gdn_backend')}, fa_version={meta.get('fa_version')})")
        bad = df[(df.gpu_util_pre > 5) | (df.foreign_mem_mib > 2048)] if "gpu_util_pre" in df else df.iloc[:0]
        print(f"rows with foreign GPU load: {len(bad)}")
        if (df.op == "fa").any():
            print("\nFA us/token (p50), rows = t, cols = c:")
            print(fa_table(df).to_string())
        if (df.op == "gdn").any():
            sp = gdn_t_spread(df)
            print("\nGDN p50 spread across t at fixed c (key check: < 5%):")
            print(sp.to_string())
            worst = sp.spread_pct.max()
            print(f"-> worst spread {worst:.1f}%  => {'PASS' if worst < 5 else 'FAIL'} (max-min, sensitive to one outlier)")
            tr, outl = gdn_t_trend(df)
            wt = tr.trend_pct.abs().max()
            print(f"-> systematic trend, t>=128K vs t<=16K: worst {wt:.1f}%  => {'PASS' if wt < 5 else 'FAIL'}")
            print(f"   outliers (>10% off the median across t): {outl or 'none'}")
            a_, b_, r2 = gdn_fit(df)
            print(f"\nGDN fit p50 = a + b*ceil(c/64):  a = {a_ * 1000:.1f} us (fixed per call), "
                  f"b = {b_ * 1000:.2f} us per 64-token chunk, R^2 = {r2:.3f}")
            print(f"   a dominates up to c ~ {64 * a_ / b_:.0f} tokens (where a = b*ceil(c/64))")
    if a.fig:
        figure([x for x in loaded if "triton" not in x[0]], a.fig)


if __name__ == "__main__":
    main()
