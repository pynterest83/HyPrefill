#!/usr/bin/env python3
"""Summarize a Layered/chunked demo directory (bench/step00_layered_demo.sh) into summary.csv
and move the bulky per-request JSON and logs to ~/hyprefill_data (they are not committed).

SLO attainment follows the Layered Prefill paper (arXiv 2510.08055 §6): a request attains the
SLO if TTFT <= --ttft-slo and every TBT after it <= --tbt-slo (defaults: their arXiv SLOs).
Goodput (PROPOSAL §4.4) = highest measured rate with attainment >= 90%.

  python bench/summarize_demo.py results/step00/layered_demo/<dir> [--keep-raw]
"""
import argparse, csv, glob, json, os, pathlib, re, shutil

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--ttft-slo", type=float, default=10.0, help="seconds")
ap.add_argument("--tbt-slo", type=float, default=0.125, help="seconds")
ap.add_argument("--keep-raw", action="store_true")
a = ap.parse_args()
home = pathlib.Path.home()
# cgroup CPU throttling per benchmark run (bench/cgroup_cpu.py), when the run recorded it
def throttled(D):
    f = D / "cgroup_cpu.csv"
    return {r["tag"]: r.get("throttled_s", "") for r in csv.DictReader(open(f))} if f.exists() else {}

for D in a.dirs:
    D = pathlib.Path(D)
    rows, thr = [], throttled(D)
    for f in sorted(D.glob("*.json")):
        if f.name == "config.json":
            continue
        d = json.load(open(f))
        m = re.match(r"(chunked|layered)_(\d+)(?:_r([\d.]+))?", f.stem)
        ok = sum(1 for t, it in zip(d["ttfts"], d["itls"])
                 if t is not None and t <= a.ttft_slo and all(x <= a.tbt_slo for x in it))
        rows.append(dict(run=f.stem, mode=m.group(1), rate=float(m.group(3) or d["request_rate"]),
                         completed=d["completed"], request_throughput=round(d["request_throughput"], 4),
                         slo_attain_pct=round(100 * ok / len(d["ttfts"]), 2),
                         **{k: round(d[k], 3) for k in ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                                                         "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms",
                                                         "mean_e2el_ms", "p99_e2el_ms"]},
                         mean_input_len=round(sum(d["input_lens"]) / len(d["input_lens"]), 1),
                         cpu_throttled_s=thr.get(f.stem, "")))
    if not rows:
        print(f"{D}: no result JSON"); continue
    with open(D / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{D}: SLO TTFT<={a.ttft_slo}s, TBT<={a.tbt_slo * 1000:.0f}ms")
    for mode in ("chunked", "layered"):
        rs = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["rate"])
        good = [r["rate"] for r in rs if r["slo_attain_pct"] >= 90]
        print(f"  {mode:8s} " + "  ".join(f"{r['rate']:.1f}:{r['slo_attain_pct']:.0f}%" for r in rs)
              + f"   goodput >= {max(good) if good else 0:.1f} req/s")
    hit = [r["run"] for r in rows if r["cpu_throttled_s"] not in ("", "0.0", "0")]
    if hit:
        print("  WARNING cgroup CPU throttling in: " + ", ".join(hit) + " (column cpu_throttled_s)")
    if not a.keep_raw:
        raw = home / "hyprefill_data/step00/layered_demo" / D.name
        raw.mkdir(parents=True, exist_ok=True)
        for f in D.iterdir():
            if f.name not in ("config.json", "driver.log", "summary.csv", "cgroup_cpu.csv"):
                assert not (raw / f.name).exists(), raw / f.name
                shutil.move(str(f), raw / f.name)
        print(f"  raw -> {raw}")
