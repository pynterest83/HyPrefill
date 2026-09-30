#!/usr/bin/env python3
"""Energy per token of each (mode, rate) of a Layered/chunked demo, as the Layered Prefill paper
defines it (total GPU energy / (prompt + generated tokens)), from the power samples the demo
driver logged (gpu_<tag>.csv, 200 ms) and the benchmark windows in driver.log.

  python bench/demo_energy.py results/step00/layered_demo/<dir> [--raw ~/hyprefill_data/step00/layered_demo/<dir>]
"""
import argparse, datetime as dt, glob, json, os, re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--raw", default=None, help="where the per-request JSON and gpu csv live if moved out of results/")
a = ap.parse_args()
raw = a.raw or a.dir
lines = open(os.path.join(a.dir, "driver.log")).read().splitlines()
ts = lambda l: dt.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
rows, mode_tag = [], None
for i, l in enumerate(lines):
    m = re.search(r"=== (\w+)_(\d+): server", l)
    if m:
        mode_tag = f"{m.group(1)}_{m.group(2)}"
    m = re.search(r"benchmark: \d+ requests at ([\d.]+) req/s", l)
    if m and mode_tag:
        end = next((ts(x) for x in lines[i + 1:] if "client exit" in x), None)
        if end is None:  # rate still running
            continue
        rows.append(dict(tag=mode_tag, rate=float(m.group(1)), start=ts(l), end=end))
out = []
for r in rows:
    g = pd.read_csv(os.path.join(raw, f"gpu_{r['tag']}.csv"), header=None, skipinitialspace=True,
                    names=["time", "gpu", "clk", "power", "util", "reasons"])
    g["time"] = pd.to_datetime(g.time, format="%Y/%m/%d %H:%M:%S.%f")
    g["power"] = g.power.str.replace(" W", "").astype(float)
    w = g[(g.time >= r["start"]) & (g.time <= r["end"])]
    joules = 0.0
    for _, s in w.groupby("gpu"):
        s = s.sort_values("time")
        dtsec = s.time.diff().dt.total_seconds().fillna(0)
        joules += float((s.power * dtsec).sum())
    j = json.load(open(glob.glob(os.path.join(raw, f"{r['tag']}_r{r['rate']}.json"))[0]))
    tokens = j["total_input_tokens"] + j["total_output_tokens"]
    out.append(dict(mode=r["tag"].split("_")[0], rate=r["rate"], seconds=(r["end"] - r["start"]).total_seconds(),
                    energy_kJ=joules / 1e3, tokens=tokens, mJ_per_token=1e3 * joules / tokens,
                    mJ_per_output_token=1e3 * joules / j["total_output_tokens"], mean_power_W=w.power.mean() * w.gpu.nunique()))
d = pd.DataFrame(out).sort_values(["rate", "mode"])
pd.set_option("display.width", 200)
print(d.round(2).to_string(index=False))
p = d.pivot(index="rate", columns="mode", values="mJ_per_token")
if {"chunked", "layered"} <= set(p.columns):
    p["layered_vs_chunked_%"] = 100 * (p.layered - p.chunked) / p.chunked
    print(p.round(2).to_string())
d.to_csv(os.path.join(a.dir, "energy.csv"), index=False)
print("->", os.path.join(a.dir, "energy.csv"))
