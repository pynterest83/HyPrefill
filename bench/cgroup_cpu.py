#!/usr/bin/env python3
"""CPU throttling of this container by its cgroup (cpu.max quota, ~32 cores on quangch1-dev-0).

The engine loop of vLLM and of the layered-prefill fork is single-threaded Python, so a
throttled period stalls the GPU and inflates wall-clock numbers (TTFT, TBT, step time). Every
benchmark records the cpu.stat counters before and after a run; a non-zero `nr_throttled`
means the run was slowed by the quota (often other jobs of this container running at the same
time) and should be checked or re-run. Stdlib only: also runs in the fork's env.

  from cgroup_cpu import snapshot, delta;  s = snapshot(); ...; meta["cgroup_cpu"] = delta(s)
  python bench/cgroup_cpu.py snap > /tmp/s.json
  python bench/cgroup_cpu.py delta /tmp/s.json --tag chunked_r2.5 --csv OUT/cgroup_cpu.csv
"""
import argparse, csv, json, os, pathlib, sys, time

CG = pathlib.Path("/sys/fs/cgroup")


def snapshot():
    s = {"time": time.time()}
    try:
        for line in (CG / "cpu.stat").read_text().splitlines():
            k, v = line.split()
            s[k] = int(v)
        quota, period = (CG / "cpu.max").read_text().split()
        s["quota_cores"] = None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):  # no cgroup v2 here
        pass
    return s


def delta(before, after=None):
    after = after or snapshot()
    d = {"wall_s": round(after["time"] - before["time"], 1), "quota_cores": after.get("quota_cores")}
    if "nr_throttled" not in after or "nr_throttled" not in before:
        return {**d, "available": False}
    d.update(nr_periods=after["nr_periods"] - before["nr_periods"],
             nr_throttled=after["nr_throttled"] - before["nr_throttled"],
             throttled_s=round((after["throttled_usec"] - before["throttled_usec"]) / 1e6, 3),
             cpu_s=round((after["usage_usec"] - before["usage_usec"]) / 1e6, 1))
    d["mean_cores_used"] = round(d["cpu_s"] / d["wall_s"], 2) if d["wall_s"] > 0 else None
    return d


def warn(d, what="run"):
    if d.get("nr_throttled"):
        print(f"WARNING cgroup CPU throttling during {what}: {d['nr_throttled']} periods, "
              f"{d['throttled_s']} s (quota {d['quota_cores']} cores, mean use {d['mean_cores_used']})",
              file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snap", "delta"])
    ap.add_argument("before", nargs="?")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default=None, help="append one row to this CSV")
    a = ap.parse_args()
    if a.cmd == "snap":
        print(json.dumps(snapshot())); return
    d = delta(json.loads(pathlib.Path(a.before).read_text()))
    warn(d, a.tag or "run")
    if a.csv:
        row = {"tag": a.tag, **d}
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    print(json.dumps(d))


if __name__ == "__main__":
    main()
