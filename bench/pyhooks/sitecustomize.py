# Loaded by every Python process when bench/pyhooks is on PYTHONPATH (including multiprocessing
# spawn workers). With HYP_DYNAMO_RECOMPILE_LIMIT set, raise torch._dynamo's recompile limit
# without editing third-party code: the layered-prefill fork hits the default limit (8) in
# forward_attention, whose guards depend on a Python int (max_seqlen_k_dec), and then falls
# back to eager (seen on 2026-09-29 in both chunked and layered mode).
import os

_lim = os.environ.get("HYP_DYNAMO_RECOMPILE_LIMIT")
if _lim:
    try:
        import torch._dynamo.config as _c
        for _name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
            if hasattr(_c, _name):
                setattr(_c, _name, max(int(_lim), getattr(_c, _name)))
    except Exception as _e:  # never break the interpreter
        print(f"[sitecustomize] could not raise dynamo recompile limit: {_e}")


# With HYP_PROFILE_DIR set, profile a window of nanovllm ModelRunner.run calls in the rank-0
# model process (the layered-prefill fork runs its engine core in a separate process, so the
# profiler cannot be wrapped from outside). Observation only: the fork's code is not changed.
#   HYP_PROFILE_DIR=<dir> HYP_PROFILE_START=<calls to skip> HYP_PROFILE_CALLS=<calls to record>
_pdir = os.environ.get("HYP_PROFILE_DIR")
if _pdir:
    import importlib.abc, importlib.machinery, sys as _sys

    def _patch(mod):
        MR = mod.ModelRunner
        orig = MR.run
        start, calls = int(os.environ.get("HYP_PROFILE_START", "2000")), int(os.environ.get("HYP_PROFILE_CALLS", "200"))
        state = {"n": 0, "prof": None}

        def run(self, *a, **k):
            if getattr(self, "rank", 0) != 0:
                return orig(self, *a, **k)
            state["n"] += 1
            if state["n"] == start:
                import torch
                state["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                state["prof"].__enter__()
            out = orig(self, *a, **k)
            if state["prof"] is not None and state["n"] == start + calls:
                import torch
                torch.cuda.synchronize()
                state["prof"].__exit__(None, None, None)
                os.makedirs(_pdir, exist_ok=True)
                state["prof"].export_chrome_trace(os.path.join(_pdir, f"rank0_calls{start}-{start + calls}.json"))
                print(f"[sitecustomize] profiled ModelRunner.run calls {start}..{start + calls} -> {_pdir}", flush=True)
                state["prof"] = None
            return out
        MR.run = run

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "nanovllm.engine.model_runner":
                return None
            _sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            _sys.meta_path.insert(0, self)
            if spec is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)
            loader.exec_module = exec_module
            return spec
    _sys.meta_path.insert(0, _Finder())
