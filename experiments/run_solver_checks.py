"""
Reliability of the graded-mesh HJB solver near the floor (items: grid, time step, control
bounds, independent time discretisation).

Part A  (PART=A): optimal cushion multiplier m = pi* x / (x - a) at t = T/2 for cushions
        y = x - a in {1e-3, 1e-5, 1e-7, 1e-9, 1e-11}, and the floor marginal value
        v_x(0, a + 1e-9), for r = 2% and r = 0, under
          base      M=2000, q=3.5, 8 steps/day, control lagged one step
          M4000     spatial refinement
          q3, q4    different grading of the mesh
          spd32     temporal refinement
          howard2   control recomputed implicitly (2 policy-iteration sweeps per step)
          pimax4    control bound pi_max = 4 instead of 2
Part B  (PART=B): price of certification CE(V) - CE(V_K), K in {50, 100, 200}, for r = 2% and
        r = 0, under spd32 lagged, spd32 howard1 and M4000 spd32, to separate the gap from
        discretisation error (compare with the spd32 / spd64 runs of run_price_rate.py).
Output: results/extensions/solver_checks_<PART>.json (resumable).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded

RES = os.path.join(os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results")), "extensions")
PART = os.environ.get("PART", "A")
OUT = os.path.join(RES, f"solver_checks_{PART}.json")
CASES = {"r=0.02": dict(r=0.02, mu=0.08), "r=0": dict(r=0.0, mu=0.06)}
YS = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
BASE = dict(M=2000, q=3.5, steps_per_day=8)
VARIANTS_A = {"base": {}, "M4000": dict(M=4000), "q3": dict(q=3.0), "q4": dict(q=4.0), "spd32": dict(steps_per_day=32),
              "howard2": dict(howard=2), "pimax4": dict(pi_max=4.0)}
VARIANTS_B = {"spd32": dict(steps_per_day=32), "spd32_howard1": dict(steps_per_day=32, howard=1),
              "M4000_spd32": dict(M=4000, steps_per_day=32)}
KS = [50, 100, 200]


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for cname, ckw in CASES.items():
        if PART == "A":
            for vname, vkw in VARIANTS_A.items():
                key = f"{cname}|{vname}"
                if key in out:
                    continue
                kw = {**BASE, **vkw, **ckw}
                s = solve_hjb_graded(keep_times=(0.5,), **kw)
                y, x, pi = s["y"], s["x"], s["pi"][0.5]
                m = pi * x / np.where(y > 0, y, np.nan)
                vx = np.gradient(s["v0"], y)
                out[key] = {"m": np.interp(YS, y[1:], m[1:]).tolist(), "vx_1e-9": float(np.interp(1e-9, y, vx)),
                            "ce": s["ce"], "min_dy": float(y[1] - y[0])}
                json.dump(out, open(OUT, "w"), indent=1)
                print(key, np.round(out[key]["m"], 3), round(out[key]["vx_1e-9"], 4), flush=True)
        else:
            for vname, vkw in VARIANTS_B.items():
                key = f"{cname}|{vname}"
                rec = out.setdefault(key, {"K": [], "gap_bp": []})
                kw = {**BASE, **vkw, **ckw}
                if "V" not in rec:
                    rec["V"] = solve_hjb_graded(keep_times=(), **kw)["ce"]
                    json.dump(out, open(OUT, "w"), indent=1)
                for K in KS:
                    if K in rec["K"]:
                        continue
                    rec["K"].append(K)
                    rec["gap_bp"].append((rec["V"] - solve_hjb_graded(keep_times=(), K_cap=K, **kw)["ce"]) * 1e4)
                    json.dump(out, open(OUT, "w"), indent=1)
                    print(key, K, rec["gap_bp"][-1], flush=True)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    main()
