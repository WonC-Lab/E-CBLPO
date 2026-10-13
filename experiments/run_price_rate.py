"""
Rate of the price of certification, CE(V) - CE(V_K), for large K (graded-mesh HJB).

Prediction (formal boundary-layer scaling, Section 4 of the manuscript):
  r > 0 : the square-root layer {m* > K} has width 4ra/(sigma^2 K^2), the Hamiltonian deficit
          there is O(1/K) and the occupation mass is O(sqrt(width))  =>  price ~ K^{-2};
  r = 0 : the Gaussian layer is exponentially thin except in the last O(1/(sigma^2 K^2)) years
          before maturity, where the cap binds on a region of width O(1/K)  =>  price ~ K^{-3}.
Also checks the layer width directly: the largest y at which the unconstrained multiplier
exceeds K, against 4ra/(sigma^2 K^2).
The time step matters at large K (lagged-control error is first order in dt): run with
SPD=32 and SPD=64 and extrapolate (2 g_64 - g_32).
Output: results/extensions/price_rate_spd{SPD}[_CASE].json (resumable; env CASE selects one case).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
SPD = int(os.environ.get("SPD", 8))
ONLY = os.environ.get("CASE")
OUT = os.path.join(RES, "extensions", f"price_rate_spd{SPD}" + (f"_{ONLY}" if ONLY else "") + ".json")
KS = [10, 15, 20, 30, 50, 70, 100, 150, 200]
CASES = {"r=0.02": dict(r=0.02, mu=0.08), "r=0.005": dict(r=0.005, mu=0.065), "r=0": dict(r=0.0, mu=0.06)}
GRID = dict(M=2000, q=3.5, steps_per_day=SPD)


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, kw in CASES.items():
        if ONLY and name != ONLY:
            continue
        rec = out.setdefault(name, {"params": kw, "K": [], "gap_bp": []})
        if "V" not in rec:
            s = solve_hjb_graded(keep_times=(0.5,), **GRID, **kw)
            rec["V"] = s["ce"]
            m = s["pi"][0.5] * s["x"] / np.where(s["y"] > 0, s["y"], np.nan)
            rec["layer"] = {}
            for K in KS:
                above = s["y"][(m > K) & np.isfinite(m)]
                rec["layer"][str(K)] = float(above.max()) if above.size else 0.0
            json.dump(out, open(OUT, "w"), indent=1)
        for K in KS:
            if K in rec["K"]:
                continue
            ce = solve_hjb_graded(keep_times=(), K_cap=K, **GRID, **kw)["ce"]
            rec["K"].append(K)
            rec["gap_bp"].append((rec["V"] - ce) * 1e4)
            json.dump(out, open(OUT, "w"), indent=1)
            print(name, K, rec["gap_bp"][-1], flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    main()
