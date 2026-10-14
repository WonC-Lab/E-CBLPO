"""
Item 5 numerics (observations, not proofs): for r = 2% at t in {0.1, 0.5, 0.9},
  (a) the optimal dollar exposure pi* x near the floor and its ratio to sqrt(x-a);
  (b) the sign of Y_zeta_zeta (convexity of the dual cushion in zeta) near the free boundary;
  (c) the time average int_0^{0.9} pi* x dt at fixed x - a (Proposition prop:intsf predicts -> 0).
Output: results/extensions/smooth_fit_check.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
TIMES = tuple(np.round(np.linspace(0.0, 0.9, 19), 6))
s = solve_hjb_graded(mu=0.08, sigma=0.2, r=0.02, alpha=0.2, gamma=2.0, T=1.0, M=2000, q=3.5, steps_per_day=32,
                     keep_times=TIMES)
y, x = s["y"], s["x"]
YS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
out = {"y": YS, "times": {}}
avg = np.zeros(len(YS))
for t in sorted(s["pi"]):
    e = s["pi"][t] * x                       # dollar exposure per unit H
    ev = np.interp(YS, y, e)
    v = s["v"][t]; vx = np.gradient(v, y); zeta = np.log(vx)
    # Y as a function of zeta: Y = y; dY/dzeta and d2Y/dzeta2 by finite differences on the mesh
    Yz = np.gradient(y, zeta); Yzz = np.gradient(Yz, zeta)
    rec = {"exposure": ev.tolist(), "exposure_over_sqrt_y": (ev / np.sqrt(YS)).tolist(),
           "Yzz": np.interp(YS, y, Yzz).tolist()}
    out["times"][str(t)] = rec
    avg += ev / len(s["pi"])
out["time_average_exposure"] = avg.tolist()
out["predicted_sqrt_constant"] = float(2 * np.sqrt(0.02 * 0.8) / 0.2)   # 2 sqrt(r a) / sigma
out["predicted_Yzz"] = float(2 * 0.02 * 0.8 / 0.3 ** 2)            # 2 r a / theta^2
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "smooth_fit_check.json"), "w"), indent=1)
for t in ("0.1", "0.5", "0.9"):
    r_ = out["times"][t]
    print("t", t, "exposure/sqrt(y):", np.round(r_["exposure_over_sqrt_y"], 3), " Yzz:", np.round(r_["Yzz"], 4))
print("time-average exposure:", np.round(avg, 6))
