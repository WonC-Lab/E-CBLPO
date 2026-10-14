"""
Item 10: independent check of the primal graded-mesh solver against the dual solver (src/agents/dual_solver.py),
r = 2%, gamma = 2, alpha = 20%, theta = 0.3, T = 1: certainty equivalent, floor marginal value ln v_x(t, a+) and the
optimal multiplier at x - a in {1e-3, 1e-4, 1e-5} at t = 0.5, under refinement of both solvers.
Output: results/extensions/primal_dual.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
from src.agents.dual_solver import solve_dual
YS = [1e-3, 1e-4, 1e-5]
out = {"primal": {}, "dual": {}}
for spd in (8, 32):
    s = solve_hjb_graded(mu=0.08, sigma=0.2, r=0.02, alpha=0.2, gamma=2.0, T=1.0, M=2000, q=3.5, steps_per_day=spd, keep_times=(0.0, 0.5))
    y, x = s["y"], s["x"]
    m = s["pi"][0.5] * x / np.where(y > 0, y, np.nan)
    lnvx = {t: float(np.log(np.interp(1e-7, y, np.gradient(s["v"][t], y)))) for t in (0.0, 0.5)}
    out["primal"][f"spd{spd}"] = {"ce": s["ce"], "ln_vx_a": lnvx, "m": np.interp(YS, y[1:], m[1:]).tolist()}
    print("primal", spd, out["primal"][f"spd{spd}"], flush=True)
for N, spd in ((2000, 8), (4000, 32), (8000, 32)):
    d = solve_dual(N=N, steps_per_day=spd)
    xi = d["xi"]
    Y = d["keep"][0.5]["Y"]
    Yxi = np.gradient(Y, xi)
    mm = (0.3 / 0.2) * np.abs(Yxi) / np.where(Y > 0, Y, np.nan)
    # multiplier as a function of the cushion: interpolate on the decreasing branch Y(xi)
    pos = Y > 0
    order = np.argsort(Y[pos])
    m_at = np.interp(YS, Y[pos][order], mm[pos][order]).tolist()
    out["dual"][f"N{N}_spd{spd}"] = {"ce": d["ce"], "ln_vx_a": {t: float(z) for t, z in d["zeta_a"].items()}, "m": m_at}
    print("dual", N, spd, out["dual"][f"N{N}_spd{spd}"], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "primal_dual.json"), "w"), indent=1)
