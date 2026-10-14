"""
Numerical sanity check of Proposition prop:floor (floor equation) and of the one-sided boundary
inequality v_x(t,1-) >= (1-gamma) v(t,1) (Lemma lem:g): for r in {2%, 0.5%}, compare
v(t,a) - v(T,a) with r a int_t^T v_x(u,a+) du, with v_x(u,a+) read at x - a = 1e-7 from the graded solver.
This is a consistency check of the solver against a proved identity, not evidence for the identity.
Output: results/extensions/floor_equation_check.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
out = {}
TIMES = tuple(np.round(np.linspace(0, 1, 41), 6))
for r, mu in [(0.02, 0.08), (0.005, 0.065)]:
    s = solve_hjb_graded(mu=mu, sigma=0.2, r=r, alpha=0.2, gamma=2.0, T=1.0, M=2000, q=3.5, steps_per_day=32, keep_times=TIMES)
    y, a = s["y"], s["a"]
    ts = sorted(s["v"])
    va = np.array([s["v"][t][0] for t in ts])
    vxa = np.array([np.interp(1e-7, y, np.gradient(s["v"][t], y)) for t in ts])
    v1 = np.array([s["v"][t][-1] for t in ts])
    vx1 = np.array([(s["v"][t][-1] - s["v"][t][-2]) / (y[-1] - y[-2]) for t in ts])
    UA = a ** (1 - 2.0) / (1 - 2.0)
    lhs = va - UA                                    # v(t,a) - v(T,a), v(T,a) = U(a)
    # trapezoid of r a v_x(u,a+) from t to T (the last stored time is the step before T; append T with U'(a))
    tt = np.r_[ts, 1.0]; g = np.r_[vxa, a ** -2.0] * r * a
    rhs = np.array([np.trapezoid(g[i:], tt[i:]) for i in range(len(ts))])
    out[f"r={r}"] = {"t": list(map(float, ts)), "lhs": lhs.tolist(), "rhs": rhs.tolist(),
                     "max_rel_err": float(np.max(np.abs(lhs - rhs)[:-1] / np.abs(rhs[:-1]))),
                     "boundary_gap_min": float(np.min(vx1 - (1 - 2.0) * v1))}
    print(f"r={r}: max relative error of the floor equation {out[f'r={r}']['max_rel_err']:.4f};"
          f" min of v_x(t,1-)-(1-gamma)v(t,1) = {out[f'r={r}']['boundary_gap_min']:.2e}")
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "floor_equation_check.json"), "w"), indent=1)
