"""
Item 7: behaviour of the floor marginal value v_x(t, a+) as r -> 0 (theta = 0.3 fixed, mu = r + 0.06),
compared with the explicit bound of Theorem thm:uniform. v_x is read at y = x - a = 1e-7 (the solver
checks of run_solver_checks.py show the multiplier is grid-stable down to y = 1e-7) and at t = 0, T/2.
Output: results/extensions/r_scaling.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
OUT = os.path.join(ROOT, "results", "extensions", "r_scaling.json")
RS = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.0]
sig, gam, alpha, T, theta = 0.2, 2.0, 0.2, 1.0, 0.3
a = 1 - alpha

def bound(r):
    nu = theta ** 2 / 2
    A = r * a / (2 * nu + max(r - nu, 0) ** 2 / r)
    L = np.sqrt((1 - a) / A)
    U = lambda x: x ** (1 - gam) / (1 - gam)
    B = (max(U(1.0), U(1.0) * np.exp((1 - gam) * (r + theta ** 2 / (2 * gam)) * T)) - U(a)) / (1 - a)
    zL = max(gam * np.log(1 / a), np.log(B))
    return dict(A=A, L=L, zeta_L=zL, bound_ln_vx=zL + L)

out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for r in RS:
    key = f"{r:g}"
    if key in out:
        continue
    s = solve_hjb_graded(mu=r + theta * sig, sigma=sig, r=r, alpha=alpha, gamma=gam, T=T, M=2000, q=3.5,
                         steps_per_day=8, keep_times=(0.0, 0.5))
    y = s["y"]
    vx = np.gradient(s["v0"], y)
    rec = {"ln_vx_1e-7": float(np.log(np.interp(1e-7, y, vx))), "ln_vx_1e-5": float(np.log(np.interp(1e-5, y, vx))),
           "ln_vx_1e-3": float(np.log(np.interp(1e-3, y, vx)))}
    if r > 0:
        rec.update(bound(r))
    out[key] = rec
    json.dump(out, open(OUT, "w"), indent=1)
    print(key, rec, flush=True)
