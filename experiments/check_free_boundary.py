"""
Numerical observations (not proofs) for Remark rem:smoothfit / Proposition prop:sqrt, r = 2%:
  (a) the free boundary zeta_a(t) = ln v_x(t, a+) on a grid of dates (is it monotone?);
  (b) the ratio of Y_zetazeta near the free boundary to the square-root value 2ra/theta^2, and the implied
      Y_t / (ra) from the dual equation  Y_t = ra + rY - (nu - r) Y_zeta - nu Y_zetazeta  (condition (7) is Y_t -> 0);
  (c) sup over dates of m(t,x) sqrt(x-a) (Proposition prop:optreg bounds it; the square-root law predicts 2 sqrt(ra)/sigma).
Same solver settings as check_smooth_fit.py. Output: results/extensions/free_boundary_check.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
mu, sig, r, alpha, gam, T = 0.08, 0.2, 0.02, 0.2, 2.0, 1.0
a = 1 - alpha; th = (mu - r) / sig; nu = th ** 2 / 2
TIMES = tuple(np.round(np.linspace(0.0, 0.95, 20), 6))
s = solve_hjb_graded(mu=mu, sigma=sig, r=r, alpha=alpha, gamma=gam, T=T, M=2000, q=3.5, steps_per_day=32, keep_times=TIMES)
y = s["y"]
YS = [1e-4, 1e-5, 1e-6, 1e-7]
out = {"y": YS, "dates": [], "zeta_a": [], "Yzz_ratio": [], "Yt_over_ra": [], "m_sqrt_y": []}
for t in sorted(s["pi"]):
    v = s["v"][t]; vx = np.gradient(v, y); zeta = np.log(vx)
    Yz = np.gradient(y, zeta); Yzz = np.gradient(Yz, zeta)
    yz = np.interp(YS, y, Yz); yzz = np.interp(YS, y, Yzz)
    Yt = r * a + r * np.array(YS) - (nu - r) * yz - nu * yzz
    e = s["pi"][t] * s["x"]
    out["dates"].append(float(t))
    out["zeta_a"].append(float(np.interp(1e-7, y, zeta)))
    out["Yzz_ratio"].append((yzz / (2 * r * a / th ** 2)).tolist())
    out["Yt_over_ra"].append((Yt / (r * a)).tolist())
    out["m_sqrt_y"].append((np.interp(YS, y, e) / np.sqrt(YS)).tolist())
out["sqrt_constant"] = float(2 * np.sqrt(r * a) / sig)
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "free_boundary_check.json"), "w"), indent=1)
for i, t in enumerate(out["dates"]):
    print(f"t={t:.2f} zeta_a={out['zeta_a'][i]:.4f} Yzz/(2ra/th^2)={np.round(out['Yzz_ratio'][i],3)} "
          f"Yt/ra={np.round(out['Yt_over_ra'][i],3)} m*sqrt(y)={np.round(out['m_sqrt_y'][i],3)}")
print("square-root constant", round(out["sqrt_constant"], 4))
