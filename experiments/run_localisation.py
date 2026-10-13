"""
Item 5: accuracy of the fixed-floor (OBPI) localisation of Theorem thm:gauss at r = 0.
For each setting (mu, sigma, tau) and cushion y = x - a, compare the HJB multiplier m = pi x / y with the
explicit m_A = (theta/sigma) (-A_zeta / A) evaluated at the HJB's zeta = ln v_x(t, x), and the explicit
cushion A(tau, zeta) with y. Includes horizons close to maturity (tau = 0.1, 0.05, 0.02).
Rows tau in {0.5, 0.9, 0.25} reproduce Table tab:gauss.  Output: results/extensions/localisation.json
"""
import json, os, sys
import numpy as np
from scipy.special import erfcx
from scipy.optimize import brentq
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
OUT = os.path.join(ROOT, "results", "extensions", "localisation.json")
gam, alpha, T = 2.0, 0.2, 1.0
a = 1 - alpha
YS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
SETTINGS = [(0.06, 0.20, (0.5, 0.1, 0.05, 0.02)), (0.06, 0.30, (0.9,)), (0.10, 0.20, (0.25,))]
SPD = [int(v) for v in os.environ.get("SPD", "8").split(",")]     # time steps per day (convergence check)
R = lambda z: np.sqrt(np.pi / 2) * erfcx(z / np.sqrt(2))     # Mills ratio
phi = lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

def fixed_floor(zeta, tau, theta):
    nu, s = theta ** 2 / 2, theta * np.sqrt(tau)
    w = (zeta + nu * tau - gam * np.log(1 / a)) / s
    eps = s / gam
    A = a * phi(w) * (R(w - eps) - R(w))
    Az = -(a / gam) * phi(w) * R(w - eps)
    return A, -Az / A, w

def m_of_y(yy, tau, theta, sig):
    """fully explicit: solve A(tau, zeta) = y for zeta, return (theta/sigma)(-A_zeta/A) there"""
    g = lambda z: np.log(fixed_floor(z, tau, theta)[0]) - np.log(yy)
    lo = gam * np.log(1 / a) - theta ** 2 / 2 * tau
    z = brentq(g, lo - 3 * theta * np.sqrt(tau), lo + 60 * theta * np.sqrt(tau))
    return float(theta / sig * fixed_floor(z, tau, theta)[1])

out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for spd, (mu, sig, taus) in [(d, st) for d in SPD for st in SETTINGS]:
    theta = mu / sig
    s = solve_hjb_graded(mu=mu, sigma=sig, r=0.0, alpha=alpha, gamma=gam, T=T, M=2000, q=3.5, steps_per_day=spd,
                         keep_times=tuple(round(T - t, 10) for t in taus))
    y, x = s["y"], s["x"]
    for tau in taus:
        f = round(T - tau, 10)
        pi, v = s["pi"][f], s["v"][f]
        m = pi * x / np.where(y > 0, y, np.nan)
        zeta = np.log(np.gradient(v, y))
        rec = {"y": YS, "m_hjb": [], "m_A": [], "A_over_y": [], "w": [], "m_explicit": []}
        for yy in YS:
            z = float(np.interp(yy, y, zeta))
            A, ratio, w = fixed_floor(z, tau, theta)
            rec["m_hjb"].append(float(np.interp(yy, y[1:], m[1:])))
            rec["m_A"].append(float(theta / sig * ratio))
            rec["A_over_y"].append(float(A / yy)); rec["w"].append(float(w))
            rec["m_explicit"].append(m_of_y(yy, tau, theta, sig))
        key = f"mu={mu},sigma={sig},tau={tau},spd={spd}"
        out[key] = rec
        print(key)
        for i, yy in enumerate(YS):
            print(f"  y={yy:.0e} w={rec['w'][i]:.2f} m_hjb={rec['m_hjb'][i]:.2f} m_A={rec['m_A'][i]:.2f} "
                  f"rel={rec['m_hjb'][i]/rec['m_A'][i]-1:+.4f} A/y-1={rec['A_over_y'][i]-1:+.4f} "
                  f"m_expl={rec['m_explicit'][i]:.2f} rel_expl={rec['m_hjb'][i]/rec['m_explicit'][i]-1:+.4f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
