"""
Boundary-layer behaviour of the finite-horizon drawdown-constrained optimum (graded-mesh HJB).

(1) r > 0: ratio  pi*(t, a+y) / [ (2/sigma) sqrt(r y / a) ]  for y -> 0, across mu, gamma, sigma, r,
    alpha and t.  Conjectured limit: 1 (independent of mu, gamma, t).
(2) r = 0: slope of  m^2 sigma^2 (T - t)  against ln(1/y), m = pi* x / y the cushion multiplier.
(3) Price of the certification constraint V - V_K for r = 0.02, 0.04, 0.
Output: results/extensions/boundary_layer.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))


def main():
    out = {"r_positive": [], "r_zero": [], "price": {}}
    ys = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    for kw in [{}, {"mu": 0.12}, {"mu": 0.05}, {"gamma": 5.0}, {"gamma": 1.5}, {"sigma": 0.3}, {"r": 0.04}, {"r": 0.005}, {"alpha": 0.10}]:
        s = solve_hjb_graded(M=1600, q=2.5, **kw)
        pred = (2 / kw.get("sigma", 0.2)) * np.sqrt(kw.get("r", 0.02) / s["a"])
        row = {"params": kw, "y": ys.tolist()}
        for tf in (0.1, 0.5, 0.9):
            row[f"ratio_t{tf}"] = (np.interp(ys, s["y"], s["pi"][tf]) / np.sqrt(ys) / pred).tolist()
        out["r_positive"].append(row)
        print("r>0", kw, np.round(row["ratio_t0.5"], 3), flush=True)
    yz = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    for sig in (0.2, 0.3):
        s = solve_hjb_graded(M=3000, q=3.5, r=0.0, mu=0.06, sigma=sig, keep_times=(0.1, 0.3, 0.5, 0.7, 0.9))
        slopes = {}
        for tf in (0.1, 0.3, 0.5, 0.7, 0.9):
            m = np.interp(yz, s["y"], s["pi"][tf]) * (s["a"] + yz) / yz
            slopes[str(tf)] = float(np.polyfit(np.log(1 / yz), m ** 2 * sig ** 2 * (1 - tf), 1)[0])
        out["r_zero"].append({"sigma": sig, "slope_m2_sigma2_tau_vs_log": slopes})
        print("r=0 sigma", sig, {k: round(v, 2) for k, v in slopes.items()}, flush=True)
    # (2b) dual test at r = 0: slope of the multiplier against ln v_x should be 1/(sigma theta tau);
    #      and the floor marginal value v_x(t, a+) is finite for r > 0, infinite for r = 0.
    out["dual_slope_r0"] = []
    for mu, sig in [(0.06, 0.2), (0.06, 0.3), (0.10, 0.2), (0.04, 0.2)]:
        for tau in (0.25, 0.5, 0.9):
            s = solve_hjb_graded(T=tau, days=int(round(tau * 252)), keep_times=(0.0,), M=3000, q=3.5, r=0.0, mu=mu, sigma=sig)
            y = s["y"]; vx = np.gradient(s["v0"], y); m = s["pi"][0.0] * s["x"] / np.where(y > 0, y, np.nan)
            sel = (y > 1e-10) & (y < 1e-5) & np.isfinite(m) & (vx > 0) & (m * y < 1.99)
            lv, mm = np.log(vx[sel]), m[sel]
            k = np.argsort(lv); lv, mm = lv[k], mm[k]
            dl = np.diff(lv); loc = np.diff(mm)[dl > 1e-6] / dl[dl > 1e-6]
            pred = 1 / (sig * (mu / sig) * tau)
            out["dual_slope_r0"].append({"mu": mu, "sigma": sig, "tau": tau, "median_slope": float(np.median(loc)), "prediction": pred})
            print("dual slope", mu, sig, tau, round(float(np.median(loc)) / pred, 3), flush=True)
    out["floor_marginal_value"] = {}
    for r, mu in [(0.02, 0.08), (0.0, 0.06)]:
        s = solve_hjb_graded(M=3000, q=3.5, r=r, mu=mu, keep_times=())
        vx = np.gradient(s["v0"], s["y"])
        out["floor_marginal_value"][str(r)] = {"y": [1e-9, 1e-7, 1e-5, 1e-3], "v_x": np.interp([1e-9, 1e-7, 1e-5, 1e-3], s["y"], vx).tolist()}
        print("v_x near floor, r =", r, np.round(out["floor_marginal_value"][str(r)]["v_x"], 3), flush=True)
    Ks = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50]
    for r, mu in [(0.02, 0.08), (0.04, 0.10), (0.0, 0.06)]:
        kw = dict(M=1600, q=3.0, r=r, mu=mu, keep_times=())
        V = solve_hjb_graded(**kw)["ce"]
        out["price"][str(r)] = {"K": Ks, "gap_bp": [(V - solve_hjb_graded(K_cap=K, **kw)["ce"]) * 1e4 for K in Ks]}
        print("price r", r, np.round(out["price"][str(r)]["gap_bp"], 3), flush=True)
    os.makedirs(os.path.join(RES, "extensions"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "extensions", "boundary_layer.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
