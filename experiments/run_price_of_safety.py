"""
Price of discrete-time safety (constant-coefficient market).

For a grid of cushion-multiplier caps K, solve the finite-horizon HJB restricted to
|pi| W <= K h (K = 1: the robustly safe class of De Franco-Nicolle-Pham / Hsieh for
long-only portfolios; K = inf: unrestricted continuous-time optimum), then execute each
optimal feedback in the daily discrete-time environment with costs.

Reports, per K and alpha:
  V_K            continuous-time certified optimum (CE units)
  gap_bp         (V_inf - V_K) * 1e4        -- price of the certification constraint
  CE_disc, viol  discrete-time execution (5 evaluation sets x 4000 paths)
  ell_implied    1/K -- the one-period loss the cap protects against (with eta = c = 0)

Output: results/extensions/price_of_safety.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb import solve_hjb, HJBPolicy
from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
KS = [1.0, 1.5, 2.0, 3.0, 4.7, 7.0, 10.0, 15.0, 25.0, 50.0, None]
ALPHAS = [0.10, 0.20]


def ce(W):
    return float(np.mean(W ** -1.0) ** -1.0)


def main():
    out = {}
    for alpha in ALPHAS:
        rows = []
        V_inf = None
        for K in KS:
            sol = solve_hjb(alpha=alpha, K_cap=K)
            if K is None:
                V_inf = sol["ce_continuous"]
            pol = HJBPolicy(sol, 252)
            ces, viols = [], []
            for k in range(5):
                env = DrawdownPortfolioEnv(make_market("constant_bs", num_assets=1), batch_size=4000, alpha=alpha)
                o = env.reset(10_000 + k)
                for _ in range(252):
                    o, *_ = env.step(pol.act(o))
                s = env.summary()
                ces.append(ce(s["W_T"])); viols.append(float(s["violated"].mean()))
            rows.append({"K": K if K is not None else "inf", "V_K": sol["ce_continuous"], "CE_disc": float(np.mean(ces)),
                         "CE_disc_se": float(np.std(ces, ddof=1) / np.sqrt(5)), "viol": float(np.mean(viols))})
            print(f"alpha={alpha:.2f} K={str(K):>5}: V_K={sol['ce_continuous']:.5f}  discrete CE={np.mean(ces):.4f} viol={100*np.mean(viols):.2f}%", flush=True)
        for r in rows:
            r["gap_bp"] = (V_inf - r["V_K"]) * 1e4
        out[str(alpha)] = rows
    os.makedirs(os.path.join(RES, "extensions"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "extensions", "price_of_safety.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
