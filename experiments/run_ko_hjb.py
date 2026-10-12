"""
Kim-Omberg: numerical finite-horizon optimum (2-D HJB) executed on the benchmark's
evaluation sets, written in the same raw format so make_tables.py includes it.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from src.agents.hjb_ko import solve_hjb_ko, HJBKOPolicy
from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from run_benchmark import save_raw, metrics, ALPHA, TC, ETA, MARKETS

K = (1 - ETA) / (MARKETS["kim_omberg"]["ell"] + 2 * TC)
GRID = dict(Mx=241, Mth=91, steps_per_day=6)


def main():
    for name, cap in [("HJB optimum (continuous-time)", None), ("HJB optimum (certified class)", K)]:
        pol = HJBKOPolicy(solve_hjb_ko(alpha=ALPHA, K_cap=cap, **GRID), 252, name)
        for k in range(5):
            env = DrawdownPortfolioEnv(make_market("kim_omberg"), batch_size=4000, alpha=ALPHA, tc=TC)
            o = env.reset(10_000 + k)
            for _ in range(252):
                o, *_ = env.step(pol.act(o))
            s = env.summary()
            save_raw("kim_omberg", name, k, s["W_T"], s["max_dd"], s["mean_gross"], s["wealth_path"] if k == 0 else None)
            m = metrics(s["W_T"], s["max_dd"], s["mean_gross"])
            print(f"{name} eval {k}: CE={m['CE']:.4f} viol={100*m['viol']:.2f}% gross={m['gross']:.3f}", flush=True)


if __name__ == "__main__":
    main()
