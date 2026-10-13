"""
Optimum of the robustly safe class (cushion multiplier at most one; De Franco-Nicolle-Pham
2020, Hsieh 2026), computed with the same HJB solvers as the other numerical benchmarks
(K_cap = 1) and executed on the benchmark's evaluation sets.  Written in the raw format so
make_tables.py includes it as "HJB optimum (robust class, K=1)".
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.hjb_ko import solve_hjb_ko, HJBKOPolicy
from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from run_benchmark import save_raw, metrics, ALPHA, TC, RHO, MARKETS
from run_ko_hjb import GRID as KO_GRID

NAME = "HJB optimum (robust class, K=1)"


def run(market, pol):
    for k in range(5):
        env = DrawdownPortfolioEnv(make_market(market, **MARKETS[market]["kw"]), batch_size=4000, alpha=ALPHA, tc=TC)
        o = env.reset(10_000 + k)
        for _ in range(252):
            o, *_ = env.step(pol.act(o))
        s = env.summary()
        save_raw(market, NAME, k, s["W_T"], s["max_dd"], s["mean_gross"], s["wealth_path"] if k == 0 else None)
        m = metrics(s["W_T"], s["max_dd"], s["mean_gross"])
        print(f"{market} eval {k}: CE={m['CE']:.4f} viol={100*m['viol']:.2f}% gross={m['gross']:.3f}", flush=True)


def main():
    run("constant_bs", HJBPolicy(solve_hjb(alpha=ALPHA, gamma=RHO, K_cap=1.0), 252, NAME))
    run("kim_omberg", HJBKOPolicy(solve_hjb_ko(alpha=ALPHA, K_cap=1.0, **KO_GRID), 252, NAME))


if __name__ == "__main__":
    main()
