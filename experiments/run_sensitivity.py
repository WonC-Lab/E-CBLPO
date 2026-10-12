"""
Sensitivity to risk aversion and horizon (constant-coefficient market, alpha = 20%).

Configurations (gamma, horizon in trading days): (1.5, 252), (3, 252), (5, 252), (2, 126), (2, 504);
(2, 252) is the main benchmark.  For each: GZ, TIPP (m=3), certified and continuous-time HJB
optima (solved for the same gamma and horizon) and E-CBLPO (3 training seeds), evaluated on
3 evaluation sets x 4000 paths with the CE computed at the configuration's gamma.
Output: results/extensions/sensitivity/*.json (resumable)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import GrossmanZhou, TIPP
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.pathwise import PathwisePolicy

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(RES, "extensions", "sensitivity")
ALPHA, ETA, ELL, TC, B, SEEDS = 0.20, 0.05, 0.20, 0.001, 4000, 3
K = (1 - ETA) / (ELL + 2 * TC)
CONFIGS = [(1.5, 252), (3.0, 252), (5.0, 252), (2.0, 126), (2.0, 504)]


def ce(W, g):
    return float(np.mean(np.log(W))) if g == 1 else float(np.mean(W ** (1 - g)) ** (1 / (1 - g)))


def job(kind, g, T, k):
    torch.set_num_threads(1)
    tag = f"g{g}_T{T}"
    m = make_market("constant_bs", num_assets=1)
    if kind == "learned":
        pol = PathwisePolicy(1, 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=g, r=m.r, seed=k, name="E-CBLPO")
        pol.train(m, iters=200, B=512, T=T, seed=k, verbose=False)
        ev = pol.evaluate(m, B, T, 50_000 + k)
        res = {"E-CBLPO": {"CE": ce(ev["W_T"], g), "viol": float(np.mean(ev["max_dd"] > ALPHA + 1e-9)), "gross": float(ev["mean_gross"].mean())}}
        np.save(os.path.join(OUT, f"WT_{tag}_E-CBLPO__{k}.npy"), ev["W_T"])
    else:
        days = T
        pols = [GrossmanZhou(g), TIPP(3.0),
                HJBPolicy(solve_hjb(alpha=ALPHA, gamma=g, T=T / 252, days=days), T, "HJB optimum (continuous-time)"),
                HJBPolicy(solve_hjb(alpha=ALPHA, gamma=g, T=T / 252, days=days, K_cap=K), T, "HJB optimum (certified class)")]
        res = {}
        for p in pols:
            env = DrawdownPortfolioEnv(m, batch_size=B, horizon=T, alpha=ALPHA, tc=TC, gamma=g)
            o = env.reset(50_000 + k)
            for _ in range(T):
                o, *_ = env.step(p.act(o))
            s = env.summary()
            res[p.name] = {"CE": ce(s["W_T"], g), "viol": float(s["violated"].mean()), "gross": float(s["mean_gross"].mean())}
            np.save(os.path.join(OUT, f"WT_{tag}_{p.name.replace(' ', '_')}__{k}.npy"), s["W_T"])
    json.dump(res, open(os.path.join(OUT, f"{tag}__{kind}__{k}.json"), "w"), indent=2)
    return tag, kind, k, res


def run(j):
    return job(*j)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(kind, g, T, k) for (g, T) in CONFIGS for kind in ["rules", "learned"] for k in range(SEEDS)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"g{j[1]}_T{j[2]}__{j[0]}__{j[3]}.json"))]
    with Pool(2) as pool:
        for tag, kind, k, res in pool.imap_unordered(run, jobs):
            print(tag, kind, k, {n: (round(v["CE"], 4), round(100 * v["viol"], 2)) for n, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
