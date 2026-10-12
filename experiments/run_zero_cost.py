"""
Optimality gap without transaction costs.

With c = 0 the HJB solutions are the exact (continuous-time) optima of the problem the
learned policy is trained on, so  CE(HJB executed daily) - CE(E-CBLPO)  isolates the
optimisation gap of learning, and the HJB comparison is not confounded by costs.

Markets: constant_bs (1-D HJB) and kim_omberg (2-D HJB).  E-CBLPO: 3 training seeds,
evaluated on evaluation sets 10_000 + k (k = seed), 4000 paths, c = 0 everywhere.
Output: results/extensions/zero_cost/*.json (one file per job, resumable).
"""
import glob
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
from src.agents.hjb_ko import solve_hjb_ko, HJBKOPolicy
from src.agents.pathwise import PathwisePolicy

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(RES, "extensions", "zero_cost")
ALPHA, ETA, ELL, T, B = 0.20, 0.05, 0.20, 252, 4000
K = (1 - ETA) / ELL
KW = {"constant_bs": {"num_assets": 1}, "kim_omberg": {}}
SEEDS = 3


def ce(W):
    return float(np.mean(W ** -1.0) ** -1.0)


def met(W, dd, g):
    return {"CE": ce(W), "viol": float(np.mean(dd > ALPHA + 1e-9)), "gross": float(np.mean(g))}


def job(kind, market, k):
    torch.set_num_threads(1)
    path = os.path.join(OUT, f"{market}__{kind}__{k}.json")
    if kind == "learned":
        m = make_market(market, **KW[market])
        fdim = 3 + m.features(m.reset(1, np.random.default_rng(0))).shape[-1]
        pol = PathwisePolicy(m.N, fdim, alpha=ALPHA, ell=ELL, tc=0.0, eta=ETA, r=m.r, seed=k, name="E-CBLPO")
        pol.train(m, iters=200, B=512, T=T, seed=k, verbose=False)
        ev = pol.evaluate(m, B, T, 10_000 + k)
        res = {"E-CBLPO": met(ev["W_T"], ev["max_dd"], ev["mean_gross"])}
    else:
        if market == "constant_bs":
            pols = [HJBPolicy(solve_hjb(alpha=ALPHA), T, "HJB optimum (continuous-time)"),
                    HJBPolicy(solve_hjb(alpha=ALPHA, K_cap=K), T, "HJB optimum (certified class)")]
        else:
            g = dict(Mx=241, Mth=91, steps_per_day=6)
            pols = [HJBKOPolicy(solve_hjb_ko(alpha=ALPHA, **g), T, "HJB optimum (continuous-time)"),
                    HJBKOPolicy(solve_hjb_ko(alpha=ALPHA, K_cap=K, **g), T, "HJB optimum (certified class)")]
        pols += [GrossmanZhou(2.0), TIPP(3.0)]
        res = {}
        for p in pols:
            env = DrawdownPortfolioEnv(make_market(market, **KW[market]), batch_size=B, alpha=ALPHA, tc=0.0)
            o = env.reset(10_000 + k)
            for _ in range(T):
                o, *_ = env.step(p.act(o))
            s = env.summary()
            res[p.name] = met(s["W_T"], s["max_dd"], s["mean_gross"])
    json.dump(res, open(path, "w"), indent=2)
    return market, kind, k, res


def run(j):
    return job(*j)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(kind, m, k) for m in KW for kind in ["learned", "rules"] for k in range(SEEDS)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[1]}__{j[0]}__{j[2]}.json"))]
    with Pool(2) as pool:
        for market, kind, k, res in pool.imap_unordered(run, jobs):
            print(market, kind, k, {n: round(v["CE"], 4) for n, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
