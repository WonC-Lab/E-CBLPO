"""
Estimation risk: the investor observes only a noisy estimate of the drift.

Market `estimated_bs`: per path mu ~ N(0.08, 0.03^2) (prior), observed mu_hat = mu + N(0, 0.045^2)
(about 20 years of data at 20% volatility).  Posterior weight on mu_hat: w = 0.31.

Strategies (all evaluated on identical paths, 3 evaluation sets x 4000 paths):
  HJB certified, oracle      -- uses the true mu (infeasible reference)
  HJB certified, plug-in     -- plugs mu_hat into the HJB solution (naive)
  HJB certified, Bayes       -- plugs the posterior mean mu0 + w (mu_hat - mu0) (knows the prior)
  GZ / Merton plug-in (mu_hat), TIPP (m=3, mu-free), cash
  E-CBLPO                    -- trained on the same market, sees only mu_hat (3 seeds)
Output: results/extensions/estimation/*.json (resumable)
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
from src.agents.rules import GrossmanZhou, TIPP, MertonMyopic, Cash
from src.agents.hjb import solve_hjb
from src.agents.pathwise import PathwisePolicy

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(RES, "extensions", "estimation")
ALPHA, ETA, ELL, TC, T, B, SEEDS = 0.20, 0.05, 0.20, 0.001, 252, 4000, 3
K = (1 - ETA) / (ELL + 2 * TC)
R = 0.02
PGRID = np.round(np.arange(-0.10, 0.2501, 0.01), 4)


class HJBGridPolicy:
    """Certified HJB optimum, solved on a grid of excess drifts and interpolated in the drift."""

    def __init__(self, table, x, mode, mkt, name):
        self.tab, self.x, self.mode, self.mkt, self.name = table, x, mode, mkt, name

    def premium(self, o):
        if self.mode == "oracle":
            return o["oracle_ex"][:, 0]
        p_hat = o["ex"][:, 0]
        if self.mode == "plugin":
            return p_hat
        p0 = self.mkt.mu0 - R
        return p0 + self.mkt.shrink * (p_hat - p0)

    def act(self, o):
        n = min(int(round(o["t_frac"][0] * T)), self.tab.shape[1] - 1)
        p = np.clip(self.premium(o), PGRID[0], PGRID[-1])
        xr = np.clip(o["W"] / o["H"], self.x[0], 1.0)
        g = np.clip(np.searchsorted(PGRID, p) - 1, 0, len(PGRID) - 2)
        w = (p - PGRID[g]) / (PGRID[g + 1] - PGRID[g])
        lo = np.array([np.interp(xr[b], self.x, self.tab[g[b], n]) for b in range(len(xr))])
        hi = np.array([np.interp(xr[b], self.x, self.tab[g[b] + 1, n]) for b in range(len(xr))])
        return ((1 - w) * lo + w * hi)[:, None]


def hjb_table():
    f = os.path.join(OUT, "hjb_table.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["tab"], d["x"]
    tabs = []
    for p in PGRID:
        sol = solve_hjb(mu=R + p, alpha=ALPHA, K_cap=K, M=201, steps_per_day=8)
        tabs.append(sol["pi"][::sol["steps_per_day"]])        # daily slices
    tab = np.stack(tabs); np.savez_compressed(f, tab=tab, x=sol["x"])
    return tab, sol["x"]


def ce(W):
    return float(np.mean(W ** -1.0) ** -1.0)


def met(W, dd, g):
    return {"CE": ce(W), "viol": float(np.mean(dd > ALPHA + 1e-9)), "gross": float(np.mean(g)),
            "W_T": None}


def job(kind, k):
    torch.set_num_threads(1)
    path = os.path.join(OUT, f"{kind}__{k}.json")
    mkt = make_market("estimated_bs")
    if kind == "learned":
        pol = PathwisePolicy(1, 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, r=R, seed=k, name="E-CBLPO")
        pol.train(mkt, iters=200, B=512, T=T, seed=k, verbose=False)
        ev = pol.evaluate(mkt, B, T, 40_000 + k)
        res = {"E-CBLPO": {"CE": ce(ev["W_T"]), "viol": float(np.mean(ev["max_dd"] > ALPHA + 1e-9)), "gross": float(ev["mean_gross"].mean())}}
        np.save(os.path.join(OUT, f"WT_E-CBLPO__{k}.npy"), ev["W_T"])
    else:
        tab, x = hjb_table()
        pols = [HJBGridPolicy(tab, x, "oracle", mkt, "HJB certified (oracle mu)"),
                HJBGridPolicy(tab, x, "plugin", mkt, "HJB certified (plug-in mu_hat)"),
                HJBGridPolicy(tab, x, "bayes", mkt, "HJB certified (Bayes posterior mean)"),
                GrossmanZhou(2.0), MertonMyopic(2.0), TIPP(3.0), Cash()]
        res = {}
        for p in pols:
            env = DrawdownPortfolioEnv(make_market("estimated_bs"), batch_size=B, alpha=ALPHA, tc=TC)
            o = env.reset(40_000 + k)
            for _ in range(T):
                o, *_ = env.step(p.act(o))
            s = env.summary()
            res[p.name] = {"CE": ce(s["W_T"]), "viol": float(s["violated"].mean()), "gross": float(s["mean_gross"].mean())}
            np.save(os.path.join(OUT, f"WT_{p.name.replace(' ', '_')}__{k}.npy"), s["W_T"])
    json.dump(res, open(path, "w"), indent=2)
    return kind, k, res


def run(j):
    return job(*j)


def main():
    os.makedirs(OUT, exist_ok=True)
    hjb_table()
    jobs = [(kind, k) for kind in ["learned", "rules"] for k in range(SEEDS)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}__{j[1]}.json"))]
    with Pool(2) as pool:
        for kind, k, res in pool.imap_unordered(run, jobs):
            print(kind, k, {n: (round(v["CE"], 4), round(100 * v["viol"], 2)) for n, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
