"""
E3  Zero-shot transfer on real data: the ECP policies trained in E2 on the nine sector ETFs
    (N = 9, checkpoints results/empirical/E2/checkpoints/<y>_seed<k>.pt) are executed without
    retraining on the 50 largest US common stocks of year y (N = 50).

For each year y in 2004-2025 there is one episode: the trading days of y, starting with
W_0 = H_0 = 1 (the top-50 file contains each year's constituents only for that year).
Coefficients are estimated from the constituents' daily returns over the three previous
years (data/raw/wrds_top50_history/top50_history.csv): Ledoit-Wolf covariance and means
shrunk halfway to the cross-sectional average.  The same estimates are used by every
plug-in strategy.  Missing returns (e.g. after a delisting) are treated as zero.
Output: results/empirical/E3/<year>.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from src.sim.historical import HistoricalPaths
from src.agents.rules import default_rules
from src.agents.pathwise import PathwisePolicy
from run_empirical import RAW, RES, ALPHA, TC, ETA, GAMMA, ELL, ECPAct, run_strategy

HIST = os.path.join(ROOT, "data", "raw", "wrds_top50_history", "top50_history.csv")


def ledoit_wolf(X):
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(X).covariance_
    except ImportError:                                   # constant-correlation shrinkage fallback
        S = np.cov(X, rowvar=False); d = np.sqrt(np.diag(S))
        rbar = (S / np.outer(d, d))[np.triu_indices(len(d), 1)].mean()
        F = rbar * np.outer(d, d); np.fill_diagonal(F, np.diag(S))
        return 0.5 * S + 0.5 * F


def main():
    ff = pd.read_csv(os.path.join(RAW, "ff_daily.csv"), parse_dates=["date"]).set_index("date")["rf"]
    cur = pd.read_csv(os.path.join(RAW, "top50_daily.csv"), parse_dates=["date"])
    top = pd.read_csv(os.path.join(RAW, "top50_by_year.csv"))
    hist = pd.read_csv(HIST, parse_dates=["date"])
    out_dir = os.path.join(RES, "E3"); os.makedirs(out_dir, exist_ok=True)
    for y in range(2004, 2026):
        cks = sorted(glob.glob(os.path.join(RES, "E2", "checkpoints", f"{y}_seed*.pt")))
        if not cks:
            continue
        perm = top[top.year == y].permno.tolist()
        R = cur[cur.date.dt.year == y].pivot(index="date", columns="permno", values="ret").reindex(columns=perm)
        R = R.join(ff, how="inner"); rf = R.pop("rf")
        H = hist[hist.year == y].pivot(index="date", columns="permno", values="ret").reindex(columns=perm)
        H = H.join(ff, how="inner"); rfh = H.pop("rf")
        Xh = (H.values - rfh.values[:, None])
        Xh = np.where(np.isnan(Xh), np.nanmean(Xh, axis=0, keepdims=True), Xh)   # short histories
        Xh = np.nan_to_num(Xh)
        m = Xh.mean(0) * 252
        ex = 0.5 * m + 0.5 * m.mean()
        cov = ledoit_wolf(Xh) * 252
        r_hat = float(rfh.values[-21:].mean() * 252)
        T = len(R)
        mk = lambda: HistoricalPaths(np.nan_to_num(R.values), rf.values, [0], ex, cov, r_hat)
        Xe = np.nan_to_num(R.values) - rf.values[:, None]
        res = {"year": y, "N": len(perm), "T": T, "starts": [str(R.index[0].date())],
               "stress_exceed": [bool((np.abs(Xe) > ELL).any())], "ex": ex.tolist(), "r": r_hat, "strategies": {}}
        import run_empirical
        run_empirical.T = T
        for s in default_rules(GAMMA):
            res["strategies"][s.name] = run_strategy(mk(), s)
        for ck in cks:
            pol = PathwisePolicy(len(perm), 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=GAMMA, r=r_hat, name="ECP")
            pol.actor.load_state_dict(torch.load(ck))
            res["strategies"][f"ECP (seed {ck[-4]})"] = run_strategy(mk(), ECPAct(pol, "ECP"))
        json.dump(res, open(os.path.join(out_dir, f"{y}.json"), "w"))
        print(y, {k: round(v["W_T"][0], 4) for k, v in res["strategies"].items()}, flush=True)


if __name__ == "__main__":
    main()
