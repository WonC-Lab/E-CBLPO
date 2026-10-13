"""
Empirical study on CRSP data (out of sample, no look-ahead).

E1  US equity market, 1 asset: CRSP value-weighted market (Fama-French Mkt-RF + RF), with
    the realised daily risk-free rate as the cash leg.  Evaluation years 1982-2025.
E2  Nine Select Sector SPDR ETFs (N = 9).  Evaluation years 2004-2025.

For every evaluation year y:
  * estimation window: the W years before 1 January y (E1: 20 years; E2: 10 years, at least 5);
    annualised mean excess return and covariance are estimated from it (plug-in);
    the cash rate r is the average risk-free rate of the last 21 trading days of the window;
  * episodes: one-year (252-day) investments starting on the first trading day of each month
    of y, with W_0 = H_0 = 1, alpha = 20%, 10 bp costs, gamma = 2 (as in the simulations);
  * strategies: cash, 60% mix, Merton, CPPI, TIPP, Grossman-Zhou (plug-in); in E1 also the
    finite-horizon HJB optimum (continuous-time, certified K = 4.7, robust K = 1) with the
    plug-in coefficients; ECP trained on an i.i.d. bootstrap of the window's daily excess
    returns (3 seeds), then executed on the realised paths.  The i.i.d. bootstrap keeps the
    marginal distribution (fat tails, crash days) but not serial dependence: with a block
    bootstrap the policy learns to trade on the short-horizon autocorrelation of the pre-1990
    index (non-synchronous trading), which does not persist out of sample (EMP_BLOCK=20).
    Training uses detach_prev=True and guard=True (see PathwisePolicy): on bootstrap data the
    position recurrence can make the 252-step gradient explode.
Raw CRSP files are read from data/raw/wrds/ (licensed; not in the repository).
Output: results/empirical/<exp>/<year>.json (resumable), aggregated by make_empirical_tables.py.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.sim.historical import HistoricalPaths, BootstrapMarket, estimate
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import default_rules
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.pathwise import PathwisePolicy

RAW = os.path.join(ROOT, "data", "raw", "wrds")
RES = os.path.join(os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results")), "empirical")
ALPHA, TC, ETA, GAMMA, ELL, T = 0.20, 0.001, 0.05, 2.0, 0.20, 252
K_CERT = (1 - ETA) / (ELL + 2 * TC)
BLOCK = int(os.environ.get("EMP_BLOCK", 1))   # 1 = i.i.d. bootstrap of daily excess returns
ITERS, BATCH, SEEDS = int(os.environ.get("EMP_ITERS", 150)), 512, (0, 1, 2)


def load(exp):
    ff = pd.read_csv(os.path.join(RAW, "ff_daily.csv"), parse_dates=["date"]).set_index("date").sort_index()
    if exp == "E1":
        R = (ff["mktrf"] + ff["rf"]).to_frame("MKT")
        return R, ff["rf"], range(1982, 2026), 20, 20
    etf = pd.read_csv(os.path.join(RAW, "sector_etf_daily.csv"), parse_dates=["date"])
    R = etf.pivot(index="date", columns="ticker", values="ret").sort_index().dropna()
    R = R.join(ff["rf"], how="inner")
    rf = R.pop("rf")
    return R, rf, range(2004, 2026), 10, 5


class ECPAct:
    """Execute a trained PathwisePolicy inside DrawdownPortfolioEnv."""

    def __init__(self, pol, name):
        self.pol, self.name = pol, name

    @torch.no_grad()
    def act(self, o):
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        u = self.pol.weights(f(o["ex"]), f(o["vol"]), f(o["merton_dir"]), f(o["mkt"]), f(o["W"]), f(o["H"]),
                             f(o["u_prev"]), float(o["t_frac"][0]))
        return u.numpy().astype(np.float64)


def run_strategy(market, strat):
    env = DrawdownPortfolioEnv(market, batch_size=len(market.starts), horizon=T, alpha=ALPHA, tc=TC, gamma=GAMMA)
    o = env.reset(0)
    turnover = np.zeros(env.B)
    for _ in range(T):
        prev = env.u_prev.copy()
        o, *_ = env.step(strat.act(o))
        turnover += np.abs(env.u_prev - prev).sum(1)
    s = env.summary()
    return {"W_T": s["W_T"].tolist(), "max_dd": s["max_dd"].tolist(), "gross": s["mean_gross"].tolist(),
            "turnover": turnover.tolist()}


def job(args):
    exp, y = args
    out_f = os.path.join(RES, exp, f"{y}.json")
    if os.path.exists(out_f):
        return exp, y, "cached"
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)      # subnormal floats can slow training by orders of magnitude
    t0 = time.time()
    R, rf, _, wy, wmin = load(exp)
    dates = R.index
    win = (dates >= pd.Timestamp(f"{y - wy}-01-01")) & (dates < pd.Timestamp(f"{y}-01-01"))
    if win.sum() < wmin * 250:
        return exp, y, "window too short"
    X = (R[win].values - rf[win].values[:, None])
    ex, cov = estimate(X)
    r_hat = float(rf[win].values[-21:].mean() * 252)
    Rv, rfv = R.values, rf.values
    ys = np.where(dates.year == y)[0]
    starts = [ys[dates[ys].month == mth][0] for mth in range(1, 13) if (dates[ys].month == mth).any()]
    starts = np.array([s for s in starts if s + T <= len(dates)])
    if len(starts) == 0:
        return exp, y, "no complete episode"
    mk = lambda: HistoricalPaths(Rv, rfv, starts, ex, cov, r_hat)
    res = {"year": y, "starts": [str(dates[s].date()) for s in starts], "ex": ex.tolist(),
           "vol": np.sqrt(np.diag(cov)).tolist(), "r": r_hat,
           "stress_exceed": [bool((np.abs(Rv[s:s + T] - rfv[s:s + T, None]) > ELL).any()) for s in starts],
           "strategies": {}}
    strats = default_rules(GAMMA)
    if R.shape[1] == 1:
        mu, sig = r_hat + float(ex[0]), float(np.sqrt(cov[0, 0]))
        for name, cap in [("HJB optimum (continuous-time)", None), ("HJB optimum (certified class)", K_CERT),
                          ("HJB optimum (robust class, K=1)", 1.0)]:
            strats.append(HJBPolicy(solve_hjb(mu=mu, sigma=sig, r=r_hat, alpha=ALPHA, gamma=GAMMA, K_cap=cap), T, name))
    for s in strats:
        res["strategies"][s.name] = run_strategy(mk(), s)
    for seed in SEEDS:
        pol = PathwisePolicy(R.shape[1], 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=GAMMA, r=r_hat, seed=seed,
                             name="ECP", detach_prev=True)
        hist = pol.train(BootstrapMarket(X, ex, cov, r_hat, mean_block=BLOCK), iters=ITERS, B=BATCH, T=T, seed=1000 * y + seed, verbose=False,
                         guard=True)
        res.setdefault("ecp_skipped_updates", []).append(int(sum(h["skipped"] for h in hist)))
        res["strategies"][f"ECP (seed {seed})"] = run_strategy(mk(), ECPAct(pol, "ECP"))
        ck = os.path.join(RES, exp, "checkpoints"); os.makedirs(ck, exist_ok=True)
        torch.save(pol.actor.state_dict(), os.path.join(ck, f"{y}_seed{seed}.pt"))
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    json.dump(res, open(out_f, "w"))
    return exp, y, f"{res['seconds']:.0f}s"


def main():
    exps = sys.argv[1:] or ["E1", "E2"]
    jobs = [(e, y) for e in exps for y in load(e)[2]]
    with Pool(int(os.environ.get("WORKERS", 2))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
