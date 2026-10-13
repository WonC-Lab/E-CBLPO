"""
Item 15 diagnostics on E1 (why TIPP beats ECP / the certified HJB out of sample).

For every evaluation year and the same monthly episodes as run_empirical.py:
  zero_cost  all rules, the plug-in HJB policies and the trained ECP checkpoints re-executed with
             transaction costs set to 0 (no retraining): separates trading costs from the rest;
  oracle     look-ahead diagnostic: Grossman-Zhou, Merton and the HJB policies (continuous-time and
             certified) given the episode's realised annualised excess drift and volatility instead
             of the plug-in estimates (10 bp costs). TIPP and CPPI use no estimates and are unchanged.
             ECP is not evaluated with oracle inputs: each year's network is trained at one fixed
             (plug-in) drift, so it has never seen the drift input vary.
Output: results/empirical/diagnostics/E1/<year>.json; aggregated by make_empirical_diagnostics.py.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from run_empirical import load, ECPAct, ALPHA, TC, ETA, GAMMA, ELL, T, K_CERT, SEEDS, RES
from src.sim.historical import HistoricalPaths, estimate
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import default_rules
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.pathwise import PathwisePolicy
OUT = os.path.join(RES, "diagnostics", "E1")


def run(market, strat, tc):
    env = DrawdownPortfolioEnv(market, batch_size=len(market.starts), horizon=T, alpha=ALPHA, tc=tc, gamma=GAMMA)
    o = env.reset(0)
    for _ in range(T):
        o, *_ = env.step(strat.act(o))
    s = env.summary()
    return {"W_T": s["W_T"].tolist(), "max_dd": s["max_dd"].tolist(), "gross": s["mean_gross"].tolist()}


def hjb_pols(mu, sig, r):
    return [HJBPolicy(solve_hjb(mu=mu, sigma=sig, r=r, alpha=ALPHA, gamma=GAMMA, K_cap=cap), T, name)
            for name, cap in [("HJB optimum (continuous-time)", None), ("HJB optimum (certified class)", K_CERT)]]


def job(y):
    f = os.path.join(OUT, f"{y}.json")
    if os.path.exists(f):
        return y, "cached"
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    base = json.load(open(os.path.join(RES, "E1", f"{y}.json")))
    R, rf, _, wy, _ = load("E1")
    dates = R.index
    win = (dates >= pd.Timestamp(f"{y - wy}-01-01")) & (dates < pd.Timestamp(f"{y}-01-01"))
    X = R[win].values - rf[win].values[:, None]
    ex, cov = estimate(X)
    r_hat = float(rf[win].values[-21:].mean() * 252)
    assert abs(r_hat - base["r"]) < 1e-12
    Rv, rfv = R.values, rf.values
    starts = np.array([dates.get_loc(pd.Timestamp(s)) for s in base["starts"]])
    mk = lambda st, e, c: HistoricalPaths(Rv, rfv, st, e, c, r_hat)
    out = {"year": y, "starts": base["starts"], "zero_cost": {}, "oracle": {}, "oracle_coeffs": []}
    mu, sig = r_hat + float(ex[0]), float(np.sqrt(cov[0, 0]))
    for s in default_rules(GAMMA) + hjb_pols(mu, sig, r_hat):
        out["zero_cost"][s.name] = run(mk(starts, ex, cov), s, 0.0)
    for seed in SEEDS:
        pol = PathwisePolicy(1, 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=GAMMA, r=r_hat, seed=seed, name="ECP",
                             detach_prev=True)
        pol.actor.load_state_dict(torch.load(os.path.join(RES, "E1", "checkpoints", f"{y}_seed{seed}.pt")))
        out["zero_cost"][f"ECP (seed {seed})"] = run(mk(starts, ex, cov), ECPAct(pol, "ECP"), 0.0)
    for s0 in starts:
        Xo = Rv[s0:s0 + T] - rfv[s0:s0 + T, None]
        eo, co = estimate(Xo)
        out["oracle_coeffs"].append([float(eo[0]), float(np.sqrt(co[0, 0]))])
        rules = [r_ for r_ in default_rules(GAMMA) if r_.name in ("Grossman-Zhou", "Merton (unconstrained)")]
        for s in rules + hjb_pols(r_hat + float(eo[0]), float(np.sqrt(co[0, 0])), r_hat):
            res = run(mk(np.array([s0]), eo, co), s, TC)
            for k, v in res.items():
                out["oracle"].setdefault(s.name, {}).setdefault(k, []).extend(v)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f, "w"))
    return y, "done"


if __name__ == "__main__":
    with Pool(int(os.environ.get("WORKERS", 4))) as p:
        for r in p.imap_unordered(job, list(load("E1")[2])):
            print(*r, flush=True)
