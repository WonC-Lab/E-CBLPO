"""
Item 12: sensitivity of the results to the transaction-cost convention (Remark rem:drift). Every strategy is
re-executed (no retraining) with costs charged (a) on the change of target weights, as in the paper, and (b) on
the move from the drift-adjusted pre-trade weights. Settings: Black-Scholes and Kim-Omberg (5 evaluation sets of
4,000 paths, the paper's evaluation seeds; ECP checkpoints of run_benchmark.py) and the CRSP market study E1
(all 524 episodes; ECP checkpoints of run_empirical.py). Reported: CE, mean annual turnover, mean maximum drawdown,
violation rate. The cap implied by Remark rem:drift is (1-eta)/(ell+2*lambda*c) with lambda = 1.48.
Output: results/extensions/cost_sensitivity.json
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.sim.historical import HistoricalPaths, estimate
from src.agents.rules import default_rules
from src.agents.pathwise import PathwisePolicy
from src.agents.hjb import solve_hjb, HJBPolicy
import run_empirical as RE

OUT = os.path.join(ROOT, "results", "extensions", "cost_sensitivity.json")
ALPHA, TC, ETA, RHO, T = 0.2, 0.001, 0.05, 2.0, 252
RULES = ("TIPP (m=3)", "TIPP (m=5)", "Grossman-Zhou", "CPPI (m=3)")


def run(market, strat, B, seed, drift, T_=T):
    env = DrawdownPortfolioEnv(market, batch_size=B, horizon=T_, alpha=ALPHA, tc=TC, gamma=RHO, drift_cost=drift)
    o = env.reset(seed)
    for _ in range(T_):
        o, *_ = env.step(strat.act(o))
    s = env.summary()
    return s["W_T"], s["max_dd"], s["turnover"]


def sim_job(args):
    mkt, k = args
    torch.set_num_threads(1)
    spec = {"constant_bs": dict(kw={"num_assets": 1}, ell=0.2), "kim_omberg": dict(kw={}, ell=0.2)}[mkt]
    m = make_market(mkt, **spec["kw"])
    fdim = 3 + m.features(m.reset(1, np.random.default_rng(0))).shape[-1]
    pol = PathwisePolicy(m.N, fdim, alpha=ALPHA, ell=spec["ell"], tc=TC, eta=ETA, rho=RHO, r=m.r, name="ECP")
    pol.actor.load_state_dict(torch.load(os.path.join(ROOT, "results", "checkpoints", f"{mkt}__E-CBLPO__seed{k}.pt")))
    strats = [s for s in default_rules(RHO) if s.name in RULES] + [RE.ECPAct(pol, "ECP")]
    out = {}
    for s in strats:
        for drift in (False, True):
            W, dd, to = run(make_market(mkt, **spec["kw"]), s, 4000, 10_000 + k, drift)
            out[f"{s.name}|{'drift' if drift else 'target'}"] = {"W_T": W.tolist(), "max_dd": dd.tolist(), "turnover": to.tolist()}
    return ("sim", mkt, k, out)


def emp_job(y):
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    base = json.load(open(os.path.join(RE.RES, "E1", f"{y}.json")))
    R, rf, _, wy, _ = RE.load("E1")
    dates = R.index
    win = (dates >= pd.Timestamp(f"{y - wy}-01-01")) & (dates < pd.Timestamp(f"{y}-01-01"))
    ex, cov = estimate(R[win].values - rf[win].values[:, None])
    r_hat = base["r"]
    starts = np.array([dates.get_loc(pd.Timestamp(s)) for s in base["starts"]])
    mk = lambda: HistoricalPaths(R.values, rf.values, starts, ex, cov, r_hat)
    mu, sig = r_hat + float(ex[0]), float(np.sqrt(cov[0, 0]))
    strats = [s for s in default_rules(RHO) if s.name in RULES]
    strats.append(HJBPolicy(solve_hjb(mu=mu, sigma=sig, r=r_hat, alpha=ALPHA, gamma=RHO, K_cap=RE.K_CERT), T, "HJB optimum (certified class)"))
    for seed in range(3):
        pol = PathwisePolicy(1, 3, alpha=ALPHA, ell=RE.ELL, tc=TC, eta=ETA, rho=RHO, r=r_hat, name="ECP", detach_prev=True)
        pol.actor.load_state_dict(torch.load(os.path.join(RE.RES, "E1", "checkpoints", f"{y}_seed{seed}.pt")))
        strats.append(RE.ECPAct(pol, f"ECP{seed}"))
    out = {}
    for s in strats:
        for drift in (False, True):
            W, dd, to = run(mk(), s, len(starts), 0, drift)
            out[f"{s.name}|{'drift' if drift else 'target'}"] = {"W_T": W.tolist(), "max_dd": dd.tolist(), "turnover": to.tolist()}
    return ("emp", "E1", y, out)


def dispatch(j):
    return sim_job(j[1]) if j[0] == "sim" else emp_job(j[1])


if __name__ == "__main__":
    jobs = [("sim", (m, k)) for m in ("constant_bs", "kim_omberg") for k in range(5)] + [("emp", y) for y in RE.load("E1")[2]]
    res = {"sim": {}, "emp": {}}
    with Pool(int(os.environ.get("WORKERS", 2))) as p:
        for kind, a, b, out in p.imap_unordered(dispatch, jobs):
            res[kind].setdefault(a, {})[str(b)] = out
            print(kind, a, b, flush=True)
    json.dump(res, open(OUT, "w"))
