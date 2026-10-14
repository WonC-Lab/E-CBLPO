"""
Item 14: what happens to the certified policies on days on which the one-period stress bound
ell = 20% is exceeded (|daily excess return| > ell for some asset), in E2 (sector ETFs) and E3
(zero-shot top-50 stocks). The trained ECP checkpoints are re-executed (no retraining) and, on
each exceedance day, we record the weight on every exceeding asset, the portfolio return, the
cushion h = W - aH before the day, and the fraction of the cushion lost that day.
A violation needs the loss to exceed the whole cushion; Proposition prop:discrete excludes it
whenever no asset breaches ell, with the worst case in which every position loses ell at once.
Output: results/empirical/stress_exceedance.json
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
import run_empirical
from run_empirical import RAW, RES, ALPHA, TC, ETA, GAMMA, ELL, ECPAct, load
from src.sim.historical import HistoricalPaths, estimate
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.pathwise import PathwisePolicy
from src.agents.rules import default_rules
from run_empirical_zeroshot import ledoit_wolf, HIST
a = 1 - ALPHA


def trace(market, strat, T, Xex):
    """Xex: (B, T, N) daily excess returns of each episode (to find exceedance days)."""
    env = DrawdownPortfolioEnv(market, batch_size=len(market.starts), horizon=T, alpha=ALPHA, tc=TC, gamma=GAMMA)
    o = env.reset(0); ev = []; min_ratio = np.inf
    for t in range(T):
        W0, H0 = env.W.copy(), env.H.copy()
        u = strat.act(o)
        o, *_ = env.step(u)
        u = env.u_prev
        h0 = W0 - a * H0; h1 = env.W - a * H0
        min_ratio = min(min_ratio, float((h1 / h0).min()))
        for b in np.where((np.abs(Xex[:, t]) > ELL).any(1))[0]:
            idx = np.where(np.abs(Xex[b, t]) > ELL)[0]
            ev.append({"episode": int(b), "day": t, "assets": idx.tolist(), "falling": bool(Xex[b, t, idx].min() < -ELL),  # flag only: no CRSP returns are stored
                       "weights": u[b, idx].tolist(), "gross": float(np.abs(u[b]).sum()),
                       "cushion_before": float(h0[b] / H0[b]), "cushion_lost_frac": float(1 - h1[b] / h0[b]),
                       "port_ret": float(env.W[b] / W0[b] - 1)})
    sm = env.summary()
    return ev, min_ratio, sm["violated"].tolist(), sm["W_T"].tolist()


def e3():
    out = {}
    ff = pd.read_csv(os.path.join(RAW, "ff_daily.csv"), parse_dates=["date"]).set_index("date")["rf"]
    cur = pd.read_csv(os.path.join(RAW, "top50_daily.csv"), parse_dates=["date"])
    top = pd.read_csv(os.path.join(RAW, "top50_by_year.csv"))
    hist = pd.read_csv(HIST, parse_dates=["date"])
    for y in range(2004, 2026):
        f = os.path.join(RES, "E3", f"{y}.json")
        if not os.path.exists(f):
            continue
        base = json.load(open(f))
        if not base["stress_exceed"][0]:
            continue
        perm = top[top.year == y].permno.tolist()
        R = cur[cur.date.dt.year == y].pivot(index="date", columns="permno", values="ret").reindex(columns=perm)
        R = R.join(ff, how="inner"); rf = R.pop("rf")
        # identical estimates to run_empirical_zeroshot.py (checked below against the stored W_T)
        H = hist[hist.year == y].pivot(index="date", columns="permno", values="ret").reindex(columns=perm)
        H = H.join(ff, how="inner"); rfh = H.pop("rf")
        Xh = np.nan_to_num(np.where(np.isnan(H.values - rfh.values[:, None]),
                                    np.nanmean(H.values - rfh.values[:, None], axis=0, keepdims=True),
                                    H.values - rfh.values[:, None]))
        m = Xh.mean(0) * 252
        ex = 0.5 * m + 0.5 * m.mean()
        cov = ledoit_wolf(Xh) * 252
        assert np.allclose(ex, base["ex"])
        T = len(R)
        Xe = np.nan_to_num(R.values) - rf.values[:, None]
        rec = {}
        for ck in sorted(glob.glob(os.path.join(RES, "E2", "checkpoints", f"{y}_seed*.pt"))):
            pol = PathwisePolicy(len(perm), 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=GAMMA, r=base["r"], name="ECP")
            pol.actor.load_state_dict(torch.load(ck))
            mk = HistoricalPaths(np.nan_to_num(R.values), rf.values, [0], ex, cov, base["r"])
            ev, mr, vi, WT = trace(mk, ECPAct(pol, "ECP"), T, Xe[None])
            assert abs(WT[0] - base["strategies"][f"ECP (seed {ck[-4]})"]["W_T"][0]) < 1e-6
            rec[f"seed {ck[-4]}"] = {"events": ev, "min_cushion_ratio": mr, "violated": vi}
        out[str(y)] = rec
        print("E3", y, {k: (len(v["events"]), round(v["min_cushion_ratio"], 3)) for k, v in rec.items()}, flush=True)
    return out


def e2():
    out = {}
    R, rf, years, wy, wmin = load("E2")
    dates = R.index; Rv, rfv = R.values, rf.values
    for y in years:
        f = os.path.join(RES, "E2", f"{y}.json")
        if not os.path.exists(f):
            continue
        base = json.load(open(f))
        if not any(base["stress_exceed"]):
            continue
        win = (dates >= pd.Timestamp(f"{y - wy}-01-01")) & (dates < pd.Timestamp(f"{y}-01-01"))
        ex, cov = estimate(R[win].values - rf[win].values[:, None])
        starts = np.array([dates.get_loc(pd.Timestamp(s)) for s in base["starts"]])
        Xe = np.stack([Rv[s:s + 252] - rfv[s:s + 252, None] for s in starts])
        rec = {}
        strats = [(f"seed {k}", None) for k in (0, 1, 2)] + [(s.name, s) for s in default_rules(GAMMA)
                                                              if s.name in ("TIPP (m=3)", "Grossman-Zhou", "CPPI (m=3)")]
        for name, s in strats:
            if s is None:
                pol = PathwisePolicy(R.shape[1], 3, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=GAMMA, r=base["r"],
                                     name="ECP", detach_prev=True)
                pol.actor.load_state_dict(torch.load(os.path.join(RES, "E2", "checkpoints", f"{y}_{name.replace(' ', '')}.pt")))
                s = ECPAct(pol, "ECP")
            ev, mr, vi, WT = trace(HistoricalPaths(Rv, rfv, starts, ex, cov, base["r"]), s, 252, Xe)
            key = f"ECP (seed {name[-1]})" if name.startswith("seed") else name
            assert np.allclose(WT, base["strategies"][key]["W_T"], atol=1e-6), (y, name)
            rec[name] = {"events": ev, "min_cushion_ratio": mr, "violated": vi}
        out[str(y)] = rec
        print("E2", y, {k: (len(v["events"]), round(v["min_cushion_ratio"], 3)) for k, v in rec.items()}, flush=True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    res = {"E2": e2(), "E3": e3()}
    json.dump(res, open(os.path.join(RES, "stress_exceedance.json"), "w"), indent=1)
