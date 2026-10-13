"""
TIPP with the multiplier chosen ex ante (adds a strategy to the E1 and E2 results).

For every evaluation year y, TIPP is backtested on the same estimation window as the other
strategies (one-year episodes started on the first trading day of every month inside the
window, entirely within it), for m in M_GRID.  The chosen multiplier is the one with the
highest CE among those with no violation in the window (if none is violation-free, the one
with the fewest violations).  It is then applied to the year-y episodes.  Writes the strategy
"TIPP (m chosen ex ante)" and the chosen m ("tipp_m") into results/empirical/<exp>/<y>.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from src.sim.historical import HistoricalPaths, estimate
from src.agents.rules import TIPP
from run_empirical import load, run_strategy, RES, ALPHA, T

M_GRID = np.round(np.arange(1.0, 10.01, 0.5), 2)
NAME = "TIPP (m chosen ex ante)"


def ce(W):
    return float(np.mean(np.asarray(W) ** -1.0) ** -1.0)


def main(exp):
    R, rf, years, wy, wmin = load(exp)
    dates, Rv, rfv = R.index, R.values, rf.values
    for y in years:
        f = os.path.join(RES, exp, f"{y}.json")
        if not os.path.exists(f):
            continue
        res = json.load(open(f))
        win = np.where((dates >= pd.Timestamp(f"{y - wy}-01-01")) & (dates < pd.Timestamp(f"{y}-01-01")))[0]
        X = Rv[win] - rfv[win, None]
        ex, cov = estimate(X)
        first = [win[(dates[win].year == yy) & (dates[win].month == mm)][0]
                 for yy in sorted(set(dates[win].year)) for mm in range(1, 13)
                 if ((dates[win].year == yy) & (dates[win].month == mm)).any()]
        starts_in = np.array([s for s in first if s + T <= win[-1] + 1])
        scores = []
        for m in M_GRID:
            o = run_strategy(HistoricalPaths(Rv, rfv, starts_in, ex, cov, res["r"]), TIPP(float(m)))
            viol = int((np.asarray(o["max_dd"]) > ALPHA + 1e-9).sum())
            scores.append((viol, -ce(o["W_T"]), float(m)))
        viol, nce, m_star = min(scores)
        starts = np.array([dates.get_loc(pd.Timestamp(s)) for s in res["starts"]])
        res["strategies"][NAME] = run_strategy(HistoricalPaths(Rv, rfv, starts, np.array(res["ex"]),
                                                               cov, res["r"]), TIPP(m_star))
        res["tipp_m"] = m_star
        res["tipp_window"] = {"episodes": int(len(starts_in)), "violations": viol, "ce": -nce}
        json.dump(res, open(f, "w"))
        print(exp, y, "m* =", m_star, "window viol", viol, "CE_window %.4f" % -nce, flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:] or ["E1", "E2"]:
        main(e)
