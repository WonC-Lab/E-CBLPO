"""Item 13: E1 with the i.i.d. bootstrap (paper, EMP_BLOCK=1) against a stationary block bootstrap with
mean block length 20 days (EMP_BLOCK=20 RESULTS_DIR=results/block20 run_empirical.py E1). Same episodes,
same evaluation; only the training data of ECP differ. Year-bootstrap intervals as in
make_empirical_tables.py. Output: results/empirical/bootstrap_comparison.json"""
import glob, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G = 2.0
ce = lambda W: float(np.mean(np.asarray(W) ** (1 - G)) ** (1 / (1 - G)))


def load(d):
    yl, ecp, cert, viol, turn, gross, pre90 = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "[0-9]*.json"))):
        r = json.load(open(f)); n = len(r["starts"]); yl += [r["year"]] * n
        S = r["strategies"]
        ecp.append(np.array([S[f"ECP (seed {k})"]["W_T"] for k in range(3)]))
        viol.append(np.array([np.array(S[f"ECP (seed {k})"]["max_dd"]) > 0.2 + 1e-9 for k in range(3)]))
        turn.append(np.array([S[f"ECP (seed {k})"]["turnover"] for k in range(3)]))
        gross.append(np.array([S[f"ECP (seed {k})"]["gross"] for k in range(3)]))
        cert.append(np.array(S["HJB optimum (certified class)"]["W_T"]))
    c = lambda L: np.concatenate(L, axis=-1)
    return np.array(yl), c(ecp), c(cert), c(viol), c(turn), c(gross)


I = load(os.path.join(ROOT, "results", "empirical", "E1"))
B = load(os.path.join(ROOT, "results", "block20", "empirical", "E1"))
assert (I[0] == B[0]).all() and np.allclose(I[2], B[2])          # same episodes, same HJB benchmark
yl = I[0]; uy = np.unique(yl); groups = [np.where(yl == y)[0] for y in uy]
rng = np.random.default_rng(0)
boot = [np.concatenate([groups[i] for i in rng.integers(0, len(uy), len(uy))]) for _ in range(2000)]
ECE = lambda W, idx=slice(None): float(np.mean([ce(W[k, idx]) for k in range(3)]))
d = lambda idx=slice(None): 1e4 * (ECE(B[1], idx) - ECE(I[1], idx))
pre = yl < 1990
out = {"iid": {"CE": ECE(I[1]), "viol": float(I[3].mean()), "turnover": float(I[4].mean()), "gross": float(I[5].mean()),
               "turnover_pre1990": float(I[4][:, pre].mean()), "turnover_post1990": float(I[4][:, ~pre].mean())},
       "block20": {"CE": ECE(B[1]), "viol": float(B[3].mean()), "turnover": float(B[4].mean()), "gross": float(B[5].mean()),
                   "turnover_pre1990": float(B[4][:, pre].mean()), "turnover_post1990": float(B[4][:, ~pre].mean())},
       "cert_HJB_CE": ce(I[2]),
       "block_minus_iid_bp": [d(), float(np.percentile([d(i) for i in boot], 2.5)), float(np.percentile([d(i) for i in boot], 97.5))],
       "block_minus_iid_pre1990_bp": d(np.where(pre)[0]), "block_minus_iid_post1990_bp": d(np.where(~pre)[0])}
import pandas as pd
ff = pd.read_csv(os.path.join(ROOT, "data", "raw", "wrds", "ff_daily.csv"), parse_dates=["date"]).set_index("date")["mktrf"]
out["lag1_autocorr"] = {f"{a}-{b}": float(ff[a:b].autocorr(1)) for a, b in [("1962", "1981"), ("1970", "1989"), ("1990", "2009")]}
json.dump(out, open(os.path.join(ROOT, "results", "empirical", "bootstrap_comparison.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
