"""Aggregate results/extensions/cost_sensitivity.json (raw, not tracked) into cost_sensitivity_summary.json."""
import json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = json.load(open(os.path.join(ROOT, "results", "extensions", "cost_sensitivity.json")))
ce = lambda W: float(np.mean(np.asarray(W) ** -1.0) ** -1.0)
out = {}
for mkt, seeds in D["sim"].items():
    names = sorted({k.split("|")[0] for k in seeds["0"]})
    for n in names:
        for conv in ("target", "drift"):
            R = [seeds[k][f"{n}|{conv}"] for k in sorted(seeds)]
            out.setdefault(mkt, {}).setdefault(n, {})[conv] = {
                "CE": float(np.mean([ce(r["W_T"]) for r in R])),
                "turnover": float(np.mean([np.mean(r["turnover"]) for r in R])),
                "mean_maxdd": float(np.mean([np.mean(r["max_dd"]) for r in R])),
                "viol": float(np.mean([np.mean(np.asarray(r["max_dd"]) > 0.2 + 1e-9) for r in R]))}
years = D["emp"]["E1"]
names = sorted({k.split("|")[0] for k in years[next(iter(years))]})
for n in names:
    for conv in ("target", "drift"):
        W = np.concatenate([years[y][f"{n}|{conv}"]["W_T"] for y in sorted(years)])
        to = np.concatenate([years[y][f"{n}|{conv}"]["turnover"] for y in sorted(years)])
        dd = np.concatenate([years[y][f"{n}|{conv}"]["max_dd"] for y in sorted(years)])
        out.setdefault("E1", {}).setdefault(n, {})[conv] = {"CE": ce(W), "turnover": float(to.mean()),
                                                             "mean_maxdd": float(dd.mean()), "viol": float(np.mean(dd > 0.2 + 1e-9))}
if "ECP0" in out["E1"]:
    out["E1"]["ECP"] = {conv: {k: float(np.mean([out["E1"][f"ECP{s}"][conv][k] for s in range(3)])) for k in ("CE", "turnover", "mean_maxdd", "viol")}
                        for conv in ("target", "drift")}
    for s in range(3):
        out["E1"].pop(f"ECP{s}")
lam = (1 + 0.02 / 252 + 0.2) / (1 - 0.95 * 0.2)
out["caps"] = {"target": 0.95 / (0.2 + 2 * 0.001), "drift (lambda c)": 0.95 / (0.2 + 2 * lam * 0.001), "lambda": lam}
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "cost_sensitivity_summary.json"), "w"), indent=1)
for mkt in ("constant_bs", "kim_omberg", "E1"):
    print(mkt)
    for n, r in out[mkt].items():
        t, d = r["target"], r["drift"]
        print(f"  {n:32s} CE {t['CE']:.4f}->{d['CE']:.4f} ({1e4*(d['CE']-t['CE']):+.1f}bp) turnover {t['turnover']:.2f}->{d['turnover']:.2f}"
              f" maxDD {t['mean_maxdd']:.4f}->{d['mean_maxdd']:.4f} viol {t['viol']:.4f}->{d['viol']:.4f}")
print(out["caps"])
