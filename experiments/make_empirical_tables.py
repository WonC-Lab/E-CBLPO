"""
Aggregate results/empirical/<exp>/<year>.json into tables and a figure.

Episodes overlap (12 monthly starts per year, one year each), so uncertainty is measured by
a bootstrap over evaluation years (all episodes of a year resampled together).  ECP is the
average over its three training seeds (per-seed CE, then mean), and every difference is
computed on identical episodes.
Output: results/empirical/summary.json, results/tables/table_empirical_<exp>.tex,
        results/figures/fig8_empirical.pdf
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
EMP = os.path.join(RES, "empirical")
ALPHA, GAMMA, NBOOT = 0.20, 2.0, 2000
ORDER = ["Cash", "Constant mix (60% risky)", "Merton (unconstrained)", "CPPI (m=3)", "TIPP (m=3)", "TIPP (m=5)",
         "TIPP (m chosen ex ante)", "Grossman-Zhou", "HJB optimum (robust class, K=1)", "HJB optimum (continuous-time)",
         "HJB optimum (certified class)", "ECP"]
LABEL = {"Grossman-Zhou": "Grossman--Zhou", "TIPP (m chosen ex ante)": "TIPP ($m$ chosen ex ante)", "Constant mix (60% risky)": "Constant mix (60\\% risky)",
         "HJB optimum (robust class, K=1)": "HJB plug-in (robust class, $K=1$)",
         "HJB optimum (continuous-time)": "HJB plug-in (continuous-time)",
         "HJB optimum (certified class)": "HJB plug-in (certified class)", "ECP": "\\textbf{ECP}"}
CRISES = {"1987 crash": ("1986-11-01", "1987-10-19"), "2000--02 bear market": ("2000-01-01", "2002-12-31"),
          "2008 crisis": ("2007-10-01", "2008-12-31"), "2020 Covid crash": ("2019-04-01", "2020-03-23")}


def ce(W):
    W = np.asarray(W, float)
    return float(np.mean(W ** (1 - GAMMA)) ** (1 / (1 - GAMMA)))


def load(exp):
    years = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(EMP, exp, "*.json")))]
    data = {}   # strategy -> dict of arrays over all episodes, plus year labels
    yl, starts, stress = [], [], []
    for d in years:
        n = len(d["starts"])
        yl += [d["year"]] * n; starts += d["starts"]; stress += d["stress_exceed"]
        seeds = [k for k in d["strategies"] if k.startswith("ECP (seed")]
        for name, v in d["strategies"].items():
            if name.startswith("ECP (seed"):
                continue
            for key in v:
                data.setdefault(name, {}).setdefault(key, []).extend(v[key])
        for key in d["strategies"][seeds[0]]:
            stack = np.array([d["strategies"][s][key] for s in seeds])        # (seeds, n)
            data.setdefault("ECP", {}).setdefault(key + "_seeds", []).append(stack)
    if "ECP" in data:
        for key in list(data["ECP"]):
            data["ECP"][key] = np.concatenate(data["ECP"][key], axis=1)
        for key in ["W_T", "max_dd", "gross", "turnover"]:
            data["ECP"][key] = data["ECP"][key + "_seeds"].mean(0) if key != "W_T" else data["ECP"]["W_T_seeds"]
    return data, np.array(yl), np.array(starts), np.array(stress)


def ce_of(data, name, idx=None):
    W = np.asarray(data[name]["W_T"])
    if W.ndim == 2:                                   # ECP: mean over seeds of per-seed CE
        return float(np.mean([ce(w if idx is None else w[idx]) for w in W]))
    return ce(W if idx is None else W[idx])


def summarize(exp):
    data, yl, starts, stress = load(exp)
    names = [n for n in ORDER if n in data]
    years = np.unique(yl)
    rng = np.random.default_rng(0)
    boot = [np.concatenate([np.where(yl == y)[0] for y in rng.choice(years, len(years))]) for _ in range(NBOOT)]
    out = {"years": [int(years.min()), int(years.max())], "episodes": int(len(yl)),
           "stress_exceed_episodes": int(stress.sum()), "rows": {}}
    refs = [r for r in ["Grossman-Zhou", "TIPP (m=3)", "TIPP (m chosen ex ante)", "HJB optimum (certified class)"] if r in data]
    for n in names:
        v = data[n]
        W = np.asarray(v["W_T"]); Wm = W.mean(0) if W.ndim == 2 else W
        mdd = np.asarray(v["max_dd"])
        row = {"CE": ce_of(data, n), "mean_ret": float(Wm.mean() - 1), "viol": float((mdd > ALPHA + 1e-9).mean()),
               "viol_n": int((mdd > ALPHA + 1e-9).sum()), "mean_maxdd": float(mdd.mean()), "worst_maxdd": float(mdd.max()),
               "gross": float(np.mean(v["gross"])), "turnover": float(np.mean(v["turnover"]))}
        if n == "ECP":
            for ref in refs:
                d = (ce_of(data, n) - ce_of(data, ref)) * 1e4
                bs = np.array([(ce_of(data, n, b) - ce_of(data, ref, b)) * 1e4 for b in boot])
                row[f"d_vs_{ref}"] = [d, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["rows"][n] = row
    crises = {}
    for c, (a, b) in CRISES.items():
        sel = np.where((starts >= a) & (starts <= b))[0]
        if len(sel):
            crises[c] = {"episodes": int(len(sel)), "viol": {n: int((np.asarray(data[n]["max_dd"])[sel] > ALPHA + 1e-9).sum())
                                                            for n in names}}
    out["crises"] = crises
    # per-year CE differences (for the figure)
    out["per_year"] = {}
    for n in [r for r in ["TIPP (m=3)", "Grossman-Zhou", "HJB optimum (certified class)"] if r in data]:
        out["per_year"][n] = [[int(y), (ce_of(data, "ECP", np.where(yl == y)[0]) - ce_of(data, n, np.where(yl == y)[0])) * 1e4]
                              for y in years]
    out["per_year_viol"] = {n: [[int(y), int((np.asarray(data[n]["max_dd"])[yl == y] > ALPHA + 1e-9).sum())] for y in years]
                            for n in names}
    return out


def tex(exp, s, caption):
    L = [r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{" + caption + "}", r"\label{tab:emp_" + exp + "}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Strategy & CE & Mean return (\%) & Viol.\ (\%) & Mean / worst max DD (\%) & Gross & Turnover \\", r"\midrule"]
    for n, r in s["rows"].items():
        L.append(f"{LABEL.get(n, n)} & {r['CE']:.4f} & {100*r['mean_ret']:.2f} & {100*r['viol']:.1f} & "
                 f"{100*r['mean_maxdd']:.1f} / {100*r['worst_maxdd']:.1f} & {r['gross']:.2f} & {r['turnover']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def figure(s1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for n, col in [("TIPP (m=3)", "#e377c2"), ("HJB optimum (certified class)", "#444444")]:
        if n in s1["per_year"]:
            y, d = np.array(s1["per_year"][n]).T
            off = -0.2 if n.startswith("TIPP") else 0.2
            a1.bar(y + off, d, width=0.4, color=col, label=f"ECP − {n.replace('HJB optimum', 'HJB plug-in')}")
    a1.axhline(0, color="k", lw=0.8); a1.set_ylabel("CE difference (bp)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a1.set_title("US market, out of sample: ECP against TIPP and the certified HJB plug-in, by evaluation year", fontsize=10)
    for n, col, mk in [("HJB optimum (continuous-time)", "k", "*"), ("Merton (unconstrained)", "#d62728", "v"),
                       ("CPPI (m=3)", "#8c564b", "<"), ("ECP", "#2ca02c", "^")]:
        y, v = np.array(s1["per_year_viol"][n]).T
        a2.plot(y, v, mk + "-", color=col, label=n.replace("HJB optimum", "HJB plug-in"), ms=5)
    a2.set_ylabel("episodes violating\nthe 20% limit (of 12)"); a2.set_xlabel("evaluation year"); a2.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RES, "figures", "fig8_empirical.pdf")); fig.savefig(os.path.join(RES, "figures", "fig8_empirical.png"), dpi=200)
    plt.close(fig)


def main():
    summary = {}
    caps = {"E1": "US equity market (CRSP value-weighted, 1 asset), out of sample, evaluation years {y0}--{y1}: "
                  "{n} one-year episodes starting each month, $\\alpha=20\\%$, 10\\,bp costs, $\\gamma=2$. Plug-in "
                  "coefficients from the previous 20 years; ECP trained on a block bootstrap of the same window "
                  "(mean of 3 seeds). Viol.: share of episodes with maximum drawdown above $\\alpha$.",
            "E2": "Nine Select Sector SPDR ETFs, out of sample, evaluation years {y0}--{y1}: {n} one-year episodes. "
                  "Plug-in coefficients from the previous 10 years (at least 5). Columns as in Table~\\ref{{tab:emp_E1}}.",
            "E3": "Zero-shot transfer: ECP trained on the nine sector ETFs (E2, same year) executed without retraining "
                  "on the 50 largest US common stocks, {y0}--{y1}: one episode per calendar year ({n} episodes). Plug-in "
                  "coefficients from the constituents' previous 3 years (Ledoit--Wolf covariance, means shrunk halfway "
                  "to the cross-sectional average). Columns as in Table~\\ref{{tab:emp_E1}}."}
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    for exp in ["E1", "E2", "E3"]:
        if not glob.glob(os.path.join(EMP, exp, "*.json")):
            continue
        s = summary[exp] = summarize(exp)
        cap = caps[exp].format(y0=s["years"][0], y1=s["years"][1], n=s["episodes"])
        open(os.path.join(RES, "tables", f"table_empirical_{exp}.tex"), "w").write(tex(exp, s, cap))
        print(f"== {exp}: years {s['years']}, {s['episodes']} episodes, stress bound exceeded in {s['stress_exceed_episodes']}")
        for n, r in s["rows"].items():
            extra = "  ".join(f"{k[5:]}: {v[0]:+.1f} [{v[1]:+.1f},{v[2]:+.1f}]" for k, v in r.items() if k.startswith("d_vs"))
            print(f"  {n:34s} CE={r['CE']:.4f} ret={100*r['mean_ret']:5.2f}% viol={100*r['viol']:5.1f}% "
                  f"mdd={100*r['mean_maxdd']:4.1f}/{100*r['worst_maxdd']:4.1f} gross={r['gross']:.2f} turn={r['turnover']:.1f} {extra}")
        for c, v in s["crises"].items():
            print(f"  {c}: {v['episodes']} episodes; violations", {k: x for k, x in v["viol"].items() if x})
    if "E1" in summary:
        cols = ["Merton (unconstrained)", "Constant mix (60% risky)", "CPPI (m=3)", "HJB optimum (continuous-time)",
                "HJB optimum (certified class)", "TIPP (m=5)", "TIPP (m chosen ex ante)", "ECP"]
        head = ["Merton", "60\\% mix", "CPPI (3)", "HJB (cont.)", "HJB (cert.)", "TIPP (5)", "TIPP (ex ante)", "ECP"]
        L = [r"\begin{table}[t]", r"\centering\footnotesize",
             r"\caption{US market: one-year episodes that violate the 20\% drawdown limit, by start date: 1987 crash, "
             r"November 1986--19 October 1987; 2000--02, January 2000--December 2002; 2008, October 2007--December "
             r"2008; 2020, April 2019--23 March 2020. The stress bound $\ell=20\%$ was never exceeded by a daily excess "
             r"return in 1982--2025; the largest move is $-17.4\%$ on 19 October 1987 (the S\&P~500 fell $20.5\%$).}",
             r"\label{tab:emp_crises}", r"\begin{tabular}{lc" + "c" * len(cols) + "}", r"\toprule",
             "Period & Episodes & " + " & ".join(head) + r" \\", r"\midrule"]
        for c, v in summary["E1"]["crises"].items():
            L.append(f"{c} & {v['episodes']} & " + " & ".join(str(v['viol'].get(n, 0)) for n in cols) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(RES, "tables", "table_empirical_crises.tex"), "w").write("\n".join(L))
    json.dump(summary, open(os.path.join(EMP, "summary.json"), "w"), indent=1)
    if "E1" in summary:
        figure(summary["E1"])


if __name__ == "__main__":
    main()
