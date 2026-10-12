"""
Build all result tables from results/ (nothing is typed in by hand).

Outputs (results/tables/):
  summary.json        aggregated metrics per market x strategy
  table_<market>.tex  LaTeX table for the paper
  tables.md           the same, as Markdown

Statistics
----------
* Each strategy has one value per evaluation set k = 0..S-1 (4000 paths each; learned
  policy trained with seed k is evaluated on set k).  We report mean and standard error
  across the S sets.
* Delta CE vs Grossman-Zhou is computed per evaluation set on identical paths (common
  random numbers) and tested with a two-sided paired t-test across sets (df = S-1).
* Violation rates are pooled over all S x 4000 paths; the 95% upper bound uses the exact
  Clopper-Pearson interval (the "rule of three" when zero violations are observed).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))

ORDER = ["Cash", "Constant mix (60% risky)", "Merton (unconstrained)", "CPPI (m=3)", "TIPP (m=3)", "TIPP (m=5)",
         "Grossman-Zhou", "HJB optimum (continuous-time)", "HJB optimum (certified class)",
         "PPO", "PPO-Lagrangian", "E-CBLPO (PPO-trained)", "E-CBLPO (no GZ prior)", "E-CBLPO (MLP)", "E-CBLPO"]
MARKET_LABEL = {"constant_bs": "Black--Scholes, constant coefficients (1 asset)",
                "kim_omberg": "Kim--Omberg stochastic risk premium (1 asset)",
                "heston": "Multi-asset Heston stochastic volatility (5 assets)",
                "jump": "Merton jump-diffusion with market crashes (1 asset)"}
GAMMA, ALPHA = 2.0, 0.20


def fname(s):
    return s.replace(" ", "_").replace("/", "-")


def ce(W):
    return float(np.mean(W ** (1 - GAMMA)) ** (1 / (1 - GAMMA)))


def load(market):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, market, "raw", "*__eval*.npz")):
        base = os.path.basename(f)[:-4]
        strat, k = base.rsplit("__eval", 1)
        out[strat][int(k)] = np.load(f)
    return out


def clopper_pearson_upper(x, n, conf=0.95):
    if x == n:
        return 1.0
    return float(stats.beta.ppf(1 - (1 - conf) / 2, x + 1, n - x)) if x > 0 else 1 - ((1 - conf) / 2) ** (1 / n)


def summarise(market):
    raw = load(market)
    names = {fname(s): s for s in ORDER}
    gz = raw.get(fname("Grossman-Zhou"), {})
    rows = {}
    for key, per_k in raw.items():
        name = names.get(key, key)
        ks = sorted(per_k)
        CE = np.array([ce(per_k[k]["W_T"]) for k in ks])
        EW = np.array([per_k[k]["W_T"].mean() for k in ks])
        DD = np.array([per_k[k]["max_dd"].mean() for k in ks])
        GR = np.array([per_k[k]["mean_gross"].mean() for k in ks])
        n_paths = int(sum(per_k[k]["max_dd"].size for k in ks))
        n_viol = int(sum((per_k[k]["max_dd"] > ALPHA + 1e-9).sum() for k in ks))
        common = [k for k in ks if k in gz]
        d = np.array([ce(per_k[k]["W_T"]) - ce(gz[k]["W_T"]) for k in common])
        if len(d) >= 2 and np.std(d, ddof=1) > 0:
            t, p = stats.ttest_1samp(d, 0.0)
        else:
            t, p = float("nan"), float("nan")
        se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
        rows[name] = {
            "n_sets": len(ks), "n_paths": n_paths,
            "CE": float(CE.mean()), "CE_se": se(CE),
            "dCE_vs_GZ_bp": float(d.mean() * 1e4) if len(d) else float("nan"),
            "dCE_vs_GZ_se_bp": se(d) * 1e4 if len(d) > 1 else float("nan"),
            "t_vs_GZ": float(t), "p_vs_GZ": float(p),
            "mean_W": float(EW.mean()), "mean_maxdd": float(DD.mean()),
            "viol_rate": n_viol / n_paths, "viol_count": n_viol, "viol_ub95": clopper_pearson_upper(n_viol, n_paths),
            "gross": float(GR.mean()),
        }
    return rows


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def to_latex(market, rows):
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{" + MARKET_LABEL[market] + r". Mean over " + str(next(iter(rows.values()))["n_sets"]) +
             r" evaluation sets of 4{,}000 one-year paths ($\alpha=20\%$, 10\,bp costs, CRRA $\rho=2$). "
             r"CE: certainty-equivalent terminal wealth ($\pm$ s.e.\ across sets). $\Delta$CE: paired difference to "
             r"Grossman--Zhou on identical paths, in basis points, with paired $t$-test $p$-value. "
             r"Viol.: share of paths with $\max_t D_t>\alpha$ (95\% Clopper--Pearson upper bound in brackets). "
             r"Gross: mean $\|u_t\|_1$.}",
             r"\label{tab:" + market + "}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Strategy & CE & $\Delta$CE vs GZ (bp) & $p$ & Viol.\ (\%) & Mean max DD (\%) & Gross \\", r"\midrule"]
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]
        if name == "PPO":
            lines.append(r"\midrule")
        tex_name = name.replace("%", r"\%").replace("Grossman-Zhou", "Grossman--Zhou")
        nm = r"\textbf{" + tex_name + "}" if name == "E-CBLPO" else tex_name
        d = "--" if name == "Grossman-Zhou" else f"${r['dCE_vs_GZ_bp']:+.1f}\\pm{r['dCE_vs_GZ_se_bp']:.1f}$"
        p = "--" if name == "Grossman-Zhou" else "$" + fmt_p(r["p_vs_GZ"]) + "$"
        v = f"{100*r['viol_rate']:.2f} [{100*r['viol_ub95']:.2f}]"
        lines.append(f"{nm} & {r['CE']:.4f} $\\pm$ {r['CE_se']:.4f} & {d} & {p} & {v} & {100*r['mean_maxdd']:.2f} & {r['gross']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def to_md(market, rows):
    out = [f"### {MARKET_LABEL[market].replace('--', '–')}", "",
           "| Strategy | CE (± s.e.) | ΔCE vs GZ, bp (± s.e.) | p | Viol. % [95% UB] | Mean max DD % | Gross |",
           "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]
        d = "—" if name == "Grossman-Zhou" else f"{r['dCE_vs_GZ_bp']:+.1f} ± {r['dCE_vs_GZ_se_bp']:.1f}"
        p = "—" if name == "Grossman-Zhou" else fmt_p(r["p_vs_GZ"])
        nm = f"**{name}**" if name == "E-CBLPO" else name
        out.append(f"| {nm} | {r['CE']:.4f} ± {r['CE_se']:.4f} | {d} | {p} | {100*r['viol_rate']:.2f} [{100*r['viol_ub95']:.2f}] | {100*r['mean_maxdd']:.2f} | {r['gross']:.3f} |")
    return "\n".join(out)


def main():
    outdir = os.path.join(RES, "tables")
    os.makedirs(outdir, exist_ok=True)
    allrows, md = {}, []
    for market in MARKET_LABEL:
        if not os.path.isdir(os.path.join(RES, market, "raw")):
            continue
        rows = summarise(market)
        allrows[market] = rows
        open(os.path.join(outdir, f"table_{market}.tex"), "w").write(to_latex(market, rows))
        md.append(to_md(market, rows))
    json.dump(allrows, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    open(os.path.join(outdir, "tables.md"), "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
