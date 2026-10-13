"""Aggregate results/extensions/nscaling into results/extensions/nscaling_summary.json and a table."""
import glob, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "results", "extensions", "nscaling")
cases = ["N5", "N10", "N20", "N50", "N100", "N20_corr0", "N20_corr0.6", "N20_coverr"]
out = {}
for c in cases:
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, f"{c}__seed*.json")))]
    if not R:
        continue
    row = {"seeds": len(R)}
    for k in ("ECP", "TIPP (m=5)", "Grossman-Zhou"):
        ce = np.array([r[k]["CE"] for r in R])
        row[k] = {"CE": float(ce.mean()), "CE_se": float(ce.std(ddof=1) / np.sqrt(len(ce))),
                  "viol": float(np.mean([r[k]["viol"] for r in R])), "gross": float(np.mean([r[k]["gross"] for r in R]))}
    d = np.array([1e4 * (r["ECP"]["CE"] - r["TIPP (m=5)"]["CE"]) for r in R])
    row["ECP_minus_TIPP5_bp"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
    row["us_per_decision"] = float(1e6 * np.median([r["ECP_sec_per_decision"] for r in R]))
    row["peak_rss_MB"] = float(np.max([r["peak_rss_MB"] for r in R]))
    out[c] = row
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "nscaling_summary.json"), "w"), indent=1)
L = [r"\begin{tabular}{l ccc cc cc}", r"\toprule",
     r"Case & \method{} CE & viol. & \method{}$-$TIPP(5) (bp) & GZ CE & GZ viol. & $\mu$s/decision & peak RSS (MB)\\", r"\midrule"]
lab = {"N5": "$N=5$ (training)", "N10": "$N=10$", "N20": "$N=20$", "N50": "$N=50$", "N100": "$N=100$",
       "N20_corr0": "$N=20$, $\\rho=0$", "N20_corr0.6": "$N=20$, $\\rho=0.6$", "N20_coverr": "$N=20$, estimated corr."}
for c, r in out.items():
    L.append(f"{lab[c]} & {r['ECP']['CE']:.4f} & {100*r['ECP']['viol']:.2f}\\% & ${r['ECP_minus_TIPP5_bp'][0]:+.1f}\\pm{r['ECP_minus_TIPP5_bp'][1]:.1f}$"
             f" & {r['Grossman-Zhou']['CE']:.4f} & {100*r['Grossman-Zhou']['viol']:.2f}\\% & {r['us_per_decision']:.0f} & {r['peak_rss_MB']:.0f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tables", "table_nscaling.tex"), "w").write("\n".join(L) + "\n")
print(json.dumps(out, indent=1))
