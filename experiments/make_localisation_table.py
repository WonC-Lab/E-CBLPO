"""Table tab:gauss from results/extensions/localisation.json (32 steps/day; 8 steps/day in brackets for
the relative error, to show the time-step effect near maturity)."""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = json.load(open(os.path.join(ROOT, "results", "extensions", "localisation.json")))
rows = [(0.06, 0.3, 0.9), (0.06, 0.2, 0.5), (0.1, 0.2, 0.25), (0.06, 0.2, 0.1), (0.06, 0.2, 0.05), (0.06, 0.2, 0.02)]
cols = [1e-3, 1e-5, 1e-7, 1e-9]
L = [r"\begin{tabular}{l cccc cccc}", r"\toprule",
     r" & \multicolumn{4}{c}{$m$: HJB / explicit, at $x-a=$} & \multicolumn{4}{c}{relative error (\%), 32 [8] steps/day}\\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r"$(\mu,\sigma,\tau)$ & $10^{-3}$ & $10^{-5}$ & $10^{-7}$ & $10^{-9}$ & $10^{-3}$ & $10^{-5}$ & $10^{-7}$ & $10^{-9}$\\", r"\midrule"]
for mu, sg, tau in rows:
    a = d[f"mu={mu},sigma={sg},tau={tau},spd=32"]; b = d[f"mu={mu},sigma={sg},tau={tau},spd=8"]
    idx = [a["y"].index(c) for c in cols]
    left = [f"{a['m_hjb'][i]:.2f} / {a['m_explicit'][i]:.2f}" for i in idx]
    right = [f"{100*(a['m_hjb'][i]/a['m_explicit'][i]-1):.1f} [{100*(b['m_hjb'][i]/b['m_explicit'][i]-1):.1f}]" for i in idx]
    L.append(f"$({mu*100:.0f}\\%,{sg*100:.0f}\\%,{tau:g})$ & " + " & ".join(left + right) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tables", "table_localisation.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
