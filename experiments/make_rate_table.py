"""
Items 8-9: rate of the price of certification with error bars.
gap_R = 2*gap_64 - gap_32 (first-order Richardson in the time step; spd = steps per day).
Error bar = |gap_64 - gap_32| (time) + |M4000 - M2000| at spd32 (space)
            + |Howard-implicit - lagged| at spd32 (control lag), the last two where available.
A point is 'resolved' if gap_R > 2 * error bar. Local slopes use resolved neighbours only.
Inputs: results/extensions/price_rate_spd{32,64}_r=*.json, solver_checks_B.json.
Output: results/extensions/rate_table.json and results/tables/table_rate.tex.
"""
import json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E = os.path.join(ROOT, "results", "extensions")
B = json.load(open(os.path.join(E, "solver_checks_B.json")))
out = {}
for case, p in [("r=0.02", 2), ("r=0.005", 2), ("r=0", 3)]:
    g32 = json.load(open(os.path.join(E, f"price_rate_spd32_{case}.json")))[case]
    g64 = json.load(open(os.path.join(E, f"price_rate_spd64_{case}.json")))[case]
    K = np.array(g32["K"]); a32 = np.array(g32["gap_bp"]); a64 = np.array(g64["gap_bp"])
    R = 2 * a64 - a32
    err = np.abs(a64 - a32)
    for k, KK in enumerate(K):
        for var in ("M4000_spd32", "spd32_howard1"):
            rec = B.get(f"{case}|{var}")
            if rec and KK in rec["K"]:
                err[k] += abs(rec["gap_bp"][rec["K"].index(KK)] - a32[k])
    ok = R > 2 * err
    slope = [None] + [float(np.log(R[i] / R[i - 1]) / np.log(K[i] / K[i - 1])) if ok[i] and ok[i - 1] else None
                      for i in range(1, len(K))]
    out[case] = dict(K=K.tolist(), gap_R=R.tolist(), err=err.tolist(), resolved=ok.tolist(), slope=slope,
                     scaled=(R * K ** p).tolist(), scaled_err=(err * K ** p).tolist(), power=p)
json.dump(out, open(os.path.join(E, "rate_table.json"), "w"), indent=1)

def f(x, e):
    """value +- one-significant-digit error, both at the error's decimal place"""
    d = max(0, -int(np.floor(np.log10(e)))) if e > 0 else 4
    return f"{x:.{d}f}$\\pm${e:.{d}f}"
a, b, c5 = out["r=0.02"], out["r=0"], out["r=0.005"]
for d in (a, c5):   # interest-rate component K^2 [gap(r) - gap(0)]
    d["rate_part"] = [K * K * (g - g0) for K, g, g0 in zip(d["K"], d["gap_R"], b["gap_R"])]
    d["rate_part_err"] = [K * K * (e + e0) for K, e, e0 in zip(d["K"], d["err"], b["err"])]
json.dump(out, open(os.path.join(E, "rate_table.json"), "w"), indent=1)
lines = [r"\begin{tabular}{r ccc ccc cc}", r"\toprule",
         r" & \multicolumn{3}{c}{$r=2\%$} & \multicolumn{3}{c}{$r=0$} & \multicolumn{2}{c}{$K^2[\mathrm{gap}(r)-\mathrm{gap}(0)]$}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
         r"$K$ & gap (bp) & $K^2\cdot$gap & slope & gap (bp) & $K^3\cdot$gap$/10^3$ & slope & $r=2\%$ & $r=0.5\%$\\", r"\midrule"]
for i, KK in enumerate(a["K"]):
    cells = []
    for d, sc in ((a, 1.0), (b, 1e-3)):
        star = "" if d["resolved"][i] else "$^\\dagger$"
        s_ = d["slope"][i]
        cells += [f(d["gap_R"][i], d["err"][i]) + star, f(d["scaled"][i] * sc, d["scaled_err"][i] * sc),
                  "--" if s_ is None else f"{s_:.2f}"]
    cells += [f(d["rate_part"][i], d["rate_part_err"][i]) for d in (a, c5)]
    lines.append(f"{KK} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tables", "table_rate.tex"), "w").write("\n".join(lines) + "\n")
for c, d in out.items():
    print(c)
    for i in range(len(d["K"])):
        print(f"  K={d['K'][i]:4d} gap={d['gap_R'][i]:.6f} err={d['err'][i]:.6f} ok={d['resolved'][i]} "
              f"K^p*gap={d['scaled'][i]:.1f}+-{d['scaled_err'][i]:.1f} slope={d['slope'][i]}")
