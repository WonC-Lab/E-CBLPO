"""Aggregate the ablations (run_ablation.py, plus 'no GZ prior' and 'MLP' from run_benchmark.py) into
results/extensions/ablation_summary.json and results/tables/table_ablation.tex.
Sample efficiency: iterations until the smoothed training-batch CE first reaches 90% of its total gain.
Stability: across-seed standard deviation of the evaluation CE, and the largest pre-clipping gradient norm."""
import glob, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
A = os.path.join(ROOT, "results", "extensions", "ablation")
RHO = 2.0
ce = lambda W: float(np.mean(np.asarray(W) ** (1 - RHO)) ** (1 / (1 - RHO)))


def eff(h):
    c = np.array([x["ce_train"] for x in h])
    s = np.convolve(c, np.ones(10) / 10, mode="valid")
    gain = s[-1] - s[0]
    return int(np.argmax(s >= s[0] + 0.9 * gain)) + 10 if gain > 0 else None


def rows(folder, bench_market):
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*__seed*.json"))):
        r = json.load(open(f)); out.setdefault(r["variant"], []).append(r)
    res = {}
    for v, R in out.items():
        cs = np.array([r["CE"] for r in R])
        res[v] = {"seeds": len(R), "CE": float(cs.mean()), "CE_sd": float(cs.std(ddof=1)) if len(R) > 1 else None,
                  "viol": float(np.mean([r["viol"] for r in R])),
                  "shift_CE": float(np.mean([r["shift"]["CE"] for r in R])) if all("shift" in r for r in R) else None,
                  "shift_viol": float(np.mean([r["shift"]["viol"] for r in R])) if all("shift" in r for r in R) else None,
                  "train_seconds": float(np.mean([r["train_seconds"] for r in R])),
                  "iters_to_90pct": (float(np.mean([eff(r["history"]) for r in R])) if R[0]["history"] else None),
                  "max_grad_norm": (float(max(max(x["grad_norm"] for x in r["history"]) for r in R)) if R[0]["history"] else None)}
    if bench_market:   # variants trained in run_benchmark.py (no shift evaluation)
        S = json.load(open(os.path.join(ROOT, "results", "tables", "summary.json")))[bench_market]
        H = [json.loads(l) for l in open(os.path.join(ROOT, "results", "training_histories.jsonl"))]
        for v in ("E-CBLPO (no GZ prior)", "E-CBLPO (MLP)"):
            if v not in S:
                continue
            hs = [h for h in H if h.get("market") == bench_market and h.get("variant") == v]
            cs = [ce(np.load(os.path.join(ROOT, "results", bench_market, "raw", f"{v.replace(' ', '_')}__eval{k}.npz"))["W_T"])
                  for k in range(5) if os.path.exists(os.path.join(ROOT, "results", bench_market, "raw", f"{v.replace(' ', '_')}__eval{k}.npz"))]
            res[v] = {"seeds": len(cs), "CE": S[v]["CE"], "CE_sd": float(np.std(cs, ddof=1)) if len(cs) > 1 else None,
                      "viol": S[v]["viol_rate"], "shift_CE": None, "shift_viol": None,
                      "train_seconds": None,   # trained earlier under a different machine load; not comparable
                      "iters_to_90pct": float(np.mean([eff(h["history"]) for h in hs])) if hs else None, "max_grad_norm": None}
    return res


out = {"heston": rows(A, "heston"), "jump": rows(os.path.join(A, "jump"), "jump")}
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "ablation_summary.json"), "w"), indent=1)
lab = {"full": "\\method{} (GZ + residual + projection)", "GZ + projection (no residual)": "GZ direction + projection, no network",
       "E-CBLPO (no GZ prior)": "residual + projection, no GZ prior", "E-CBLPO (MLP)": "non-equivariant (MLP)",
       "penalty, no projection (lam=10)": "penalty $\\lambda=10$, no projection", "penalty, no projection (lam=100)": "penalty $\\lambda=100$, no projection"}
order = ["full", "GZ + projection (no residual)", "E-CBLPO (no GZ prior)", "E-CBLPO (MLP)", "penalty, no projection (lam=10)", "penalty, no projection (lam=100)"]
f = lambda x, fmt: "--" if x is None else fmt.format(x)
L = [r"\begin{tabular}{l ccc cc cc}", r"\toprule",
     r" & \multicolumn{3}{c}{training market} & \multicolumn{2}{c}{with crashes} & \multicolumn{2}{c}{training}\\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Variant & CE & s.d. & viol. & CE & viol. & iter.\ to 90\% & s/run\\", r"\midrule"]
for mk, name in (("heston", "Heston, $N=5$"), ("jump", "Jump diffusion, $N=1$")):
    L.append(r"\multicolumn{8}{l}{\emph{" + name + r"}}\\")
    for v in order:
        if v not in out[mk]:
            continue
        r = out[mk][v]
        L.append(f"{lab[v]} & {r['CE']:.4f} & {f(r['CE_sd'], '{:.4f}')} & {100*r['viol']:.2f}\\% & {f(r['shift_CE'], '{:.4f}')} & "
                 f"{f(None if r['shift_viol'] is None else 100*r['shift_viol'], '{:.1f}\\%')} & {f(r['iters_to_90pct'], '{:.0f}')} & {f(r['train_seconds'], '{:.0f}')}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tables", "table_ablation.tex"), "w").write("\n".join(L) + "\n")
print(json.dumps(out, indent=1))
