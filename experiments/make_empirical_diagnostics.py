"""Aggregate results/empirical/diagnostics/E1 with the main E1 results (item 15).
CE over all episodes; differences on identical episodes with a bootstrap over evaluation years."""
import glob, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMP = os.path.join(ROOT, "results", "empirical")
G = 2.0
ce = lambda W: float(np.mean(np.asarray(W) ** (1 - G)) ** (1 / (1 - G)))
years = sorted(int(os.path.basename(f)[:4]) for f in glob.glob(os.path.join(EMP, "diagnostics", "E1", "*.json")))
cols = {"plug-in": {}, "zero cost": {}, "oracle drift": {}}
yl = []
for y in years:
    base = json.load(open(os.path.join(EMP, "E1", f"{y}.json")))
    d = json.load(open(os.path.join(EMP, "diagnostics", "E1", f"{y}.json")))
    n = len(base["starts"]); yl += [y] * n
    def add(col, name, W):
        cols[col].setdefault(name, []).append(np.asarray(W, float))
    for name, v in base["strategies"].items():
        if name.startswith("ECP (seed"):
            continue
        add("plug-in", name, v["W_T"])
    for k in range(3):
        add("plug-in", f"ECP{k}", base["strategies"][f"ECP (seed {k})"]["W_T"])
        add("zero cost", f"ECP{k}", d["zero_cost"][f"ECP (seed {k})"]["W_T"])
    for name, v in d["zero_cost"].items():
        if not name.startswith("ECP"):
            add("zero cost", name, v["W_T"])
    for name, v in d["oracle"].items():
        add("oracle drift", name, v["W_T"])
yl = np.array(yl)
for c in cols:
    for k in list(cols[c]):
        cols[c][k] = np.concatenate(cols[c][k])
    cols[c] = {k: v for k, v in cols[c].items() if len(v) == len(yl)}
    if "ECP0" in cols[c]:
        cols[c]["ECP"] = None

def CE(c, name, idx=None):
    if name == "ECP":
        return float(np.mean([CE(c, f"ECP{k}", idx) for k in range(3)]))
    W = cols[c][name] if idx is None else cols[c][name][idx]
    return ce(W)

rng = np.random.default_rng(0)
uy = np.unique(yl); groups = [np.where(yl == y)[0] for y in uy]
boot = [np.concatenate([groups[i] for i in rng.integers(0, len(uy), len(uy))]) for _ in range(2000)]
def diff(c1, a, c2, b):
    pt = 1e4 * (CE(c1, a) - CE(c2, b))
    bs = [1e4 * (CE(c1, a, i) - CE(c2, b, i)) for i in boot]
    return [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

names = ["TIPP (m=5)", "TIPP (m=3)", "Grossman-Zhou", "Merton (unconstrained)", "HJB optimum (continuous-time)",
         "HJB optimum (certified class)", "ECP"]
out = {"CE": {c: {n: CE(c, n) for n in names if n in cols[c]} for c in cols}}
cert = "HJB optimum (certified class)"
out["diff_bp"] = {
    "TIPP5 - cert HJB, plug-in": diff("plug-in", "TIPP (m=5)", "plug-in", cert),
    "TIPP5 - cert HJB, zero cost": diff("zero cost", "TIPP (m=5)", "zero cost", cert),
    "TIPP5 - cert HJB, oracle drift": diff("plug-in", "TIPP (m=5)", "oracle drift", cert),
    "TIPP5 - ECP, plug-in": diff("plug-in", "TIPP (m=5)", "plug-in", "ECP"),
    "TIPP5 - ECP, zero cost": diff("zero cost", "TIPP (m=5)", "zero cost", "ECP"),
    "ECP - cert HJB, plug-in": diff("plug-in", "ECP", "plug-in", cert),
    "ECP - cert HJB, zero cost": diff("zero cost", "ECP", "zero cost", cert),
    "cert HJB oracle - plug-in": diff("oracle drift", cert, "plug-in", cert),
    "GZ oracle - plug-in": diff("oracle drift", "Grossman-Zhou", "plug-in", "Grossman-Zhou"),
}
# realised vs plug-in premium
pe, oe = [], []
for y in years:
    d = json.load(open(os.path.join(EMP, "diagnostics", "E1", f"{y}.json")))
    b = json.load(open(os.path.join(EMP, "E1", f"{y}.json")))
    for c in d["oracle_coeffs"]:
        pe.append(b["ex"][0]); oe.append(c[0])
out["premium"] = {"plugin_mean": float(np.mean(pe)), "realised_mean": float(np.mean(oe)),
                  "corr": float(np.corrcoef(pe, oe)[0, 1])}
json.dump(out, open(os.path.join(EMP, "diagnostics", "summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
