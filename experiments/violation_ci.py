"""Item 14: one-sided 95% Clopper-Pearson upper bounds for the violation probability of the certified
policies when no violation is observed. Episodes within a year overlap, so for the data we count
independent units as evaluation years (E1: Januaries only give the same count)."""
import glob, json, os
from scipy.stats import beta
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMP = json.load(open(os.path.join(ROOT, "results", "empirical", "summary.json")))
ub = lambda k, n: float(beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0
out = {"simulation, 20000 paths": ub(0, 20000)}
for e in ("E1", "E2", "E3"):
    ny = len(set(int(os.path.basename(f)[:4]) for f in glob.glob(os.path.join(ROOT, "results", "empirical", e, "[0-9]*.json"))))
    out[f"{e}: 0 of {ny} years"] = ub(0, ny)
    if e != "E3":
        out[f"{e}: 0 of {EMP[e]['episodes']} episodes (overlapping, not valid)"] = ub(0, EMP[e]["episodes"])
json.dump(out, open(os.path.join(ROOT, "results", "empirical", "violation_ci.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
