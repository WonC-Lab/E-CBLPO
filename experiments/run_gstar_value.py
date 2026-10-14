"""
Item 11 (theory side): value of the larger multiplier cap of (G*) in the Black-Scholes setting of the
paper (r = 2%, mu = 8%, sigma = 20%, gamma = 2, alpha = 20%, T = 1): certified HJB values at the caps of
(G) with eta = 0.05 and eta = 0, and of (G*) with and without the same margin.
Output: results/extensions/gstar_value.json
"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.agents.hjb_graded import solve_hjb_graded
ELL, C, ETA = 0.20, 0.001, 0.05
caps = {"G (eta=0.05)": (1 - ETA) / (ELL + 2 * C), "G* (eta=0.05 margin)": (1 - ETA) / (ELL + C),
        "G (eta=0)": 1 / (ELL + 2 * C), "G* (eta=0)": 1 / (ELL + C)}
out = {}
for spd in (32, 64):
    kw = dict(mu=0.08, sigma=0.2, r=0.02, alpha=0.2, gamma=2.0, T=1.0, M=2000, q=3.5, steps_per_day=spd, keep_times=())
    V = solve_hjb_graded(**kw)["ce"]
    out[f"spd{spd}"] = {name: {"K": K, "gap_bp": 1e4 * (V - solve_hjb_graded(K_cap=K, **kw)["ce"])} for name, K in caps.items()}
    print(spd, {k: round(v["gap_bp"], 4) for k, v in out[f"spd{spd}"].items()}, flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "extensions", "gstar_value.json"), "w"), indent=1)
