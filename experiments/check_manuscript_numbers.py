"""
Recompute the numbers quoted in the manuscript text from results/ and compare them with the
values written in the text.  Prints one line per claim: OK or MISMATCH.
(The four main tables and the empirical tables are generated from the same files and are not
re-checked here.)
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
bad = 0


def ce(W, g=2.0):
    W = np.asarray(W, float)
    return float(np.mean(W ** (1 - g)) ** (1 / (1 - g)))


def raw(market, name):
    fs = sorted(glob.glob(os.path.join(RES, market, "raw", name.replace(" ", "_") + "__eval*.npz")))
    return [np.load(f) for f in fs]


def paired(market, a, b):
    A, B = raw(market, a), raw(market, b)
    d = np.array([(ce(x["W_T"]) - ce(y["W_T"])) * 1e4 for x, y in zip(A, B)])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def check(label, claimed, value, tol):
    global bad
    ok = abs(claimed - value) <= tol
    bad += not ok
    print(f"{'OK      ' if ok else 'MISMATCH'} {label}: text {claimed}  computed {value:.4g}")


S = json.load(open(os.path.join(RES, "tables", "summary.json")))
# --- Section 6: main comparisons
for mk, a, b, cl, cls in [("constant_bs", "E-CBLPO", "TIPP (m=3)", 31.0, 1.4),
                          ("constant_bs", "E-CBLPO", "HJB optimum (continuous-time)", -6.5, 1.9),
                          ("constant_bs", "E-CBLPO", "HJB optimum (certified class)", 9.4, 1.1),
                          ("constant_bs", "E-CBLPO", "Grossman-Zhou", 79.5, 3.5),
                          ("kim_omberg", "E-CBLPO", "TIPP (m=3)", 37.5, 1.8),
                          ("kim_omberg", "E-CBLPO", "Grossman-Zhou", 81.9, 6.0),
                          ("kim_omberg", "E-CBLPO", "HJB optimum (certified class)", 23.6, 0.8),
                          ("kim_omberg", "E-CBLPO", "HJB optimum (continuous-time)", 9.6, 2.1),
                          ("heston", "E-CBLPO", "TIPP (m=5)", 89.7, 2.2),
                          ("heston", "E-CBLPO", "Grossman-Zhou", 271.1, 6.4),
                          ("heston", "E-CBLPO", "E-CBLPO (MLP)", 11.4, 1.0),
                          ("heston", "E-CBLPO (no GZ prior)", "E-CBLPO", 7.7, 3.6)]:
    m, se = paired(mk, a, b)
    check(f"{mk}: {a} - {b} (bp)", cl, m, 0.06)
    check(f"{mk}: {a} - {b} s.e.", cls, se, 0.06)
check("BS certified HJB loss vs continuous (bp)", 16,
      (S["constant_bs"]["HJB optimum (continuous-time)"]["CE"] - S["constant_bs"]["HJB optimum (certified class)"]["CE"]) * 1e4, 0.6)
check("BS HJB continuous violations (%)", 6.8, 100 * S["constant_bs"]["HJB optimum (continuous-time)"]["viol_rate"], 0.05)
check("KO HJB continuous violations (%)", 7.6, 100 * S["kim_omberg"]["HJB optimum (continuous-time)"]["viol_rate"], 0.05)
check("Heston Merton violations (%)", 34.1, 100 * S["heston"]["Merton (unconstrained)"]["viol_rate"], 0.05)
check("Jump Merton violations (%)", 0.10, 100 * S["jump"]["Merton (unconstrained)"]["viol_rate"], 0.0051)
check("BS GZ gross", 0.23, S["constant_bs"]["Grossman-Zhou"]["gross"], 0.005)
check("Heston GZ gross", 1.06, S["heston"]["Grossman-Zhou"]["gross"], 0.005)
for mk, v in [("constant_bs", 6.0), ("kim_omberg", 7.1), ("heston", 59.8), ("jump", 0.8)]:
    check(f"{mk}: PPO-trained over GZ (bp)", v, S[mk]["E-CBLPO (PPO-trained)"]["dCE_vs_GZ_bp"], 0.06)
g = [S[m]["E-CBLPO"]["gross"] for m in ["constant_bs", "kim_omberg", "heston"]]
check("ECP gross min (diffusion markets)", 0.54, min(g), 0.006)
check("ECP gross max (diffusion markets)", 0.78, max(g), 0.006)
check("ECP gross jump", 0.23, S["jump"]["E-CBLPO"]["gross"], 0.005)
# zero-shot N=50
Z = json.load(open(os.path.join(RES, "extensions", "zeroshot_N50.json")))
seeds = [k for k in Z if k.isdigit()]
ces = np.array([Z[k]["E-CBLPO (trained N=5, zero-shot)"]["CE"] for k in seeds])
check("N=50 ECP CE", 1.0679, ces.mean(), 0.00006)
check("N=50 ECP CE s.e.", 0.0015, ces.std(ddof=1) / np.sqrt(len(ces)), 0.00006)
d = np.array([(Z[k]["E-CBLPO (trained N=5, zero-shot)"]["CE"] - Z[k]["TIPP (m=5)"]["CE"]) * 1e4 for k in seeds])
check("N=50 ECP - TIPP5 (bp)", 75.3, d.mean(), 0.06)
check("N=50 GZ CE", 0.998, np.mean([Z[k]["Grossman-Zhou"]["CE"] for k in seeds]), 0.0006)
check("N=50 Merton CE", 1.009, np.mean([Z[k]["Merton (unconstrained)"]["CE"] for k in seeds]), 0.0006)
# --- Section 4: price of safety table
P = json.load(open(os.path.join(RES, "extensions", "price_of_safety.json")))["0.2"]
gaps = {float(r["K"]): r["gap_bp"] for r in P if r["K"] != "inf"}
for K, v in [(1, 123), (2, 61), (3, 27), (4.7, 8.7), (7, 3.4), (10, 1.5), (15, 0.6)]:
    check(f"price K={K} (bp)", v, gaps[K], 0.5 if v >= 10 else 0.051)
viol = {float(r["K"]): r["viol"] for r in P if r["K"] != "inf"}
check("violations for K<=15 (max %)", 0, 100 * max(v for K, v in viol.items() if K <= 15), 0)
check("violations at K=25 (%) > 0", 1, float(viol[25.0] > 0), 0)
# price rate (extrapolated)
ext = {}
for c in ["r=0.02", "r=0.005", "r=0"]:
    a = json.load(open(os.path.join(RES, "extensions", f"price_rate_spd32_{c}.json")))[c]
    b = json.load(open(os.path.join(RES, "extensions", f"price_rate_spd64_{c}.json")))[c]
    ext[c] = (np.array(b["K"], float), 2 * np.array(b["gap_bp"]) - np.array(a["gap_bp"]), b)
K = ext["r=0"][0]
sl = lambda g: np.diff(np.log(g)) / np.diff(np.log(K))
check("rate r=2% slope min", -2.24, sl(ext["r=0.02"][1]).min(), 0.006)
check("rate r=2% slope max", -2.20, sl(ext["r=0.02"][1]).max(), 0.006)
check("rate r=0 first slope", -2.46, sl(ext["r=0"][1])[0], 0.006)
check("rate r=0 steepest slope", -2.90, sl(ext["r=0"][1]).min(), 0.006)
check("interest component K^2 limit r=2% (bp)", 64, (K ** 2 * (ext["r=0.02"][1] - ext["r=0"][1]))[-3:].mean(), 1.0)
check("interest component K^2 limit r=0.5% (bp)", 25, (K ** 2 * (ext["r=0.005"][1] - ext["r=0"][1]))[-3:].mean(), 1.0)
lay = ext["r=0.02"][2]["layer"]
check("layer/pred K=100", 1.10, lay["100"] / (4 * 0.02 * 0.8 / (0.04 * 100 ** 2)), 0.006)
check("layer/pred K=200", 1.03, lay["200"] / (4 * 0.02 * 0.8 / (0.04 * 200 ** 2)), 0.006)
# --- Section 3.7: boundary layer
B = json.load(open(os.path.join(RES, "extensions", "boundary_layer.json")))
fm = B["floor_marginal_value"]
for r, vals in [("0.02", [2.10, 2.24, 2.26, 2.26]), ("0.0", [2.74, 4.17, 5.70, 7.41])]:
    got = fm[r]["v_x"][::-1]                     # stored at y = 1e-9 .. 1e-3
    for c, v in zip(vals, got):
        check(f"v_x(a+) r={r}", c, v, 0.006)
ratios = [row["ratio_t0.5"][0] for row in B["r_positive"]] + [row["ratio_t0.1"][0] for row in B["r_positive"]] + \
         [row["ratio_t0.9"][0] for row in B["r_positive"]]
check("sqrt-law ratio at 1e-6, min", 1.010, min(ratios), 0.0006)
check("sqrt-law ratio at 1e-6, max", 1.047, max(ratios), 0.0006)
ds = [d["median_slope"] / d["prediction"] for d in B["dual_slope_r0"]]
check("dual slope ratio min", 0.90, min(ds), 0.006)
check("dual slope ratio max", 0.93, max(ds), 0.006)
print("\nmismatches:", bad)

# --- zero-cost comparison (Section 6)
def zc(m, ref):
    d = []
    for k in range(3):
        L = json.load(open(os.path.join(RES, "extensions", "zero_cost", f"{m}__learned__{k}.json")))["E-CBLPO"]["CE"]
        R = json.load(open(os.path.join(RES, "extensions", "zero_cost", f"{m}__rules__{k}.json")))[ref]["CE"]
        d.append((L - R) * 1e4)
    d = np.array(d); return d.mean(), d.std(ddof=1) / np.sqrt(3)
for m, ref, cl in [("constant_bs", "HJB optimum (certified class)", -7.8), ("constant_bs", "HJB optimum (continuous-time)", -15.0),
                   ("kim_omberg", "HJB optimum (certified class)", -1.1), ("kim_omberg", "HJB optimum (continuous-time)", -18.6)]:
    check(f"zero-cost {m} ECP - {ref}", cl, zc(m, ref)[0], 0.06)
# --- estimation risk (Section 6)
E_ = os.path.join(RES, "extensions", "estimation")
def est(name):
    return np.array([ce(np.load(os.path.join(E_, f"WT_{name}__{k}.npy"))) for k in range(3)])
ecp = est("E-CBLPO")
for nm, lab, cl in [("HJB_certified_(oracle_mu)", "oracle", -5.3), ("HJB_certified_(Bayes_posterior_mean)", "Bayes", 8.2),
                    ("HJB_certified_(plug-in_mu_hat)", "plug-in", 44.6), ("TIPP_(m=3)", "TIPP3", 34.4), ("Grossman-Zhou", "GZ", 73.2)]:
    check(f"estimation ECP - {lab} (bp)", cl, ((ecp - est(nm)) * 1e4).mean(), 0.06)
check("estimation ECP CE", 1.0390, ecp.mean(), 0.00006)
# --- risk aversion and horizon (Section 6)
SD = os.path.join(RES, "extensions", "sensitivity")
for tag, g, row in [("g1.5_T252", 1.5, (1.0431, 103.5, 39.5, 18.7, -11.3)), ("g3.0_T252", 3.0, (1.0344, 50.1, 29.3, 0.6, -1.6)),
                    ("g5.0_T252", 5.0, (1.0287, 17.9, 67.5, -1.7, -1.6)), ("g2.0_T126", 2.0, (1.0197, 41.7, 15.8, 5.0, -2.8)),
                    ("g2.0_T504", 2.0, (1.0797, 158.4, 60.3, 19.3, -2.8))]:
    f = lambda n: np.array([ce(np.load(os.path.join(SD, f"WT_{tag}_{n}__{k}.npy")), g) for k in range(3)])
    e = f("E-CBLPO")
    check(f"sens {tag} ECP CE", row[0], e.mean(), 0.00006)
    for n, cl in zip(["Grossman-Zhou", "TIPP_(m=3)", "HJB_optimum_(certified_class)", "HJB_optimum_(continuous-time)"], row[1:]):
        check(f"sens {tag} ECP - {n}", cl, ((e - f(n)) * 1e4).mean(), 0.06)
# --- empirical (Section 7) text numbers
EM = json.load(open(os.path.join(RES, "empirical", "summary.json")))
r1 = EM["E1"]["rows"]
check("E1 HJB cont viol (%)", 22.1, 100 * r1["HJB optimum (continuous-time)"]["viol"], 0.05)
check("E1 HJB cont worst DD (%)", 28.8, 100 * r1["HJB optimum (continuous-time)"]["worst_maxdd"], 0.05)
check("E1 cert worst DD (%)", 19.9, 100 * r1["HJB optimum (certified class)"]["worst_maxdd"], 0.05)
check("E1 ECP worst DD (%)", 19.9, 100 * r1["ECP"]["worst_maxdd"], 0.05)
check("E1 cert loss vs cont (bp)", 25, (r1["HJB optimum (continuous-time)"]["CE"] - r1["HJB optimum (certified class)"]["CE"]) * 1e4, 0.5)
check("E1 K=1 loss vs cont (bp)", 308, (r1["HJB optimum (continuous-time)"]["CE"] - r1["HJB optimum (robust class, K=1)"]["CE"]) * 1e4, 0.5)
for ref, v in [("Grossman-Zhou", (184, 56, 318)), ("TIPP (m=3)", (37, -44, 118)), ("HJB optimum (certified class)", (-26, -45, -9)),
               ("TIPP (m chosen ex ante)", (-151, -309, -26))]:
    got = r1["ECP"][f"d_vs_{ref}"]
    for c, x in zip(v, got):
        check(f"E1 ECP vs {ref}", c, x, 0.55)
check("E1 TIPP5 CE", 1.0858, r1["TIPP (m=5)"]["CE"], 0.00006)
check("E1 cert CE", 1.0805, r1["HJB optimum (certified class)"]["CE"], 0.00006)
check("E1 ex-ante TIPP CE", 1.0930, r1["TIPP (m chosen ex ante)"]["CE"], 0.00006)
check("E1 ex-ante TIPP viol (%)", 2.3, 100 * r1["TIPP (m chosen ex ante)"]["viol"], 0.05)
cr = EM["E1"]["crises"]
check("E1 crisis HJB cont violations total", 70, sum(v["viol"]["HJB optimum (continuous-time)"] for v in cr.values()), 0)
check("E1 crisis episodes total", 75, sum(v["episodes"] for v in cr.values()), 0)
r2 = EM["E2"]["rows"]
check("E2 GZ CE", 1.0335, r2["Grossman-Zhou"]["CE"], 0.00006)
check("E2 GZ gross", 1.59, r2["Grossman-Zhou"]["gross"], 0.006)
check("E2 ECP CE", 1.0378, r2["ECP"]["CE"], 0.00006)
check("E2 TIPP3 CE", 1.0538, r2["TIPP (m=3)"]["CE"], 0.00006)
check("E2 ex-ante TIPP CE", 1.0557, r2["TIPP (m chosen ex ante)"]["CE"], 0.00006)
for ref, v in [("Grossman-Zhou", (44, -237, 296)), ("TIPP (m=3)", (-160, -356, 26)), ("TIPP (m chosen ex ante)", (-178, -469, 91))]:
    for c, x in zip(v, r2["ECP"][f"d_vs_{ref}"]):
        check(f"E2 ECP vs {ref}", c, x, 0.55)
check("E2 Merton viol (%)", 21, 100 * r2["Merton (unconstrained)"]["viol"], 0.5)
check("E2 mix viol (%)", 9, 100 * r2["Constant mix (60% risky)"]["viol"], 0.5)
check("E2 CPPI viol (%)", 4, 100 * r2["CPPI (m=3)"]["viol"], 0.5)
check("E2 stress-exceeded episodes", 12, EM["E2"]["stress_exceed_episodes"], 0)
r3 = EM["E3"]["rows"]
check("E3 ECP CE", 1.0539, r3["ECP"]["CE"], 0.00006)
check("E3 GZ CE", 1.0541, r3["Grossman-Zhou"]["CE"], 0.00006)
check("E3 GZ gross", 1.92, r3["Grossman-Zhou"]["gross"], 0.006)
check("E3 GZ violating years", 2, r3["Grossman-Zhou"]["viol_n"], 0)
check("E3 stress years", 10, EM["E3"]["stress_exceed_episodes"], 0)
for c, x in zip((50, -151, 286), r3["ECP"]["d_vs_TIPP (m=3)"]):
    check("E3 ECP vs TIPP3", c, x, 0.55)
print("\nmismatches (total):", bad)
