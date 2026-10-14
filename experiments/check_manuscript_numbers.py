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
# ---- items 3, 5, 7, 8, 9: uniform bound, r-scaling, localisation, solver checks, rate table
X = os.path.join(ROOT, "results", "extensions")
RS = json.load(open(os.path.join(X, "r_scaling.json")))
for k, v, b in [("0.04", 0.70, 1.20), ("0.02", 0.81, 1.51), ("0.01", 0.92, 1.95), ("0.005", 1.01, 2.57),
                ("0.0025", 1.10, 3.45), ("0.00125", 1.18, 4.69)]:
    check(f"ln v_x(0,a+) r={k}", v, RS[k]["ln_vx_1e-7"], 0.0051)
    check(f"uniform bound r={k}", b, RS[k]["bound_ln_vx"], 0.0051)
ks = ["0.04", "0.02", "0.01", "0.005", "0.0025", "0.00125"]
inc = [(RS[ks[i + 1]]["ln_vx_1e-7"] - RS[ks[i]]["ln_vx_1e-7"]) /
       (np.sqrt(np.log(1 / float(ks[i + 1]))) - np.sqrt(np.log(1 / float(ks[i])))) for i in range(5)]
check("sqrt-log increment min", 0.57, min(inc), 0.0051); check("sqrt-log increment max", 0.65, max(inc), 0.0051)
SA = json.load(open(os.path.join(X, "solver_checks_A.json")))
for j, lim in [(0, 0.1), (1, 0.3), (2, 3.9)]:
    worst = max(100 * (max(SA[f"{c}|{v}"]["m"][j] for v in ("base", "M4000", "q3", "q4", "spd32", "howard2", "pimax4")) /
                       min(SA[f"{c}|{v}"]["m"][j] for v in ("base", "M4000", "q3", "q4", "spd32", "howard2", "pimax4")) - 1)
                for c in ("r=0.02", "r=0"))
    check(f"multiplier spread across variants y-index {j} (%) <=", lim, max(worst, lim) if worst <= lim else worst, 0.0)
check("m sqrt(y) at 1e-7, r=2%", 1.28, SA["r=0.02|base"]["m"][2] * np.sqrt(1e-7), 0.0051)
check("v_x(0,a+) r=2% all variants", 2.259, np.mean([SA[f"r=0.02|{v}"]["vx_1e-9"] for v in ("base", "M4000", "q3", "q4", "spd32", "howard2", "pimax4")]), 0.0006)
RT = json.load(open(os.path.join(X, "rate_table.json")))
check("rate r=2% K^2 gap at 10", 155, RT["r=0.02"]["scaled"][0], 0.5); check("rate r=2% K^2 gap at 200", 80, RT["r=0.02"]["scaled"][-1], 0.5)
check("rate r=0 K^3 gap/1e3 at 10", 1.5, RT["r=0"]["scaled"][0] / 1e3, 0.051); check("rate r=0 K^3 gap/1e3 at 200", 3.2, RT["r=0"]["scaled"][-1] / 1e3, 0.051)
check("weakest resolution ratio", 3.9, min(g / e for d in RT.values() for g, e in zip(d["gap_R"], d["err"])), 0.051)
i100 = RT["r=0.02"]["K"].index(100)
check("interest component r=2% at K=100", 64, RT["r=0.02"]["rate_part"][i100], 0.5); check("... error", 4, RT["r=0.02"]["rate_part_err"][i100], 0.5)
check("interest component r=0.5% at K=100", 26, RT["r=0.005"]["rate_part"][i100], 0.5); check("... error", 3, RT["r=0.005"]["rate_part_err"][i100], 0.5)
c0 = np.mean(RT["r=0"]["scaled"][5:])
check("c_0 (bp)", 3000, c0, 100); check("K_x r=2%", 50, c0 / 64, 4); check("K_x r=0.5%", 115, c0 / 26, 4)
check("r=0.5% slope min", -2.47, min(x for x in RT["r=0.005"]["slope"] if x), 0.006); check("r=0.5% slope max", -2.36, max(x for x in RT["r=0.005"]["slope"] if x), 0.006)
LO = json.load(open(os.path.join(X, "localisation.json")))
rel = lambda k, j: 100 * abs(LO[k]["m_hjb"][j] / LO[k]["m_explicit"][j] - 1)
long_ = [k for k in LO if "spd=32" in k and "tau=0.02" not in k]
check("localisation max rel err to 1e-7, tau>=0.05 (%)", 0.5, max(rel(k, j) for k in long_ for j in range(6)), 0.051)
check("localisation rel err to 1e-7, tau=0.02 (%)", 1.3, max(rel("mu=0.06,sigma=0.2,tau=0.02,spd=32", j) for j in range(6)), 0.051)
check("localisation tau=0.02 1e-7 at 8 steps (%)", 8.9, rel("mu=0.06,sigma=0.2,tau=0.02,spd=8", 5), 0.051)

# ---- items 13-15: empirical diagnostics, violation bounds, stress exceedances
DG = json.load(open(os.path.join(ROOT, "results", "empirical", "diagnostics", "summary.json")))["diff_bp"]
for key, vals in [("TIPP5 - cert HJB, zero cost", (74, 30, 132)), ("TIPP5 - cert HJB, plug-in", (53, 12, 106)),
                  ("ECP - cert HJB, zero cost", (-29, -50, -11)), ("TIPP5 - cert HJB, oracle drift", (-486, -746, -245)),
                  ("cert HJB oracle - plug-in", (539, None, None))]:
    for c, x in zip(vals, DG[key]):
        if c is not None:
            check(f"diag {key}", c, x, 0.55)
PR = json.load(open(os.path.join(ROOT, "results", "empirical", "diagnostics", "summary.json")))["premium"]
check("diag plug-in premium (%)", 6.7, 100 * PR["plugin_mean"], 0.051); check("diag realised drift (%)", 9.6, 100 * PR["realised_mean"], 0.051)
check("diag corr", -0.15, PR["corr"], 0.0051)
VC = json.load(open(os.path.join(ROOT, "results", "empirical", "violation_ci.json")))
check("CP E1 44 years (%)", 6.6, 100 * VC["E1: 0 of 44 years"], 0.051); check("CP 21 years (%)", 13.3, 100 * VC["E3: 0 of 21 years"], 0.051)
check("CP E1 524 episodes (%)", 0.57, 100 * VC["E1: 0 of 524 episodes (overlapping, not valid)"], 0.0051)
check("CP sim (%)", 0.015, 100 * VC["simulation, 20000 paths"], 0.00051)
SX = json.load(open(os.path.join(ROOT, "results", "empirical", "stress_exceedance.json")))
e3 = [x for y in SX["E3"].values() for x in y["seed 0"]["events"]]
check("E3 exceedance days", 69, len(e3), 0); check("E3 falling", 34, sum(min(x["excess"]) < -0.2 for x in e3), 0)
check("E3 max stocks per day", 7, max(len(x["assets"]) for x in e3), 0)
e3a = [x for y in SX["E3"].values() for k, v in y.items() for x in v["events"] if min(x["excess"]) < -0.2]
check("E3 max weight in falling stock (%)", 5.1, 100 * max(max(abs(t) for t in x["weights"]) for x in e3a), 0.051)
check("E3 worst port ret (%)", -3.8, 100 * min(x["port_ret"] for x in e3a), 0.051)
check("E3 max cushion lost (%)", 54, 100 * max(x["cushion_lost_frac"] for x in e3a), 0.51)
e2 = [x for y in SX["E2"].values() for k, v in y.items() if k.startswith("seed") for x in v["events"]]
check("E2 max XLE weight (%)", 4, 100 * max(max(abs(t) for t in x["weights"]) for x in e2), 0.051)
check("E2 max cushion lost (%)", 31, 100 * max(x["cushion_lost_frac"] for x in e2), 0.51)
check("E2 CPPI violations 2019-20", 6, sum(sum(v["violated"]) for y in SX["E2"].values() for k, v in y.items() if k == "CPPI (m=3)"), 0)

# ---- item 12: N-scaling
NS = json.load(open(os.path.join(ROOT, "results", "extensions", "nscaling_summary.json")))
lead = [NS[c]["ECP_minus_TIPP5_bp"][0] for c in ("N5", "N10", "N20", "N50", "N100")]
check("N-scaling lead min", 68, min(lead), 0.55); check("N-scaling lead max", 81, max(lead), 0.55)
check("N-scaling viol max", 0, max(NS[c]["ECP"]["viol"] for c in NS), 0)
check("us/decision N5", 4, NS["N5"]["us_per_decision"], 0.5); check("us/decision N100", 68, NS["N100"]["us_per_decision"], 0.55)
check("N50 lead (reproduction)", 75.3, NS["N50"]["ECP_minus_TIPP5_bp"][0], 0.051)
check("rho=0 GZ over ECP (bp)", 99, 1e4 * (NS["N20_corr0"]["Grossman-Zhou"]["CE"] - NS["N20_corr0"]["ECP"]["CE"]), 0.55)
check("cov err ECP loss (bp)", 16, 1e4 * (NS["N20"]["ECP"]["CE"] - NS["N20_coverr"]["ECP"]["CE"]), 0.55)
check("cov err GZ loss (bp)", 40, 1e4 * (NS["N20"]["Grossman-Zhou"]["CE"] - NS["N20_coverr"]["Grossman-Zhou"]["CE"]), 0.55)

# ---- item 11: ablations
AB = json.load(open(os.path.join(ROOT, "results", "extensions", "ablation_summary.json")))
h, j = AB["heston"], AB["jump"]
check("abl GZ+proj heston", 1.0388, h["GZ + projection (no residual)"]["CE"], 0.00006)
check("abl full heston", 1.0613, h["full"]["CE"], 0.00006)
check("abl no residual gap (bp)", 225, 1e4 * (h["full"]["CE"] - h["GZ + projection (no residual)"]["CE"]), 0.55)
check("abl GZ+proj jump", 1.0220, j["GZ + projection (no residual)"]["CE"], 0.00006); check("abl full jump", 1.0233, j["full"]["CE"], 0.00006)
check("abl pen10", 1.0680, h["penalty, no projection (lam=10)"]["CE"], 0.00006); check("abl pen100", 1.0685, h["penalty, no projection (lam=100)"]["CE"], 0.00006)
check("abl pen viol max (%)", 0.02, 100 * max(h[v]["viol"] for v in h if "penalty" in v), 0.0051)
check("abl pen shift viol min (%)", 81, 100 * min(h[v]["shift_viol"] for v in h if "penalty" in v), 0.51)
check("abl pen shift viol max (%)", 82, 100 * max(h[v]["shift_viol"] for v in h if "penalty" in v), 0.51)
check("abl full shift viol", 0, h["full"]["shift_viol"], 0)
check("abl shift CE loss (pp) min", 6, 100 * min(h["full"]["shift_CE"] - h[v]["shift_CE"] for v in h if "penalty" in v), 0.51)
check("abl grad full", 11, h["full"]["max_grad_norm"], 0.5)
check("abl grad pen min", 65, min(h[v]["max_grad_norm"] for v in h if "penalty" in v), 0.5)
check("abl grad pen max", 93, max(h[v]["max_grad_norm"] for v in h if "penalty" in v), 0.5)
check("abl sd pen10", 0.0046, h["penalty, no projection (lam=10)"]["CE_sd"], 0.00006); check("abl sd full", 0.0014, h["full"]["CE_sd"], 0.00006)
check("abl jump identical", 0, max(abs(j[v]["CE"] - j["full"]["CE"]) for v in j if "penalty" in v), 1e-9)

# ---- item 13: block vs iid bootstrap
BC = json.load(open(os.path.join(ROOT, "results", "empirical", "bootstrap_comparison.json")))
check("block turnover pre-1990", 10.0, BC["block20"]["turnover_pre1990"], 0.051); check("iid turnover pre-1990", 1.35, BC["iid"]["turnover_pre1990"], 0.0051)
check("block-iid pre-1990 (bp)", -106, BC["block_minus_iid_pre1990_bp"], 0.55); check("block-iid post-1990 (bp)", -14, BC["block_minus_iid_post1990_bp"], 0.55)
check("turnover post-1990 iid", 4.5, BC["iid"]["turnover_post1990"], 0.051); check("turnover post-1990 block", 4.5, BC["block20"]["turnover_post1990"], 0.051)
for c, x in zip((-30, -103, 19), BC["block_minus_iid_bp"]):
    check("block-iid all (bp)", c, x, 0.55)
check("block viol", 0, BC["block20"]["viol"], 0)
for k, c in [("1962-1981", 0.25), ("1970-1989", 0.20), ("1990-2009", -0.03)]:
    check(f"autocorr {k}", c, BC["lag1_autocorr"][k], 0.0051)

# ---- item 5: smooth fit observations; floor equation consistency
SF = json.load(open(os.path.join(X, "smooth_fit_check.json")))
for t, c in [("0.1", 1.260), ("0.5", 1.262), ("0.9", 1.265)]:
    check(f"exposure/sqrt(y) at 1e-7, t={t}", c, SF["times"][t]["exposure_over_sqrt_y"][-1], 0.0006)
check("2 sqrt(ra)/sigma", 1.265, SF["predicted_sqrt_constant"], 0.0006)
yzz = [SF["times"][t]["Yzz"][-1] for t in ("0.1", "0.5", "0.9")]
check("Yzz min", 0.359, min(yzz), 0.0006); check("Yzz max", 0.363, max(yzz), 0.0006)
check("2ra/theta^2", 0.356, SF["predicted_Yzz"], 0.0006)
import math
R0 = json.load(open(os.path.join(X, "rate_table.json")))["r=0"]
q = [sc / math.log(K) for K, sc in zip(R0["K"], R0["scaled"])]
check("K^3 gap / ln K min (r=0)", 600, min(q), 5); check("K^3 gap / ln K max (r=0)", 710, max(q), 5)

# ---- item 9/11: (G*)
GS = json.load(open(os.path.join(X, "gstar_check.json")))
check("G* violations", 0, sum(GS["violations"].values()), 0)
check("G* simulated steps (1e7)", 1.5, GS["counts"]["v_steps"] / 1e7, 0.051)
check("G* random instances (1e6)", 1.0, (GS["counts"]["i"] + GS["counts"]["iii"] + GS["counts"]["iv"]) / 1e6, 0.051)
check("cap G*", 4.975, GS["caps_limit"]["Gstar"], 0.0006); check("cap G eta0", 4.926, GS["caps_limit"]["G_eta0"], 0.0006)
check("cap G eta005", 4.680, GS["caps_limit"]["G_eta005"], 0.0006)
GV = json.load(open(os.path.join(X, "gstar_value.json")))["spd64"]
check("price G eta .05", 8.78, GV["G (eta=0.05)"]["gap_bp"], 0.0051); check("price G* eta .05", 8.68, GV["G* (eta=0.05 margin)"]["gap_bp"], 0.0051)
check("price G eta 0", 7.75, GV["G (eta=0)"]["gap_bp"], 0.0051); check("price G* eta 0", 7.66, GV["G* (eta=0)"]["gap_bp"], 0.0051)

# ---- item 10: primal vs dual
PD = json.load(open(os.path.join(X, "primal_dual.json")))
pr, du = PD["primal"]["spd32"], PD["dual"]["N8000_spd32"]
for i, (cp, cd) in enumerate([(44.64, 44.62), (131.2, 131.1), (405.5, 405.2)]):
    check(f"primal m {i}", cp, pr["m"][i], 0.051 if cp < 100 else 0.051); check(f"dual m {i}", cd, du["m"][i], 0.051)
check("primal ln vx", 0.8143, pr["ln_vx_a"]["0.0"], 0.00006); check("dual ln vx", 0.8152, du["ln_vx_a"]["0.0"], 0.00006)
check("CE diff (bp)", 0.7, 1e4 * (du["ce"] - pr["ce"]), 0.051)
check("Richardson primal CE", 1.042709, PD["primal_M"]["richardson_space"], 0.0000006); check("dual CE", 1.042709, du["ce"], 0.0000006)

# ---- item 12: cost convention
CS = json.load(open(os.path.join(X, "cost_sensitivity_summary.json")))
check("cap target", 4.70, CS["caps"]["target"], 0.0051); check("cap drift", 4.68, CS["caps"]["drift (lambda c)"], 0.0051)
check("lambda", 1.48, CS["caps"]["lambda"], 0.0051)
dd = [1e4 * (r["drift"]["CE"] - r["target"]["CE"]) for m in ("constant_bs", "kim_omberg", "E1") for r in CS[m].values()]
check("cost conv CE change min", -3.0, min(dd), 0.051); check("cost conv CE change max", 5.9, max(dd), 0.051)
d = lambda m, n: 1e4 * (CS[m][n]["drift"]["CE"] - CS[m][n]["target"]["CE"])
check("ECP BS", -3.0, d("constant_bs", "ECP"), 0.051); check("ECP KO", -1.3, d("kim_omberg", "ECP"), 0.051); check("ECP E1", 1.9, d("E1", "ECP"), 0.051)
rules = [d(m, n) for m in ("constant_bs", "kim_omberg", "E1") for n in ("TIPP (m=3)", "TIPP (m=5)", "CPPI (m=3)")]
check("rules gain min", 3, min(rules), 0.5); check("rules gain max", 6, max(rules), 0.5)
check("viol/maxdd unchanged (max abs diff)", 0, max(max(abs(r["drift"]["viol"] - r["target"]["viol"]), abs(r["drift"]["mean_maxdd"] - r["target"]["mean_maxdd"])) for m in ("constant_bs", "kim_omberg", "E1") for r in CS[m].values()), 0.0002)
for m in ("constant_bs", "E1"):
    rk = lambda conv: [n for n, _ in sorted(CS[m].items(), key=lambda kv: -kv[1][conv]["CE"])]
    check(f"ranking preserved {m}", 1, float(rk("target") == rk("drift")), 0)
check("KO GZ - TIPP5 under target (bp)", 0.2, 1e4 * (CS["kim_omberg"]["Grossman-Zhou"]["target"]["CE"] - CS["kim_omberg"]["TIPP (m=5)"]["target"]["CE"]), 0.051)

print("\nmismatches (total):", bad)
