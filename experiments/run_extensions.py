"""
Extension experiments (run after run_benchmark.py):

(a) Zero-shot dimension transfer.  The S_N-equivariant E-CBLPO policies trained on the
    5-asset Heston market (results/checkpoints/heston__E-CBLPO__seed*.pt) are executed,
    without retraining, on a 50-asset Heston market.  Rules are evaluated on the same
    paths.  The weight trajectory of one path is stored for the heat-map figure.

(b) Drawdown-limit sweep (constant-coefficient market, 1 asset): for each alpha,
    rules + both HJB optima are evaluated and E-CBLPO is trained from scratch (3 seeds).

Outputs: results/extensions/*.json, *.npz
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import GrossmanZhou, TIPP, MertonMyopic, Cash
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.pathwise import PathwisePolicy

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(RES, "extensions")
TC, ETA, RHO, T = 0.001, 0.05, 2.0, 252
B_EVAL = int(os.environ.get("B_EVAL", 4000))
PW_ITERS = int(os.environ.get("PW_ITERS", 200))
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]


def ce(W):
    return float(np.mean(W ** (1 - RHO)) ** (1 / (1 - RHO)))


def met(W_T, max_dd, gross, alpha):
    return {"CE": ce(W_T), "viol": float(np.mean(max_dd > alpha + 1e-9)), "mean_maxdd": float(np.mean(max_dd)), "gross": float(np.mean(gross))}


def eval_rule(rule, market_name, kw, seed, alpha):
    env = DrawdownPortfolioEnv(make_market(market_name, **kw), batch_size=B_EVAL, horizon=T, alpha=alpha, tc=TC)
    o = env.reset(seed)
    for _ in range(T):
        o, *_ = env.step(rule.act(o))
    s = env.summary()
    return met(s["W_T"], s["max_dd"], s["mean_gross"], alpha)


# ---------------------------------------------------------------- (a)
def job_zeroshot(seed, N=50):
    torch.set_num_threads(1)
    m = make_market("heston", num_assets=N)
    fdim = 3 + 1
    pol = PathwisePolicy(N, fdim, alpha=0.2, ell=0.2, tc=TC, eta=ETA, rho=RHO, r=m.r, name="E-CBLPO")
    sd = torch.load(os.path.join(RES, "checkpoints", f"heston__E-CBLPO__seed{seed}.pt"))
    pol.actor.load_state_dict(sd)          # DeepSets weights do not depend on N
    ev = pol.evaluate(m, B_EVAL, T, 20_000 + seed, track_weights_of=0)
    res = {"E-CBLPO (trained N=5, zero-shot)": met(ev["W_T"], ev["max_dd"], ev["mean_gross"], 0.2)}
    for rule in [Cash(), MertonMyopic(RHO), TIPP(3.0), TIPP(5.0), GrossmanZhou(RHO)]:
        res[rule.name] = eval_rule(rule, "heston", {"num_assets": N}, 20_000 + seed, 0.2)
    extra = {"weights_path0": ev["weights"], "wealth_path0": ev["wealth_path"][:, 0]} if seed == 0 else None
    return ("zeroshot", seed, res, extra)


# ---------------------------------------------------------------- (b)
def job_alpha_rules(alpha):
    K = (1 - ETA) / (0.2 + 2 * TC)
    rules = [GrossmanZhou(RHO), TIPP(3.0), TIPP(5.0), MertonMyopic(RHO), Cash(),
             HJBPolicy(solve_hjb(alpha=alpha, gamma=RHO), T, "HJB optimum (continuous-time)"),
             HJBPolicy(solve_hjb(alpha=alpha, gamma=RHO, K_cap=K), T, "HJB optimum (certified class)")]
    out = {}
    for k in range(3):
        for rule in rules:
            out.setdefault(rule.name, []).append(eval_rule(rule, "constant_bs", {"num_assets": 1}, 30_000 + k, alpha))
    return ("alpha_rules", alpha, out, None)


def job_alpha_learned(alpha, seed):
    torch.set_num_threads(1)
    m = make_market("constant_bs", num_assets=1)
    pol = PathwisePolicy(1, 3, alpha=alpha, ell=0.2, tc=TC, eta=ETA, rho=RHO, r=m.r, seed=seed, name="E-CBLPO")
    pol.train(m, iters=PW_ITERS, B=512, T=T, seed=1000 + seed, verbose=False)
    ev = pol.evaluate(m, B_EVAL, T, 30_000 + seed)
    return ("alpha_learned", (alpha, seed), {"E-CBLPO": met(ev["W_T"], ev["max_dd"], ev["mean_gross"], alpha)}, None)


def run(j):
    return {"zeroshot": job_zeroshot, "alpha_rules": job_alpha_rules, "alpha_learned": job_alpha_learned}[j[0]](*j[1:])


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [("alpha_learned", a, s) for a in ALPHAS for s in range(3)]
    jobs += [("zeroshot", s) for s in range(int(os.environ.get("ZS_SEEDS", 5)))] + [("alpha_rules", a) for a in ALPHAS]
    zs, al_rules, al_learned = {}, {}, {}
    with Pool(int(os.environ.get("WORKERS", 2))) as pool:
        for i, (kind, key, res, extra) in enumerate(pool.imap_unordered(run, jobs)):
            print(f"[{i+1}/{len(jobs)}] {kind} {key}: " + "; ".join(f"{k}: CE={v['CE']:.4f} viol={100*v['viol']:.2f}%" for k, v in list(res.items())[:2] if isinstance(v, dict)), flush=True)
            if kind == "zeroshot":
                zs[key] = res
                if extra is not None:
                    np.savez_compressed(os.path.join(OUT, "zeroshot_N50_path0.npz"), **extra)
            elif kind == "alpha_rules":
                al_rules[key] = res
            else:
                al_learned.setdefault(key[0], []).append(res["E-CBLPO"])
    json.dump({str(k): v for k, v in sorted(zs.items())}, open(os.path.join(OUT, "zeroshot_N50.json"), "w"), indent=2)
    json.dump({"rules": {str(a): al_rules[a] for a in sorted(al_rules)},
               "E-CBLPO": {str(a): al_learned[a] for a in sorted(al_learned)}},
              open(os.path.join(OUT, "alpha_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
