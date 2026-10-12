"""
Full benchmark: closed-form / rule-based strategies vs learned policies.

Every number reported in the paper is produced by this script and written to
results/<market>/... ; tables and figures are generated *only* from those files
(experiments/make_tables.py, experiments/make_figures.py).

Design
------
* Training seeds s = 0..S-1 for every learned policy (fresh simulated paths each iteration).
* Evaluation set k uses seed 10_000 + k with B_eval paths; all strategies (rules,
  HJB, learned) are evaluated on exactly the same paths (common random numbers).
  Learned policy trained with seed s is evaluated on evaluation set k = s.
* Metrics per evaluation set: certainty equivalent CE = (E[W_T^{1-g}])^{1/(1-g)},
  mean W_T, path violation rate P(max_t D_t > alpha), mean max drawdown, mean gross
  exposure (to show the strategy is not simply holding cash).

Usage:  python experiments/run_benchmark.py --seeds 5 --workers 2
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.sim.markets import make_market
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import default_rules
from src.agents.hjb import solve_hjb, HJBPolicy
from src.agents.ppo import PPOAgent, build_features
from src.agents.pathwise import PathwisePolicy

MARKETS = {
    "constant_bs": dict(kw=dict(num_assets=1), ell=0.20),
    "kim_omberg": dict(kw=dict(), ell=0.20),
    "heston": dict(kw=dict(num_assets=5), ell=0.20),
    "jump": dict(kw=dict(num_assets=1), ell=0.25),
}
ALPHA, TC, ETA, RHO, T = 0.20, 0.001, 0.05, 2.0, 252
PW_ITERS, PW_BATCH = int(os.environ.get("PW_ITERS", 200)), 512
PPO_ITERS, PPO_BATCH = int(os.environ.get("PPO_ITERS", 150)), 64
B_EVAL = int(os.environ.get("B_EVAL", 4000))
OUT = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))


def ce(W, g=RHO):
    return float(np.mean(W ** (1 - g)) ** (1 / (1 - g)))


def metrics(W_T, max_dd, mean_gross):
    return {"CE": ce(W_T), "mean_W": float(np.mean(W_T)), "viol": float(np.mean(max_dd > ALPHA + 1e-9)),
            "mean_maxdd": float(np.mean(max_dd)), "gross": float(np.mean(mean_gross))}


def save_raw(market, strat, k, W_T, max_dd, mean_gross, wealth_path=None):
    d = os.path.join(OUT, market, "raw")
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f"{strat.replace(' ', '_').replace('/', '-')}__eval{k}.npz")
    extra = {} if wealth_path is None else {"wealth_path_sample": wealth_path[:, :200]}
    np.savez_compressed(fn, W_T=W_T, max_dd=max_dd, mean_gross=mean_gross, **extra)


# ------------------------------------------------------------------ jobs
def job_rules(market_name, k):
    spec = MARKETS[market_name]
    strategies = default_rules(RHO)
    if market_name == "constant_bs":
        K = (1 - ETA) / (spec["ell"] + 2 * TC)
        strategies.append(HJBPolicy(solve_hjb(alpha=ALPHA, gamma=RHO), T, "HJB optimum (continuous-time)"))
        strategies.append(HJBPolicy(solve_hjb(alpha=ALPHA, gamma=RHO, K_cap=K), T, "HJB optimum (certified class)"))
    res = {}
    for s in strategies:
        env = DrawdownPortfolioEnv(make_market(market_name, **spec["kw"]), batch_size=B_EVAL, horizon=T, alpha=ALPHA, tc=TC)
        o = env.reset(10_000 + k)
        for _ in range(T):
            o, *_ = env.step(s.act(o))
        sm = env.summary()
        save_raw(market_name, s.name, k, sm["W_T"], sm["max_dd"], sm["mean_gross"], sm["wealth_path"] if k == 0 else None)
        res[s.name] = metrics(sm["W_T"], sm["max_dd"], sm["mean_gross"])
    return ("rules", market_name, k, res, None)


def job_pathwise(market_name, variant, seed):
    torch.set_num_threads(1)
    spec = MARKETS[market_name]
    m = make_market(market_name, **spec["kw"])
    fdim = 3 + m.features(m.reset(1, np.random.default_rng(0))).shape[-1]
    opts = {"E-CBLPO": dict(equivariant=True, gz_prior=True),
            "E-CBLPO (MLP)": dict(equivariant=False, gz_prior=True),
            "E-CBLPO (no GZ prior)": dict(equivariant=True, gz_prior=False)}[variant]
    pol = PathwisePolicy(m.N, fdim, alpha=ALPHA, ell=spec["ell"], tc=TC, eta=ETA, rho=RHO, r=m.r, seed=seed, name=variant, **opts)
    t0 = time.time()
    hist = pol.train(m, iters=PW_ITERS, B=PW_BATCH, T=T, seed=seed, verbose=False)
    ev = pol.evaluate(m, B_EVAL, T, 10_000 + seed)
    ck = os.path.join(OUT, "checkpoints"); os.makedirs(ck, exist_ok=True)
    torch.save(pol.actor.state_dict(), os.path.join(ck, f"{market_name}__{variant.replace(' ', '_')}__seed{seed}.pt"))
    save_raw(market_name, variant, seed, ev["W_T"], ev["max_dd"], ev["mean_gross"], ev["wealth_path"] if seed == 0 else None)
    return ("learned", market_name, seed, {variant: metrics(ev["W_T"], ev["max_dd"], ev["mean_gross"])},
            {"variant": variant, "seed": seed, "train_seconds": time.time() - t0, "history": hist})


def job_ppo(market_name, variant, seed):
    torch.set_num_threads(1)
    spec = MARKETS[market_name]
    mk = lambda B: DrawdownPortfolioEnv(make_market(market_name, **spec["kw"]), batch_size=B, horizon=T, alpha=ALPHA, tc=TC)
    env = mk(PPO_BATCH)
    x, g = build_features(env.reset(0))
    base = "E-CBLPO" if variant == "E-CBLPO (PPO-trained)" else variant
    ag = PPOAgent(base, env.N, x.shape[-1], g.shape[-1], alpha=ALPHA, ell=spec["ell"], tc=TC, eta=ETA, seed=seed, rho=RHO)
    t0 = time.time()
    hist = ag.train(env, iters=PPO_ITERS, seed=seed, verbose=False)
    ev = mk(B_EVAL); o = ev.reset(10_000 + seed)
    for _ in range(T):
        o, *_ = ev.step(ag.act(o))
    sm = ev.summary()
    ck = os.path.join(OUT, "checkpoints"); os.makedirs(ck, exist_ok=True)
    torch.save(ag.state_dict(), os.path.join(ck, f"{market_name}__{variant.replace(' ', '_')}__seed{seed}.pt"))
    save_raw(market_name, variant, seed, sm["W_T"], sm["max_dd"], sm["mean_gross"], sm["wealth_path"] if seed == 0 else None)
    return ("learned", market_name, seed, {variant: metrics(sm["W_T"], sm["max_dd"], sm["mean_gross"])},
            {"variant": variant, "seed": seed, "train_seconds": time.time() - t0, "history": hist})


def run_job(j):
    kind = j[0]
    try:
        if kind == "rules":
            return job_rules(*j[1:])
        if kind == "pw":
            return job_pathwise(*j[1:])
        return job_ppo(*j[1:])
    except Exception as e:  # never silently drop a job
        return ("error", j, repr(e), None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--markets", nargs="*", default=list(MARKETS))
    args = ap.parse_args()

    jobs = []
    for mkt in args.markets:
        for k in range(args.seeds):
            jobs.append(("rules", mkt, k))
    # slowest jobs first for load balancing
    for mkt in sorted(args.markets, key=lambda m: m != "heston"):
        pw_variants = ["E-CBLPO", "E-CBLPO (no GZ prior)"] + (["E-CBLPO (MLP)"] if MARKETS[mkt]["kw"].get("num_assets", 1) > 1 else [])
        for s in range(args.seeds):
            for v in pw_variants:
                jobs.append(("pw", mkt, v, s))
            for v in ["PPO", "PPO-Lagrangian", "E-CBLPO (PPO-trained)"]:
                jobs.append(("ppo", mkt, v, s))
    # priority: cheap rule evaluations, then E-CBLPO, then ablations / RL baselines
    prio = lambda j: 0 if j[0] == "rules" else (1 if (j[0] == "pw" and j[2] == "E-CBLPO") else 2)
    jobs = sorted(jobs, key=prio)
    kinds = os.environ.get("JOB_KINDS")
    if kinds:
        jobs = [j for j in jobs if j[0] in kinds.split(",")]

    # resume: skip jobs whose evaluation output already exists
    def done(j):
        if j[0] == "rules":
            return os.path.exists(os.path.join(OUT, j[1], "raw", f"Grossman-Zhou__eval{j[2]}.npz"))
        return os.path.exists(os.path.join(OUT, j[1], "raw", f"{j[2].replace(' ', '_')}__eval{j[3]}.npz"))
    n_all = len(jobs)
    jobs = [j for j in jobs if not done(j)]
    if len(jobs) < n_all:
        print(f"resuming: {n_all - len(jobs)} jobs already done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "per_eval_metrics.jsonl")
    hist_path = os.path.join(OUT, "training_histories.jsonl")
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool, open(log_path, "a") as flog, open(hist_path, "a") as fh:
        for i, out in enumerate(pool.imap_unordered(run_job, jobs)):
            kind, mkt, k, res, hist = out
            if kind == "error":
                print("ERROR", mkt, k, flush=True)
                flog.write(json.dumps({"error": str(mkt), "msg": k}) + "\n"); flog.flush()
                continue
            for name, met in res.items():
                flog.write(json.dumps({"market": mkt, "eval_set": k, "strategy": name, **met}) + "\n")
            flog.flush()
            if hist is not None:
                fh.write(json.dumps({"market": mkt, **hist}) + "\n"); fh.flush()
            print(f"[{i+1}/{len(jobs)}] {time.time()-t0:7.0f}s {mkt} {k} " +
                  "; ".join(f"{n}: CE={v['CE']:.4f} viol={100*v['viol']:.2f}%" for n, v in list(res.items())[:3]), flush=True)
    json.dump({"alpha": ALPHA, "tc": TC, "eta": ETA, "rho": RHO, "T": T, "B_eval": B_EVAL, "seeds": args.seeds,
               "pathwise": {"iters": PW_ITERS, "batch": PW_BATCH}, "ppo": {"iters": PPO_ITERS, "batch": PPO_BATCH},
               "markets": {k: {"kw": v["kw"], "ell": v["ell"]} for k, v in MARKETS.items()}},
              open(os.path.join(OUT, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
