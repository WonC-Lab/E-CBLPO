"""
Item 11: ablations in the five-asset Heston market (ABL_MARKET=jump: the one-asset jump-diffusion market) (5 seeds, 4,000 evaluation paths per seed,
same training budget and evaluation seeds as run_benchmark.py). New variants:
  "GZ + projection (no residual)"   the Grossman-Zhou direction through the safety layer, no network;
  "penalty, no projection (lam)"    the same network trained with a drawdown penalty
                                    lam * E[(maxDD - alpha)_+] in the loss and no safety layer.
The full method, "no GZ prior" and "MLP" are taken from run_benchmark.py (results/heston/raw and
results/training_histories.jsonl); here we rerun the full method too, to time all variants on the
same machine. Recorded: CE, violation rate, mean max drawdown, gross exposure, training seconds,
training-batch CE every iteration (sample efficiency), gradient norms (stability).
Output: results/extensions/ablation/<variant>__seed<k>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.sim.markets import make_market
from src.agents.pathwise import PathwisePolicy
MARKET = os.environ.get("ABL_MARKET", "heston")      # "heston" (N = 5) or "jump" (N = 1, common crashes)
OUT = os.path.join(ROOT, "results", "extensions", "ablation", *([] if MARKET == "heston" else [MARKET]))
MKW = {"heston": {"num_assets": 5}, "jump": {"num_assets": 1}}[MARKET]
ALPHA, TC, ETA, RHO, T = 0.2, 0.001, 0.05, 2.0, 252
ELL = {"heston": 0.2, "jump": 0.25}[MARKET]
ITERS, BATCH, B_EVAL = 200, 512, 4000
VARIANTS = {"full": dict(), "GZ + projection (no residual)": dict(residual=False),
            "penalty, no projection (lam=10)": dict(safety=False, penalty=10.0),
            "penalty, no projection (lam=100)": dict(safety=False, penalty=100.0)}


class CrashShift:
    """Out-of-distribution test: the training market plus a common crash of size CRASH (all assets fall
    by CRASH on the same day) with intensity 2 per year. The crash stays within the stress bound
    (CRASH < ell), so the certified policies remain covered by Proposition prop:discrete; the
    policies' inputs (coefficients, features) are those of the training market."""
    CRASH = {"heston": 0.15, "jump": 0.20}

    def __init__(self, base, crash, lam=2.0):
        self.b, self.c, self.lam = base, crash, lam
        self.N, self.r, self.dt = base.N, base.r, base.dt

    def __getattr__(self, k):
        return getattr(self.b, k)

    def reset(self, B, rng):
        return self.b.reset(B, rng)

    def step(self, state, rng):
        R, st = self.b.step(state, rng)
        hit = rng.random(R.shape[0]) < self.lam * self.dt
        R = np.where(hit[:, None], (1 + R) * (1 - self.c) - 1, R)
        return R, st


def job(args):
    v, seed = args
    f = os.path.join(OUT, f"{v.replace(' ', '_')}__seed{seed}.json")
    if os.path.exists(f) and "shift" in json.load(open(f)):
        return v, seed, "cached"
    torch.set_num_threads(1)
    m = make_market(MARKET, **MKW)
    fdim = 3 + m.features(m.reset(1, np.random.default_rng(0))).shape[-1]
    pol = PathwisePolicy(m.N, fdim, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=RHO, r=m.r, seed=seed, name=v, **VARIANTS[v])
    t0 = time.time()
    hist = pol.train(m, iters=ITERS, B=BATCH, T=T, seed=seed, verbose=False) if pol.residual else []
    secs = time.time() - t0
    ev = pol.evaluate(m, B_EVAL, T, 10_000 + seed)
    W, dd, g = ev["W_T"], ev["max_dd"], ev["mean_gross"]
    res = {"variant": v, "seed": seed, "CE": float(np.mean(W ** (1 - RHO)) ** (1 / (1 - RHO))),
           "viol": float(np.mean(dd > ALPHA + 1e-9)), "viol_count": int(np.sum(dd > ALPHA + 1e-9)),
           "mean_maxdd": float(dd.mean()), "gross": float(g.mean()), "train_seconds": secs,
           "W_T": W.tolist(), "history": hist}
    sh = pol.evaluate(CrashShift(make_market(MARKET, **MKW), CrashShift.CRASH[MARKET]), B_EVAL, T, 30_000 + seed)
    res["shift"] = {"crash": CrashShift.CRASH[MARKET], "CE": float(np.mean(sh["W_T"] ** (1 - RHO)) ** (1 / (1 - RHO))),
                    "viol": float(np.mean(sh["max_dd"] > ALPHA + 1e-9)), "mean_maxdd": float(sh["max_dd"].mean())}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f, "w"))
    torch.save(pol.actor.state_dict(), f.replace(".json", ".pt"))
    return v, seed, f"CE={res['CE']:.4f} viol={res['viol']:.4f} shift CE={res['shift']['CE']:.4f} viol={res['shift']['viol']:.4f} {secs:.0f}s"


if __name__ == "__main__":
    jobs = [(v, s) for s in range(5) for v in VARIANTS]
    with Pool(int(os.environ.get("WORKERS", 2))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)
