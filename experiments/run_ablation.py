"""
Item 11: ablations in the five-asset Heston market (5 seeds, 4,000 evaluation paths per seed,
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
OUT = os.path.join(ROOT, "results", "extensions", "ablation")
ALPHA, TC, ETA, RHO, T, ELL = 0.2, 0.001, 0.05, 2.0, 252, 0.2
ITERS, BATCH, B_EVAL = 200, 512, 4000
VARIANTS = {"full": dict(), "GZ + projection (no residual)": dict(residual=False),
            "penalty, no projection (lam=10)": dict(safety=False, penalty=10.0),
            "penalty, no projection (lam=100)": dict(safety=False, penalty=100.0)}


def job(args):
    v, seed = args
    f = os.path.join(OUT, f"{v.replace(' ', '_')}__seed{seed}.json")
    if os.path.exists(f):
        return v, seed, "cached"
    torch.set_num_threads(1)
    m = make_market("heston", num_assets=5)
    pol = PathwisePolicy(m.N, 4, alpha=ALPHA, ell=ELL, tc=TC, eta=ETA, rho=RHO, r=m.r, seed=seed, name=v, **VARIANTS[v])
    t0 = time.time()
    hist = pol.train(m, iters=ITERS, B=BATCH, T=T, seed=seed, verbose=False) if pol.residual else []
    secs = time.time() - t0
    ev = pol.evaluate(m, B_EVAL, T, 10_000 + seed)
    W, dd, g = ev["W_T"], ev["max_dd"], ev["mean_gross"]
    res = {"variant": v, "seed": seed, "CE": float(np.mean(W ** (1 - RHO)) ** (1 / (1 - RHO))),
           "viol": float(np.mean(dd > ALPHA + 1e-9)), "viol_count": int(np.sum(dd > ALPHA + 1e-9)),
           "mean_maxdd": float(dd.mean()), "gross": float(g.mean()), "train_seconds": secs,
           "W_T": W.tolist(), "history": hist}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f, "w"))
    return v, seed, f"CE={res['CE']:.4f} viol={res['viol']:.4f} {secs:.0f}s"


if __name__ == "__main__":
    jobs = [(v, s) for s in range(5) for v in VARIANTS]
    with Pool(int(os.environ.get("WORKERS", 2))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)
