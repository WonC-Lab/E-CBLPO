"""
Item 12: zero-shot scaling of the equivariant policy trained on N = 5 Heston assets
(results/checkpoints/heston__E-CBLPO__seed<k>.pt, 5 seeds), executed without retraining for
N in {5, 10, 20, 50, 100} (1,000 paths per seed; 250 for N = 100), with
  * CE, violations and gross exposure of the policy, TIPP (m=5) and Grossman-Zhou;
  * inference cost: wall-clock seconds per decision (one step for one path) and the peak
    resident memory of the evaluating process (each job runs in a fresh process);
  * robustness at N = 20: true cross-asset correlation 0, 0.3 (training value) or 0.6, and a
    covariance input error: the policy and Grossman-Zhou see the correlation matrix estimated
    from 252 daily observations (one draw per seed) instead of the true one.
Output: results/extensions/nscaling/<case>__seed<k>.json
"""
import json, os, resource, sys, time
from multiprocessing import Pool
import numpy as np
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.sim.markets import MultiHeston
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.agents.rules import GrossmanZhou, TIPP
from src.agents.pathwise import PathwisePolicy
OUT = os.path.join(ROOT, "results", "extensions", "nscaling")
TC, ETA, RHO, T, ALPHA = 0.001, 0.05, 2.0, 252, 0.2
B_ALL, B_N100 = int(os.environ.get("B_NS", 1000)), 250   # N = 100: 250 paths per seed (compute)


class Misspecified(MultiHeston):
    """Returns follow the true correlation; coeffs() reports a perturbed correlation."""
    def __init__(self, Lc_hat=None, **kw):
        super().__init__(**kw)
        self.Lc_true, self.Lc_hat = self.Lc, (self.Lc if Lc_hat is None else Lc_hat)

    def coeffs(self, state):
        v = state["v"]
        mu = self.base_mu[None, :] + self.lambda_v * (v - self.theta_v)
        return mu, np.sqrt(v)[:, :, None] * self.Lc_hat[None, :, :]


class FastEnv(DrawdownPortfolioEnv):
    """Same observation as DrawdownPortfolioEnv, with Sigma = sig sig^T computed by matmul instead of
    np.einsum (identical up to rounding; unoptimised einsum makes N = 100 impractical)."""
    def observe(self):
        mu, sig = self.m.coeffs(self.state)
        ex = self.m.excess_drift(self.state)
        Sigma = sig @ np.swapaxes(sig, 1, 2)
        md = np.linalg.solve(Sigma + 1e-10 * np.eye(self.N)[None], ex[..., None])[..., 0]
        vol = np.sqrt(np.diagonal(Sigma, axis1=1, axis2=2))
        return {"W": self.W.copy(), "H": self.H.copy(), "h": self.W - (1.0 - self.alpha) * self.H,
                "D": 1.0 - self.W / self.H, "t_frac": np.full(self.B, self.t / self.T), "ex": ex, "Sigma": Sigma,
                "vol": vol, "merton_dir": md, "mkt": self.m.features(self.state),
                "mkt_scale": getattr(self.m, "feature_scale", 1.0), "u_prev": self.u_prev.copy(),
                "alpha": self.alpha, "r": self.r, "oracle_ex": ex}


def simulate_fast(market, B, T, seed):
    """simulate_exogenous of src/agents/pathwise.py with matmul in place of einsum (same random draws)."""
    rng = np.random.default_rng(seed)
    st = market.reset(B, rng)
    L = {k: [] for k in ("ex", "vol", "md", "mkt", "R")}
    for t in range(T):
        mu, sig = market.coeffs(st)
        ex = market.excess_drift(st)
        Sigma = sig @ np.swapaxes(sig, 1, 2)
        L["md"].append(np.linalg.solve(Sigma + 1e-10 * np.eye(market.N)[None], ex[..., None])[..., 0])
        L["vol"].append(np.sqrt(np.diagonal(Sigma, axis1=1, axis2=2))); L["ex"].append(ex)
        L["mkt"].append(market.features(st) * getattr(market, "feature_scale", 1.0))
        R, st = market.step(st, rng); L["R"].append(R)
    return {k: torch.from_numpy(np.stack(v).astype(np.float32)) for k, v in L.items()}


def met(W, dd, g):
    return {"CE": float(np.mean(W ** (1 - RHO)) ** (1 / (1 - RHO))), "viol": float(np.mean(dd > ALPHA + 1e-9)),
            "gross": float(np.mean(g))}


def job(args):
    case, N, corr, cov_err, seed = args
    B = B_N100 if N >= 100 else B_ALL
    f = os.path.join(OUT, f"{case}__seed{seed}.json")
    if os.path.exists(f):
        return case, seed, "cached"
    torch.set_num_threads(1)
    Lc_hat = None
    if cov_err:
        C = corr * np.ones((N, N)) + (1 - corr) * np.eye(N)
        Z = np.random.default_rng(500 + seed).standard_normal((252, N)) @ np.linalg.cholesky(C).T
        Lc_hat = np.linalg.cholesky(np.corrcoef(Z, rowvar=False) + 1e-10 * np.eye(N))
    mk = lambda: Misspecified(Lc_hat=Lc_hat, num_assets=N, corr=corr)
    m = mk()
    pol = PathwisePolicy(N, 4, alpha=ALPHA, ell=0.2, tc=TC, eta=ETA, rho=RHO, r=m.r, name="ECP")
    pol.actor.load_state_dict(torch.load(os.path.join(ROOT, "results", "checkpoints", f"heston__E-CBLPO__seed{seed}.pt")))
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    E = simulate_fast(m, B, T, 20_000 + seed)
    t0 = time.time()
    with torch.no_grad():
        out = pol.rollout(E)
    sec = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res = {"case": case, "N": N, "corr": corr, "cov_err": cov_err, "seed": seed,
           "ECP": met(out["W_T"].numpy(), out["max_dd"].numpy(), out["mean_gross"].numpy()),
           "ECP_sec_per_decision": sec / (B * T), "peak_rss_MB": rss1 / 1024, "rss_before_rollout_MB": rss0 / 1024}
    for rule in (TIPP(5.0), GrossmanZhou(RHO)):
        env = FastEnv(mk(), batch_size=B, horizon=T, alpha=ALPHA, tc=TC)
        o = env.reset(20_000 + seed); t0 = time.time()
        for _ in range(T):
            o, *_ = env.step(rule.act(o))
        s = env.summary()
        res[rule.name] = met(s["W_T"], s["max_dd"], s["mean_gross"])
        res[rule.name + "_sec_per_decision"] = (time.time() - t0) / (B * T)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f, "w"), indent=1)
    return case, seed, f"ECP CE={res['ECP']['CE']:.4f} viol={res['ECP']['viol']:.4f} {res['ECP_sec_per_decision']*1e6:.1f}us"


if __name__ == "__main__":
    cases = [(f"N{N}", N, 0.3, False) for N in (5, 10, 20, 50, 100)]
    cases += [("N20_corr0", 20, 0.0, False), ("N20_corr0.6", 20, 0.6, False), ("N20_coverr", 20, 0.3, True)]
    jobs = [c + (s,) for c in cases for s in range(5)]
    with Pool(int(os.environ.get("WORKERS", 1)), maxtasksperchild=1) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)
