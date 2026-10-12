"""
Vectorised drawdown-constrained portfolio environment.

Wealth recursion over one period (weights u_t chosen at the start of the period):

    W'_t    = W_t - c * W_t * ||u_t - u_{t-1}||_1                 (proportional costs)
    W_{t+1} = W'_t * (1 + r dt + u_t^T (R_{t+1} - r dt))
    H_{t+1} = max(H_t, W_{t+1}),   D_{t+1} = 1 - W_{t+1} / H_{t+1}

A path violates the constraint if D_t > alpha at any t.  The same leverage limits
(per-asset box and gross exposure cap) are imposed on every strategy, so rules,
RL baselines and E-CBLPO trade under identical constraints.
"""
import numpy as np


class DrawdownPortfolioEnv:
    def __init__(self, market, batch_size=64, horizon=252, alpha=0.20, tc=0.001,
                 w_min=-0.5, w_max=2.0, gross_max=2.0, gamma=2.0):
        self.m = market
        self.N = market.N
        self.B = batch_size
        self.T = horizon
        self.alpha = alpha
        self.tc = tc
        self.w_min, self.w_max, self.gross_max = w_min, w_max, gross_max
        self.gamma = gamma
        self.r = market.r
        self.dt = market.dt

    # ------------------------------------------------------------------ utils
    def utility(self, W):
        if self.gamma == 1.0:
            return np.log(W)
        return (W ** (1.0 - self.gamma) - 1.0) / (1.0 - self.gamma)

    def enforce_leverage(self, u):
        u = np.clip(u, self.w_min, self.w_max)
        gross = np.abs(u).sum(axis=1, keepdims=True)
        scale = np.minimum(1.0, self.gross_max / np.maximum(gross, 1e-12))
        return u * scale

    # ------------------------------------------------------------------ core
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state = self.m.reset(self.B, self.rng)
        self.W = np.ones(self.B)
        self.H = np.ones(self.B)
        self.u_prev = np.zeros((self.B, self.N))
        self.t = 0
        self.max_dd = np.zeros(self.B)
        self.violated = np.zeros(self.B, dtype=bool)
        self.wealth_path = [self.W.copy()]
        self.exposure_path = []
        return self.observe()

    def observe(self):
        mu, sig = self.m.coeffs(self.state)
        ex = self.m.excess_drift(self.state)
        Sigma = np.einsum("bij,bkj->bik", sig, sig)
        merton_dir = np.linalg.solve(Sigma + 1e-10 * np.eye(self.N)[None], ex[..., None])[..., 0]
        vol = np.sqrt(np.einsum("bii->bi", Sigma))
        h = self.W - (1.0 - self.alpha) * self.H
        D = 1.0 - self.W / self.H
        return {
            "W": self.W.copy(), "H": self.H.copy(), "h": h, "D": D,
            "t_frac": np.full(self.B, self.t / self.T),
            "ex": ex, "Sigma": Sigma, "vol": vol, "merton_dir": merton_dir,
            "mkt": self.m.features(self.state), "mkt_scale": getattr(self.m, "feature_scale", 1.0), "u_prev": self.u_prev.copy(),
            "alpha": self.alpha, "r": self.r,
            "oracle_ex": self.m.oracle_excess(self.state) if hasattr(self.m, "oracle_excess") else ex,
        }

    def step(self, u):
        u = self.enforce_leverage(np.asarray(u, dtype=np.float64))
        cost = self.tc * np.abs(u - self.u_prev).sum(axis=1)
        W_post = self.W * (1.0 - cost)
        R, self.state = self.m.step(self.state, self.rng)
        port = self.r * self.dt + np.einsum("bi,bi->b", u, R - self.r * self.dt)
        W_new = np.maximum(W_post * (1.0 + port), 1e-12)

        reward = self.utility(W_new) - self.utility(self.W)   # telescopes to U(W_T) - U(W_0)
        self.W = W_new
        self.H = np.maximum(self.H, self.W)
        D = 1.0 - self.W / self.H
        self.max_dd = np.maximum(self.max_dd, D)
        newly = D > self.alpha + 1e-12
        self.violated |= newly
        self.u_prev = u
        self.t += 1
        self.wealth_path.append(self.W.copy())
        self.exposure_path.append(np.abs(u).sum(axis=1))
        done = self.t >= self.T
        cost_signal = np.maximum(D - self.alpha, 0.0)
        return self.observe(), reward, cost_signal, done, u

    def summary(self):
        return {
            "W_T": self.W.copy(),
            "max_dd": self.max_dd.copy(),
            "violated": self.violated.copy(),
            "mean_gross": np.mean(np.stack(self.exposure_path, 0), axis=0),
            "wealth_path": np.stack(self.wealth_path, 0),      # (T+1, B)
        }
