"""
Markets built from historical daily returns (same interface as src/sim/markets.py).

HistoricalPaths   evaluation on realised paths: B one-year episodes, each a block of
                  consecutive trading days starting at a given index; the investor's
                  coefficients are fixed estimates (plug-in), the cash leg earns the
                  realised daily risk-free rate (period_rf).
BootstrapMarket   training: stationary block bootstrap (Politis-Romano) of daily excess
                  returns from an estimation window, with mean block length mean_block;
                  mean_block = 1 is the i.i.d. bootstrap used in the paper (run_empirical.py,
                  EMP_BLOCK=1); cash earns the constant rate r.

Both expose the estimated annualised excess drift and covariance through coeffs(), so that
plug-in rules (Grossman-Zhou, Merton), HJB policies and the learned policy see exactly the
same information.  No data after the start of an episode enters its estimates.
"""
import numpy as np


def estimate(excess, dt=1.0 / 252.0):
    """Annualised mean excess return and covariance from daily excess returns (n, N)."""
    ex = excess.mean(0) / dt
    cov = np.atleast_2d(np.cov(excess, rowvar=False)) / dt
    return ex, cov


class _Plugin:
    def _set(self, ex, cov, r, dt):
        self.N = len(ex)
        self.r, self.dt = float(r), dt
        self.ex = np.asarray(ex, float)
        self.mu = self.r + self.ex
        self.sig = np.linalg.cholesky(cov + 1e-12 * np.eye(self.N))

    def coeffs(self, state):
        B = state["B"]
        return np.broadcast_to(self.mu, (B, self.N)).copy(), np.broadcast_to(self.sig, (B, self.N, self.N)).copy()

    def excess_drift(self, state):
        return np.broadcast_to(self.ex, (state["B"], self.N)).copy()

    def features(self, state):
        return np.zeros((state["B"], self.N, 0))


class HistoricalPaths(_Plugin):
    name = "historical"

    def __init__(self, R, rf, starts, ex, cov, r, dt=1.0 / 252.0):
        """R: (n_days, N) simple returns, rf: (n_days,) daily simple risk-free returns,
        starts: indices of the first day of each episode."""
        self._set(ex, cov, r, dt)
        self.R, self.rf = np.asarray(R, float), np.asarray(rf, float)
        self.starts = np.asarray(starts, int)

    def reset(self, B, rng):
        assert B == len(self.starts), "batch size must equal the number of episodes"
        return {"B": B, "t": 0}

    def period_rf(self, state):
        return self.rf[self.starts + state["t"]]

    def step(self, state, rng):
        R = self.R[self.starts + state["t"]]
        return R, {"B": state["B"], "t": state["t"] + 1}


class BootstrapMarket(_Plugin):
    name = "bootstrap"

    def __init__(self, excess, ex, cov, r, mean_block=20, dt=1.0 / 252.0):
        self._set(ex, cov, r, dt)
        self.X = np.asarray(excess, float)
        self.p = 1.0 / mean_block

    def reset(self, B, rng):
        return {"B": B, "idx": rng.integers(0, len(self.X), B)}

    def step(self, state, rng):
        idx = state["idx"]
        R = self.r * self.dt + self.X[idx]
        jump = rng.random(len(idx)) < self.p
        nxt = np.where(jump, rng.integers(0, len(self.X), len(idx)), (idx + 1) % len(self.X))
        return R, {"B": state["B"], "idx": nxt}
