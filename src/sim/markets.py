"""
Vectorised market models (B paths simulated in parallel).

Every model exposes the same interface:

    state = model.reset(B, rng)
    mu, sig = model.coeffs(state)      # mu: (B, N) drift, sig: (B, N, m) diffusion
    R, state = model.step(state, rng)  # R: (B, N) simple asset returns over one period dt

`coeffs` are the conditional coefficients known to the investor at the start of the
period (i.e. before the period's shocks are drawn).  `step` draws the period's shocks,
returns realised simple returns, and evolves the state.  Returns are simulated exactly
over dt for the log-price, so R_i > -1 always.

`excess_drift(state)` returns the instantaneous expected excess return the investor
should use for plug-in rules, including the jump compensator where relevant.
"""
import numpy as np


class ConstantBS:
    """Multi-asset Black-Scholes market with constant coefficients.

    This is the setting in which the Grossman-Zhou (1993) / Cvitanic-Karatzas (1995)
    drawdown-constrained solution is available in closed form, so it is used as the
    validation environment.
    """

    name = "constant_bs"

    def __init__(self, num_assets=1, mu=0.08, vol=0.20, corr=0.3, r=0.02, dt=1.0 / 252.0):
        self.N = num_assets
        self.r = r
        self.dt = dt
        self.mu = np.full(num_assets, mu, dtype=np.float64)
        C = corr * np.ones((num_assets, num_assets)) + (1.0 - corr) * np.eye(num_assets)
        self.sig = np.diag(np.full(num_assets, vol)) @ np.linalg.cholesky(C)

    def reset(self, B, rng):
        return {"B": B}

    def coeffs(self, state):
        B = state["B"]
        return np.broadcast_to(self.mu, (B, self.N)).copy(), np.broadcast_to(self.sig, (B, self.N, self.N)).copy()

    def excess_drift(self, state):
        mu, _ = self.coeffs(state)
        return mu - self.r

    def step(self, state, rng):
        B = state["B"]
        mu, sig = self.coeffs(state)
        dZ = rng.standard_normal((B, self.N)) * np.sqrt(self.dt)
        var = np.einsum("bij,bij->bi", sig, sig)
        logret = (mu - 0.5 * var) * self.dt + np.einsum("bij,bj->bi", sig, dZ)
        return np.expm1(logret), state

    def features(self, state):
        return np.zeros((state["B"], self.N, 0))


class KimOmberg:
    """Single risky asset with mean-reverting market price of risk (Kim & Omberg, 1996).

        mu_t = r + vol * theta_t,   d theta = k (bar - theta) dt + s_theta dB^theta,
        corr(dB^S, dB^theta) = rho.

    No closed-form solution is known for this market under a running-maximum
    drawdown constraint; myopic plug-in of the Grossman-Zhou rule ignores
    intertemporal hedging demand.
    """

    name = "kim_omberg"

    def __init__(self, kappa=1.5, bar_theta=0.3, s_theta=0.15, vol=0.20, rho=-0.5, r=0.02,
                 dt=1.0 / 252.0, theta_clip=(-1.0, 1.5)):
        self.N = 1
        self.kappa, self.bar, self.s_theta = kappa, bar_theta, s_theta
        self.vol, self.rho, self.r, self.dt = vol, rho, r, dt
        self.theta_clip = theta_clip

    def reset(self, B, rng):
        # stationary distribution of the OU process
        sd = self.s_theta / np.sqrt(2.0 * self.kappa)
        th = np.clip(self.bar + sd * rng.standard_normal(B), *self.theta_clip)
        return {"B": B, "theta": th}

    def coeffs(self, state):
        B = state["B"]
        mu = (self.r + self.vol * state["theta"])[:, None]
        sig = np.full((B, 1, 1), self.vol)
        return mu, sig

    def excess_drift(self, state):
        return (self.vol * state["theta"])[:, None]

    def step(self, state, rng):
        B = state["B"]
        th = state["theta"]
        z1 = rng.standard_normal(B)
        z2 = rng.standard_normal(B)
        dBS = z1 * np.sqrt(self.dt)
        dBth = (self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2) * np.sqrt(self.dt)
        mu = self.r + self.vol * th
        logret = (mu - 0.5 * self.vol ** 2) * self.dt + self.vol * dBS
        th_new = th + self.kappa * (self.bar - th) * self.dt + self.s_theta * dBth
        th_new = np.clip(th_new, *self.theta_clip)
        return np.expm1(logret)[:, None], {"B": B, "theta": th_new}

    def features(self, state):
        return state["theta"][:, None, None]


class MultiHeston:
    """N assets, each with its own Heston variance, constant cross-asset correlation.

        d v_i = k_v (theta_v - v_i) dt + s_v sqrt(v_i) dZ_i,   corr(dW_i, dZ_i) = rho_sv
        mu_i  = base_mu + lambda_v (v_i - theta_v)   (risk premium rises with variance)

    Asset heterogeneity: base drifts are spread around base_mu so the assets are not
    exchangeable in distribution (otherwise equal weights would be trivially optimal).
    """

    name = "heston"
    feature_scale = 10.0   # variance ~0.04 -> O(1)

    def __init__(self, num_assets=5, base_mu=0.08, mu_spread=0.03, kappa_v=2.0, theta_v=0.04,
                 s_v=0.3, rho_sv=-0.7, lambda_v=0.5, corr=0.3, r=0.02, dt=1.0 / 252.0):
        self.N = num_assets
        self.base_mu = base_mu + mu_spread * np.linspace(-1.0, 1.0, num_assets) if num_assets > 1 else np.array([base_mu])
        self.kappa_v, self.theta_v, self.s_v, self.rho_sv = kappa_v, theta_v, s_v, rho_sv
        self.lambda_v = lambda_v
        self.r, self.dt = r, dt
        C = corr * np.ones((num_assets, num_assets)) + (1.0 - corr) * np.eye(num_assets)
        self.Lc = np.linalg.cholesky(C)

    def reset(self, B, rng):
        # draw initial variances from a Gamma approx. of the CIR stationary law
        shape = 2 * self.kappa_v * self.theta_v / self.s_v ** 2
        scale = self.theta_v / shape
        v = rng.gamma(shape, scale, size=(B, self.N))
        return {"B": B, "v": np.maximum(v, 1e-4)}

    def coeffs(self, state):
        v = state["v"]
        mu = self.base_mu[None, :] + self.lambda_v * (v - self.theta_v)
        sig = np.sqrt(v)[:, :, None] * self.Lc[None, :, :]
        return mu, sig

    def excess_drift(self, state):
        mu, _ = self.coeffs(state)
        return mu - self.r

    def step(self, state, rng):
        B = state["B"]
        v = state["v"]
        mu, sig = self.coeffs(state)
        eps = rng.standard_normal((B, self.N))
        dW = (eps @ self.Lc.T) * np.sqrt(self.dt)          # correlated asset shocks
        z = rng.standard_normal((B, self.N))
        dZ = (self.rho_sv * (dW / np.sqrt(self.dt)) + np.sqrt(1 - self.rho_sv ** 2) * z) * np.sqrt(self.dt)
        logret = (mu - 0.5 * v) * self.dt + np.sqrt(v) * dW
        v_new = v + self.kappa_v * (self.theta_v - v) * self.dt + self.s_v * np.sqrt(v) * dZ
        v_new = np.maximum(v_new, 1e-4)
        return np.expm1(logret), {"B": B, "v": v_new}

    def features(self, state):
        return state["v"][:, :, None]


class JumpDiffusion:
    """Merton jump-diffusion with a common (market-wide) Poisson crash.

        dS_i/S_i- = mu_i dt + sigma dW_i + (e^{J}-1) dN,  N ~ Poisson(lam), J ~ N(mJ, sJ^2)

    The original code base used lam=2, mJ=-0.15 which implies an expected excess
    return of -22%/yr (a market no rational long investor would hold); the default here
    keeps a positive compensated risk premium.
    """

    name = "jump"

    def __init__(self, num_assets=1, mu=0.10, vol=0.20, lam=0.5, mJ=-0.12, sJ=0.04, corr=0.3,
                 r=0.02, dt=1.0 / 252.0):
        self.N = num_assets
        self.mu = np.full(num_assets, mu)
        self.lam, self.mJ, self.sJ = lam, mJ, sJ
        self.r, self.dt = r, dt
        C = corr * np.ones((num_assets, num_assets)) + (1.0 - corr) * np.eye(num_assets)
        self.sig = vol * np.linalg.cholesky(C)
        self.kbar = np.exp(mJ + 0.5 * sJ ** 2) - 1.0      # E[e^J - 1]

    def reset(self, B, rng):
        return {"B": B}

    def coeffs(self, state):
        B = state["B"]
        return np.broadcast_to(self.mu, (B, self.N)).copy(), np.broadcast_to(self.sig, (B, self.N, self.N)).copy()

    def excess_drift(self, state):
        # compensated expected excess return
        B = state["B"]
        return np.broadcast_to(self.mu + self.lam * self.kbar - self.r, (B, self.N)).copy()

    def step(self, state, rng):
        B = state["B"]
        dZ = rng.standard_normal((B, self.N)) * np.sqrt(self.dt)
        var = np.sum(self.sig ** 2, axis=1)
        logret = (self.mu - 0.5 * var) * self.dt + dZ @ self.sig.T
        n = rng.poisson(self.lam * self.dt, size=B)
        J = np.where(n > 0, rng.normal(self.mJ, self.sJ, size=B) * n, 0.0)   # common crash
        logret = logret + J[:, None]
        return np.expm1(logret), state

    def features(self, state):
        return np.zeros((state["B"], self.N, 0))


def make_market(name, **kw):
    return {"constant_bs": ConstantBS, "kim_omberg": KimOmberg, "heston": MultiHeston, "jump": JumpDiffusion}[name](**kw)
