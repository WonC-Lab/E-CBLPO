"""
Closed-form and rule-based benchmark strategies (no learning).

All rules receive the same observation dict as the learned agents and return portfolio
weights u (fractions of current wealth).  Leverage limits are then enforced by the env.

Grossman-Zhou / Cvitanic-Karatzas (GZ)
--------------------------------------
For CRRA utility with relative risk aversion rho and floor a * H_t (a = 1 - alpha), the
Azema-Yor reduction (Cherny & Obloj, 2013) maps the drawdown-constrained problem to an
unconstrained one with risk aversion rho' = a + rho (1 - a).  The constrained investor
holds, in dollars, the cushion C_t = W_t - a H_t times the unconstrained Merton fraction
of an investor with risk aversion rho':

    W_t u_t = C_t * Sigma_t^{-1} (mu_t - r 1) / rho'.

This is exact for the long-horizon problem in a market with constant coefficients and
r = 0 (Grossman & Zhou, 1993; Cvitanic & Karatzas, 1995).  Elsewhere it is used as the
standard plug-in benchmark: coefficients are replaced by their current conditional values
(myopic in Kim-Omberg, ignores hedging demand) and r > 0 is handled through the excess
drift.
"""
import numpy as np


class Cash:
    name = "Cash"

    def act(self, o):
        return np.zeros_like(o["ex"])


class MertonMyopic:
    """Unconstrained Merton fraction Sigma^{-1}(mu - r)/rho (no drawdown control)."""
    name = "Merton (unconstrained)"

    def __init__(self, rho=2.0):
        self.rho = rho

    def act(self, o):
        return o["merton_dir"] / self.rho


class GrossmanZhou:
    name = "Grossman-Zhou"

    def __init__(self, rho=2.0):
        self.rho = rho

    def act(self, o):
        a = 1.0 - o["alpha"]
        rho_eff = a + self.rho * (1.0 - a)
        cushion_frac = np.maximum(o["h"], 0.0) / o["W"]
        return cushion_frac[:, None] * o["merton_dir"] / rho_eff


class TIPP:
    """Time-invariant portfolio protection: CPPI on a ratcheting floor (1-alpha) H_t.

    Exposure m * cushion, spread equally over the risky assets.  This is the textbook
    portfolio-insurance rule that targets exactly the running-maximum floor.
    """

    def __init__(self, m=3.0):
        self.m = m
        self.name = f"TIPP (m={m:g})"

    def act(self, o):
        N = o["ex"].shape[1]
        cushion_frac = np.maximum(o["h"], 0.0) / o["W"]
        return np.repeat((self.m * cushion_frac / N)[:, None], N, axis=1)


class CPPI:
    """Classic CPPI on a static floor (1-alpha) W_0 (does NOT target the running max)."""

    def __init__(self, m=3.0, W0=1.0):
        self.m, self.W0 = m, W0
        self.name = f"CPPI (m={m:g})"

    def act(self, o):
        N = o["ex"].shape[1]
        floor = (1.0 - o["alpha"]) * self.W0
        cushion_frac = np.maximum(o["W"] - floor, 0.0) / o["W"]
        return np.repeat((self.m * cushion_frac / N)[:, None], N, axis=1)


class ConstantMix:
    def __init__(self, w=0.6):
        self.w = w
        self.name = f"Constant mix ({int(w*100)}% risky)"

    def act(self, o):
        N = o["ex"].shape[1]
        return np.full_like(o["ex"], self.w / N)


def default_rules(rho=2.0):
    return [Cash(), ConstantMix(0.6), MertonMyopic(rho), CPPI(3.0), TIPP(3.0), TIPP(5.0), GrossmanZhou(rho)]
