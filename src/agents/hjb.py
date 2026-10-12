"""
Numerical finite-horizon optimum for the single-asset Black-Scholes drawdown problem.

    max E[ W_T^{1-g} / (1-g) ]   s.t.  W_t >= a H_t  for all t,  a = 1 - alpha,
    dW = W [ (r + pi (mu - r)) dt + pi sigma dB ],   pi in [pi_min, pi_max].

Homogeneity: V(t, W, H) = H^{1-g} v(t, x),  x = W / H in [a, 1], and v solves

    v_t + max_pi { x (r + pi (mu - r)) v_x + 0.5 x^2 pi^2 sigma^2 v_xx } = 0,
    v(T, x) = x^{1-g} / (1-g),
    v_x(t, 1) = (1 - g) v(t, 1)            (running-maximum / reflection condition),
    pi(t, a) = 0                            (state constraint: diffusion must vanish).

With K_cap = K the control set is further restricted to the certified class of
Theorem 5.1 (revised), |pi| W <= K h, which is also what discrete-time pathwise safety
requires (Proposition 5.2).  Comparing the two solutions separates the "price of
discrete-time safety" from the optimisation gap of a learned policy.

Solved backwards with an implicit scheme and explicit (lagged) control, i.e. one
policy-evaluation step per time step (standard for this class of problems).
This gives the finite-horizon benchmark that the long-horizon Grossman-Zhou rule does
not provide.  Transaction costs are ignored in the PDE (they are charged when the policy
is executed in the discrete-time environment).
"""
import numpy as np
from scipy.linalg import solve_banded


def solve_hjb(mu=0.08, sigma=0.20, r=0.02, alpha=0.20, gamma=2.0, T=1.0, pi_min=-0.5, pi_max=2.0,
              M=401, steps_per_day=8, days=252, K_cap=None):
    a = 1.0 - alpha
    x = np.linspace(a, 1.0, M)
    dx = x[1] - x[0]
    nt = days * steps_per_day
    dt = T / nt
    v = x ** (1 - gamma) / (1 - gamma)
    pis = np.zeros((nt + 1, M))
    for n in range(nt - 1, -1, -1):
        # control from current (time n+1) value function
        vx = np.gradient(v, dx)
        vxx = np.zeros(M)
        vxx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx ** 2
        vxx[-1] = vxx[-2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pi = -(mu - r) * vx / (x * sigma ** 2 * vxx)
        pi = np.where(vxx < 0, pi, pi_max)
        pi = np.clip(np.nan_to_num(pi, nan=0.0), pi_min, pi_max)
        if K_cap is not None:
            # restrict to the certified class  |pi| W <= K h   <=>   |pi| <= K (1 - a / x)
            cap = K_cap * (1.0 - a / x)
            pi = np.clip(pi, -cap, cap)
        pi[0] = 0.0
        pis[n] = pi
        # implicit step:  (v^n - v^{n+1})/dt + L v^n = 0  ->  (I - dt L) v^n = v^{n+1}
        drift = x * (r + pi * (mu - r))
        diff = 0.5 * x ** 2 * pi ** 2 * sigma ** 2
        lo = np.zeros(M); di = np.ones(M); up = np.zeros(M)
        for i in range(1, M - 1):
            # upwind drift
            if drift[i] >= 0:
                cu, cl = drift[i] / dx, 0.0
            else:
                cu, cl = 0.0, -drift[i] / dx
            cd = diff[i] / dx ** 2
            lo[i] = -dt * (cd + cl)
            up[i] = -dt * (cd + cu)
            di[i] = 1 + dt * (2 * cd + cu + cl)
        # x = a: pi = 0, drift a r > 0 points inward -> forward difference
        c0 = drift[0] / dx
        di[0] = 1 + dt * c0; up[0] = -dt * c0
        # x = 1: Robin condition (v_M - v_{M-1})/dx = (1-g) v_M
        rhs = v.copy()
        di[-1] = 1.0 / dx - (1 - gamma); lo[-1] = -1.0 / dx; rhs[-1] = 0.0
        ab = np.zeros((3, M))
        ab[0, 1:] = up[:-1]; ab[1] = di; ab[2, :-1] = lo[1:]
        v = solve_banded((1, 1), ab, rhs)
    pis[nt] = pis[nt - 1]
    ce = ((1 - gamma) * v[-1]) ** (1 / (1 - gamma))   # x = 1 at t = 0 (W0 = H0 = 1)
    return {"x": x, "v0": v, "pi": pis, "dt": dt, "steps_per_day": steps_per_day, "ce_continuous": ce}


class HJBPolicy:
    """Executes the numerical optimal feedback pi*(t, W/H) in the discrete-time env."""

    def __init__(self, sol, horizon=252, name="HJB optimum (continuous-time)"):
        self.sol = sol
        self.T = horizon
        self.name = name

    def act(self, o):
        n = int(round(o["t_frac"][0] * self.T)) * self.sol["steps_per_day"]
        n = min(n, self.sol["pi"].shape[0] - 1)
        xr = np.clip(o["W"] / o["H"], self.sol["x"][0], 1.0)
        return np.interp(xr, self.sol["x"], self.sol["pi"][n])[:, None]
