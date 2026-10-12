"""
Finite-horizon drawdown-constrained optimum in the Kim-Omberg market (numerical, 2-D).

    dS/S = (r + sigma theta) dt + sigma dB^S
    d theta = kappa (bar - theta) dt + s (rho dB^S + sqrt(1-rho^2) dB^perp)

Homogeneity V(t, W, H, theta) = H^{1-g} v(t, x, theta), x = W/H in [a, 1]:

    v_t + max_pi { x (r + pi sigma theta) v_x + 0.5 x^2 pi^2 sigma^2 v_xx + x pi sigma s rho v_xtheta }
        + kappa (bar - theta) v_theta + 0.5 s^2 v_thetatheta = 0,
    pi*  = -(theta v_x + s rho v_xtheta) / (x sigma v_xx)      (incl. intertemporal hedging term)

Boundary conditions: v_x = (1-g) v at x = 1 (running maximum), pi = 0 at x = a,
second-derivative-free (linear extrapolation) in theta at the truncation edges.
Terminal v(T) = x^{1-g}/(1-g).

Scheme: lagged control; Lie splitting into an implicit x-step (tridiagonal in x for each
theta), an implicit theta-step (tridiagonal in theta for each x) and the explicit
mixed-derivative term.  Tridiagonal systems are solved column-wise with a vectorised
Thomas algorithm.  K_cap restricts to the certified class |pi| <= K (1 - a/x).
"""
import numpy as np


def _thomas(lo, di, up, rhs):
    """Solve many tridiagonal systems along axis 0 (shape (n, m)); lo[0], up[-1] unused."""
    n = di.shape[0]
    c = np.zeros_like(di); d = np.zeros_like(di)
    c[0] = up[0] / di[0]; d[0] = rhs[0] / di[0]
    for i in range(1, n):
        den = di[i] - lo[i] * c[i - 1]
        c[i] = up[i] / den if i < n - 1 else 0.0
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / den
    x = np.zeros_like(di)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def solve_hjb_ko(kappa=1.5, bar=0.3, s=0.15, vol=0.20, rho=-0.5, r=0.02, alpha=0.20, gamma=2.0, T=1.0,
                 pi_min=-0.5, pi_max=2.0, Mx=161, Mth=61, th_lo=-0.2, th_hi=0.8, steps_per_day=4, days=252,
                 K_cap=None, store_every=None):
    a = 1.0 - alpha
    x = np.linspace(a, 1.0, Mx); dx = x[1] - x[0]
    th = np.linspace(th_lo, th_hi, Mth); dth = th[1] - th[0]
    nt = days * steps_per_day; dt = T / nt
    X = x[:, None] * np.ones((1, Mth)); TH = np.ones((Mx, 1)) * th[None, :]
    v = X ** (1 - gamma) / (1 - gamma)
    store_every = store_every or steps_per_day
    pis = {}
    cap = None if K_cap is None else K_cap * (1.0 - a / X)
    drift_th = kappa * (bar - th)
    for n in range(nt - 1, -1, -1):
        vx = np.gradient(v, dx, axis=0)
        vxx = np.zeros_like(v); vxx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx ** 2; vxx[-1] = vxx[-2]
        vxt = np.gradient(vx, dth, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pi = -(TH * vx + s * rho * vxt) / (X * vol * vxx)
        pi = np.where(vxx < 0, pi, pi_max)
        pi = np.clip(np.nan_to_num(pi, nan=0.0), pi_min, pi_max)
        if cap is not None:
            pi = np.clip(pi, -cap, cap)
        pi[0] = 0.0
        if n % store_every == 0:
            pis[n] = pi.copy()
        # explicit mixed term
        rhs = v + dt * (X * pi * vol * s * rho * vxt)
        # ---- implicit x-step
        drift = X * (r + pi * vol * TH)
        diff = 0.5 * X ** 2 * pi ** 2 * vol ** 2
        cu = np.where(drift > 0, drift / dx, 0.0); cl = np.where(drift < 0, -drift / dx, 0.0)
        cd = diff / dx ** 2
        lo = -dt * (cd + cl); up = -dt * (cd + cu); di = 1 + dt * (2 * cd + cu + cl)
        # x = a: pi = 0, drift a r > 0 -> forward difference
        c0 = drift[0] / dx
        di[0] = 1 + dt * c0; up[0] = -dt * c0; lo[0] = 0.0
        # x = 1: Robin (v_M - v_{M-1})/dx = (1-g) v_M
        di[-1] = 1.0 / dx - (1 - gamma); lo[-1] = -1.0 / dx; up[-1] = 0.0
        rhs[-1] = 0.0
        v = _thomas(lo, di, up, rhs)
        # ---- implicit theta-step (transpose: theta along axis 0)
        cuT = np.where(drift_th > 0, drift_th / dth, 0.0); clT = np.where(drift_th < 0, -drift_th / dth, 0.0)
        cdT = 0.5 * s ** 2 / dth ** 2
        loT = np.broadcast_to((-dt * (cdT + clT))[:, None], (Mth, Mx)).copy()
        upT = np.broadcast_to((-dt * (cdT + cuT))[:, None], (Mth, Mx)).copy()
        diT = np.broadcast_to((1 + dt * (2 * cdT + cuT + clT))[:, None], (Mth, Mx)).copy()
        # edges: v_thth = 0  ->  v_0 - 2 v_1 + v_2 = 0 handled as linear extrapolation after solve
        diT[0] = 1.0; upT[0] = 0.0; loT[0] = 0.0
        diT[-1] = 1.0; loT[-1] = 0.0; upT[-1] = 0.0
        vT = _thomas(loT, diT, upT, v.T.copy())
        vT[0] = 2 * vT[1] - vT[2]; vT[-1] = 2 * vT[-2] - vT[-3]
        v = vT.T
    pis[0] = pis.get(0, pi)
    j0 = np.argmin(np.abs(th - bar))
    return {"x": x, "th": th, "v0": v, "pis": pis, "steps_per_day": steps_per_day, "store_every": store_every,
            "nt": nt, "v_bar": v[-1, j0]}


class HJBKOPolicy:
    """Executes pi*(t, x, theta) by bilinear interpolation in the discrete-time env."""

    def __init__(self, sol, horizon=252, name="KO HJB optimum"):
        self.sol, self.T, self.name = sol, horizon, name
        self.keys = np.array(sorted(sol["pis"]))

    def act(self, o):
        n = int(round(o["t_frac"][0] * self.T)) * self.sol["steps_per_day"]
        k = self.keys[np.argmin(np.abs(self.keys - n))]
        P = self.sol["pis"][k]
        x, th = self.sol["x"], self.sol["th"]
        xr = np.clip(o["W"] / o["H"], x[0], x[-1])
        tr = np.clip(o["mkt"][:, 0, 0], th[0], th[-1])
        i = np.clip(np.searchsorted(x, xr) - 1, 0, len(x) - 2)
        j = np.clip(np.searchsorted(th, tr) - 1, 0, len(th) - 2)
        wx = (xr - x[i]) / (x[i + 1] - x[i]); wt = (tr - th[j]) / (th[j + 1] - th[j])
        p = (P[i, j] * (1 - wx) * (1 - wt) + P[i + 1, j] * wx * (1 - wt) + P[i, j + 1] * (1 - wx) * wt + P[i + 1, j + 1] * wx * wt)
        return p[:, None]
