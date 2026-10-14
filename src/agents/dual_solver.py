"""
Independent dual solver for the finite-horizon drawdown problem (one asset, constant coefficients, r > 0).

Unknowns: the dual cushion Y(t, zeta) = x - a with zeta = ln v_x(t, x), and the ratchet boundary zeta_1(t) = ln v_x(t, 1).
Front-fixing variable xi = zeta - zeta_1(t) in [0, Xi]. With nu = theta^2 / 2,
    Y_t - zeta_1' Y_xi + nu Y_xixi + (nu - r) Y_xi - r Y = r a     where Y > 0,     Y >= 0   (obstacle problem),
    Y(t, 0) = 1 - a  (x = 1),     Y(t, Xi) = 0,     Y(T, xi) = (e^{-xi/gamma} - a)_+,
    zeta_1'(t) = (gamma - 1) (r + nu |Y_xi(t, 0)|),     zeta_1(T) = 0.
The last equation is the reflection condition v_x(t,1) = (1 - gamma) v(t,1) differentiated in time together with the
Bellman equation at x = 1. The formulation therefore assumes smooth fit and condition (B); it shares no
discretisation with the primal graded-mesh solver. Time: implicit Euler; obstacle: primal-dual active set.
Returns zeta_1(t), the free boundary zeta_a(t), Y on the grid at kept times, and CE = exp(zeta_1(0) / (1 - gamma)).
"""
import numpy as np
from scipy.linalg import solve_banded


def solve_dual(mu=0.08, sigma=0.20, r=0.02, alpha=0.20, gamma=2.0, T=1.0, Xi=4.0, N=4000, steps_per_day=32, days=252,
               keep_times=(0.0, 0.5)):
    a = 1.0 - alpha
    th = (mu - r) / sigma
    nu = th ** 2 / 2
    xi = np.linspace(0.0, Xi, N + 1)
    d = xi[1] - xi[0]
    nt = days * steps_per_day
    dt = T / nt
    Y = np.maximum(np.exp(-xi / gamma) - a, 0.0)
    z1 = 0.0
    keep = {int(round(f * nt)): f for f in keep_times}
    out = {}
    J = np.arange(1, N)
    for n in range(nt - 1, -1, -1):
        Yxi0 = (-3 * Y[0] + 4 * Y[1] - Y[2]) / (2 * d)
        z1p = (gamma - 1) * (r + nu * abs(Yxi0))
        b = (nu - r) - z1p                                   # drift coefficient of Y_xi
        # implicit: Y^n - dt*(nu Y_xixi + b Y_xi - r Y) = Y^{n+1} - dt*r*a   on interior, Dirichlet at ends
        lo = np.zeros(N + 1); di = np.ones(N + 1); up = np.zeros(N + 1)
        cd2 = nu / d ** 2
        # upwind first derivative
        cp = max(b, 0) / d; cm = max(-b, 0) / d
        lo[J] = -dt * (cd2 + cm); up[J] = -dt * (cd2 + cp); di[J] = 1 + dt * (2 * cd2 + cp + cm + r)
        rhs = Y - dt * r * a
        rhs[0] = 1 - a; rhs[-1] = 0.0
        active = np.zeros(N + 1, bool)
        for _ in range(30):
            L_ = lo.copy(); D_ = di.copy(); U_ = up.copy(); R_ = rhs.copy()
            D_[active] = 1.0; L_[active] = 0.0; U_[active] = 0.0; R_[active] = 0.0
            ab = np.zeros((3, N + 1)); ab[0, 1:] = U_[:-1]; ab[1] = D_; ab[2, :-1] = L_[1:]
            Ynew = solve_banded((1, 1), ab, R_)
            # residual of the inequality ra - LY >= 0 on the active set: (A Y - rhs) >= 0 means keep at 0
            res = np.zeros(N + 1)
            res[J] = (lo[J] * Ynew[J - 1] + di[J] * Ynew[J] + up[J] * Ynew[J + 1]) - rhs[J]
            new_active = np.zeros(N + 1, bool)
            new_active[J] = (Ynew[J] <= 0) & ~active[J] | (active[J] & (res[J] >= 0))
            new_active[J] |= Ynew[J] < 0
            if (new_active == active).all():
                break
            active = new_active
        Y = np.maximum(Ynew, 0.0); Y[0] = 1 - a
        z1 = z1 - dt * z1p
        if n in keep:
            out[keep[n]] = {"Y": Y.copy(), "zeta1": z1}
    za = {f: o["zeta1"] + xi[np.argmax(o["Y"] <= 0)] for f, o in out.items()}
    return {"xi": xi, "keep": out, "zeta1_0": z1, "zeta_a": za, "ce": float(np.exp(z1 / (1 - gamma))), "a": a, "theta": th}
