"""
Finite-horizon drawdown HJB on a graded mesh clustered at the floor.

Same problem as hjb.py, written in the cushion coordinate y = x - a in [0, alpha]
(x = W/H, a = 1 - alpha), with mesh y_j = alpha * (j / M)^q.  q > 1 concentrates points
near the floor so that the boundary-layer behaviour of the optimal policy can be resolved
down to y ~ 1e-6.

    v_t + r x v_x + sup_{pi in Pi(x)} { pi x (mu - r) v_x + 0.5 pi^2 x^2 sigma^2 v_xx } = 0
    v(T, x) = x^{1-g}/(1-g),   v_x(t, 1) = (1-g) v(t, 1),   pi(t, a) = 0,
    Pi(x) = [pi_min, pi_max]  intersected, if K_cap is given, with |pi| <= K_cap (1 - a/x).

Implicit in v with the control lagged by one step (policy evaluation per step), upwind
drift, second-order non-uniform central differences for v_xx.
"""
import numpy as np
from scipy.linalg import solve_banded


def solve_hjb_graded(mu=0.08, sigma=0.20, r=0.02, alpha=0.20, gamma=2.0, T=1.0, pi_min=-0.5, pi_max=2.0,
                     M=800, q=2.5, steps_per_day=8, days=252, K_cap=None, keep_times=(0.0, 0.1, 0.5, 0.9),
                     howard=0):
    """howard = number of extra policy-iteration sweeps per time step (0: control lagged by one
    step; k > 0: the control is recomputed from the new solution k times, a fully implicit
    discretisation in the limit)."""
    a = 1.0 - alpha
    y = alpha * (np.arange(M + 1) / M) ** q
    x = a + y
    hm = np.r_[np.nan, np.diff(y)]          # h_minus at j  (y_j - y_{j-1})
    hp = np.r_[np.diff(y), np.nan]          # h_plus  at j  (y_{j+1} - y_j)
    nt = days * steps_per_day
    dt = T / nt
    v = x ** (1 - gamma) / (1 - gamma)
    keep = {int(round(f * nt)): f for f in keep_times}
    out_pi = {}
    out_v = {}
    J = np.arange(1, M)
    hmJ, hpJ = hm[J], hp[J]
    # first derivative (central, non-uniform) and second derivative weights
    d1 = (-hpJ / (hmJ * (hmJ + hpJ)), (hpJ - hmJ) / (hmJ * hpJ), hmJ / (hpJ * (hmJ + hpJ)))
    d2 = (2 / (hmJ * (hmJ + hpJ)), -2 / (hmJ * hpJ), 2 / (hpJ * (hmJ + hpJ)))
    cap = None if K_cap is None else K_cap * (1.0 - a / x)
    for n in range(nt - 1, -1, -1):
      v_old = v
      for sweep in range(howard + 1):
          vx = np.zeros(M + 1); vxx = np.zeros(M + 1)
          vx[J] = d1[0] * v[J - 1] + d1[1] * v[J] + d1[2] * v[J + 1]
          vxx[J] = d2[0] * v[J - 1] + d2[1] * v[J] + d2[2] * v[J + 1]
          with np.errstate(divide="ignore", invalid="ignore"):
              pi = -(mu - r) * vx / (x * sigma ** 2 * vxx)
          pi = np.where(vxx < 0, pi, pi_max)
          pi = np.clip(np.nan_to_num(pi, nan=0.0), pi_min, pi_max)
          if cap is not None:
              pi = np.clip(pi, -cap, cap)
          pi[0] = 0.0
          pi[-1] = pi[-2]
          if n in keep and sweep == howard:
              out_pi[keep[n]] = pi.copy()
              out_v[keep[n]] = v.copy()      # the value function the stored control was computed from
          drift = x * (r + pi * (mu - r))
          diff = 0.5 * x ** 2 * pi ** 2 * sigma ** 2
          lo = np.zeros(M + 1); di = np.ones(M + 1); up = np.zeros(M + 1)
          dJ, sJ = drift[J], diff[J]
          cu = np.where(dJ > 0, dJ / hpJ, 0.0); cl = np.where(dJ < 0, -dJ / hmJ, 0.0)
          lo[J] = -dt * (sJ * d2[0] + cl)
          up[J] = -dt * (sJ * d2[2] + cu)
          di[J] = 1 - dt * (sJ * d2[1] - cu - cl)
          c0 = drift[0] / hp[0]
          di[0] = 1 + dt * c0; up[0] = -dt * c0
          rhs = v_old.copy()
          di[-1] = 1.0 / hm[-1] - (1 - gamma); lo[-1] = -1.0 / hm[-1]; rhs[-1] = 0.0
          ab = np.zeros((3, M + 1))
          ab[0, 1:] = up[:-1]; ab[1] = di; ab[2, :-1] = lo[1:]
          v = solve_banded((1, 1), ab, rhs)
    ce = ((1 - gamma) * v[-1]) ** (1 / (1 - gamma))
    return {"y": y, "x": x, "v0": v, "pi": out_pi, "v": out_v, "ce": ce, "a": a}
