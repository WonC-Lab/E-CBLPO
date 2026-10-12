"""
Cushion-gap safety filter.

Cushion: h_t = W_t - (1 - alpha) H_t  (distance to the drawdown floor, in wealth units).

Certified constraint (G), see paper Section 5 / docs/theory_revision.tex:

    c ||u_{t-1}||_1 + (ell + 2c) ||u_t||_1  <=  (1 - eta) h_t / W_t

* Continuous time: (G) implies W_t ||u_t||_1 <= K h_t with K = (1-eta)/(ell+2c), hence
  the diffusion coefficient of h vanishes at the boundary and the safe set is
  forward invariant almost surely (Theorem 5.1 revised).
* Discrete time with proportional cost c: if every one-period excess return satisfies
  |R_i - r dt| <= ell, then h_{t+1} >= eta h_t + c W_t ||u_t||_1 > 0, and u = 0 remains
  feasible next period (Proposition 5.2 revised).  ell is therefore an explicit stress
  parameter: the guarantee is pathwise *conditional on* one-period moves no larger
  than ell.

The filter is a radial retraction u -> s u with the largest s in [0, 1] satisfying (G).
Because 0 satisfies (G) whenever the invariant holds and (G) is convex, s u is feasible.
It commutes with asset permutations (Theorem 5.3), is Lipschitz, and is closed form.

Note: this is not the Euclidean projection used in the original manuscript's QP; the
guarantees only need feasibility, and the closed form keeps training cheap.  If the
paper keeps the "minimal-intervention QP" framing, the exact projection onto (G)
should be reported as an ablation.
"""
import numpy as np


class CushionGapFilter:
    def __init__(self, alpha=0.20, ell=0.20, tc=0.001, eta=0.05):
        self.alpha, self.ell, self.tc, self.eta = alpha, ell, tc, eta

    @property
    def K(self):
        """Implied continuous-time cushion multiplier bound: W ||u||_1 <= K h."""
        return (1.0 - self.eta) / (self.ell + 2.0 * self.tc)

    def __call__(self, u, W, H, u_prev):
        u = np.asarray(u, dtype=np.float64)
        h = W - (1.0 - self.alpha) * H
        budget = (1.0 - self.eta) * h / W - self.tc * np.abs(u_prev).sum(axis=1)
        need = (self.ell + 2.0 * self.tc) * np.abs(u).sum(axis=1)
        s = np.where(need > 0, budget / np.maximum(need, 1e-15), 1.0)
        infeasible = budget < 0          # only possible after a move larger than ell
        s = np.clip(s, 0.0, 1.0)
        return u * s[:, None], {"scale": s, "binding": s < 1.0 - 1e-9, "infeasible": infeasible}
