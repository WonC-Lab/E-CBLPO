"""
Numerical checks of the statements in docs/theory_revision.tex.

Run:  pytest -q tests/test_theory.py -s
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sim.markets import make_market, ConstantBS
from src.sim.portfolio_env import DrawdownPortfolioEnv
from src.safety.drawdown_filter import CushionGapFilter
from src.agents.ppo import EquivariantActor
from src.agents.rules import GrossmanZhou


def test_original_condition_counterexample():
    """Prop. 5.1: control u = 1{W <= w_bar} satisfies condition (10) yet violates often."""
    mu, sig, r, kappa, u0, alpha = 0.08, 0.20, 0.02, 0.5, 1.0, 0.20
    w_bar = 2 * (r + u0 * (mu - r)) / (kappa * u0 ** 2 * sig ** 2)
    env = DrawdownPortfolioEnv(ConstantBS(1, mu=mu, vol=sig, r=r), batch_size=20000, tc=0.0, alpha=alpha)
    o = env.reset(7)
    cond_ok = np.ones(env.B, dtype=bool)
    for _ in range(252):
        u = np.where(o["W"] <= w_bar, u0, 0.0)[:, None]
        W, h = o["W"], o["h"]
        lhs = W * (r + u[:, 0] * (mu - r)) - 0.5 * kappa * W ** 2 * u[:, 0] ** 2 * sig ** 2
        in_safe = h >= 0
        cond_ok &= ~in_safe | (lhs >= -2.0 * h - 1e-12)
        o, *_ = env.step(u)
    viol = env.summary()["violated"].mean()
    print(f"\n[counterexample] w_bar={w_bar:.1f}  condition (10) held on safe set: {cond_ok.mean():.4f}  violation rate={viol:.4f}")
    assert cond_ok.all()
    assert viol > 0.05


def test_filter_pathwise_invariance_under_stress_bound():
    """Prop. 5.5: with (G) enforced and |R - r dt| <= ell, no path ever violates."""
    rng = np.random.default_rng(0)
    alpha, ell, tc, eta, r, dt = 0.2, 0.2, 0.001, 0.05, 0.02, 1 / 252
    f = CushionGapFilter(alpha=alpha, ell=ell, tc=tc, eta=eta)
    B, N, T = 20000, 3, 252
    W = np.ones(B); H = np.ones(B); u_prev = np.zeros((B, N)); worst = np.inf
    for _ in range(T):
        u_raw = rng.normal(0, 5.0, (B, N))                       # adversarially large requests
        u, info = f(u_raw, W, H, u_prev)
        assert not info["infeasible"].any()
        # adversarial returns: half the paths get the worst vertex -ell*sign(u), rest uniform
        ex = rng.uniform(-ell, ell, (B, N))
        worst_mask = rng.random(B) < 0.5
        ex[worst_mask] = -ell * np.sign(u[worst_mask])
        R = ex + r * dt
        W_post = W * (1 - tc * np.abs(u - u_prev).sum(1))
        W = W_post * (1 + r * dt + (u * (R - r * dt)).sum(1))
        H = np.maximum(H, W)
        h = W - (1 - alpha) * H
        worst = min(worst, h.min())
        u_prev = u
    print(f"\n[discrete invariance] min cushion over {B} adversarial paths: {worst:.3e}")
    assert worst > 0


def test_equivariance_of_executed_policy():
    """Thm. 5.7: GZ prior + DeepSets residual + radial filter commute with permutations."""
    torch.manual_seed(0)
    B, N, F, G = 8, 6, 3, 5
    actor = EquivariantActor(F, G)
    for p in actor.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(B, N, F); g = torch.randn(B, G)
    md = torch.randn(B, N)
    perm = torch.randperm(N)
    f = CushionGapFilter()
    W = np.ones(B); H = np.full(B, 1.05); u_prev = np.random.default_rng(1).normal(0, 0.05, (B, N))

    def policy(x_, md_, up_):
        a = (actor(x_, g) + md_ / 1.2).detach().numpy()
        u = a * ((W - 0.8 * H) / W)[:, None]
        return f(u, W, H, up_)[0]

    u = policy(x, md, u_prev)
    u_p = policy(x[:, perm], md[:, perm], u_prev[:, perm.numpy()])
    err = np.abs(u[:, perm.numpy()] - u_p).max()
    print(f"\n[equivariance] max |P pi(x) - pi(P x)| = {err:.2e}")
    assert err < 1e-6


def test_gz_is_certified():
    """Prop. 5.6: W ||u_GZ||_1 = K_GZ h exactly."""
    env = DrawdownPortfolioEnv(make_market("heston", num_assets=5), batch_size=500)
    o = env.reset(3)
    gz = GrossmanZhou(2.0)
    for _ in range(50):
        u = gz.act(o)
        K_t = np.abs(o["merton_dir"]).sum(1) / 1.2
        lhs = o["W"] * np.abs(u).sum(1)
        assert np.allclose(lhs, K_t * np.maximum(o["h"], 0), rtol=1e-10, atol=1e-12)
        o, *_ = env.step(u)


def test_common_random_numbers():
    """Env-based and pathwise evaluations see identical return paths."""
    from src.agents.pathwise import simulate_exogenous
    m = make_market("kim_omberg")
    env = DrawdownPortfolioEnv(m, batch_size=50)
    env.reset(11)
    E = simulate_exogenous(make_market("kim_omberg"), 50, 5, 11, 0.2)
    for t in range(5):
        st = env.state
        R, env.state = env.m.step(st, env.rng)
        assert np.allclose(R, E["R"][t].numpy(), atol=1e-6)
