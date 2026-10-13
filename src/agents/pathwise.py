"""
Pathwise (backprop-through-wealth) training of cushion-parameterised policies.

Without price impact the asset-return paths R_{1:T} and the market state are exogenous:
they do not depend on the investor's actions.  The terminal wealth W_T(phi; R) is then a
differentiable function of the policy parameters phi (proportional costs and the radial
safety filter are Lipschitz), so

    grad_phi E[U(W_T)] = E[ grad_phi U(W_T(phi; R)) ]

is available exactly by automatic differentiation along simulated (or historical)
paths -- no likelihood-ratio policy-gradient estimator is needed.  This only requires
samples of return paths, exactly the data an RL agent would interact with.

Training uses fresh simulated paths every iteration; evaluation uses held-out seeds.
"""
import numpy as np
import torch

from .ppo import EquivariantActor, MLPActor


def simulate_exogenous(market, B, T, seed, alpha):
    """Pre-simulate all action-independent quantities for B paths."""
    rng = np.random.default_rng(seed)
    st = market.reset(B, rng)
    N = market.N
    ex_l, vol_l, md_l, mk_l, R_l = [], [], [], [], []
    for t in range(T):
        mu, sig = market.coeffs(st)
        ex = market.excess_drift(st)
        Sigma = np.einsum("bij,bkj->bik", sig, sig)
        md = np.linalg.solve(Sigma + 1e-10 * np.eye(N)[None], ex[..., None])[..., 0]
        vol = np.sqrt(np.einsum("bii->bi", Sigma))
        mk = market.features(st) * getattr(market, "feature_scale", 1.0)
        R, st = market.step(st, rng)
        ex_l.append(ex); vol_l.append(vol); md_l.append(md); mk_l.append(mk); R_l.append(R)
    f = lambda L: torch.from_numpy(np.stack(L).astype(np.float32))
    return {"ex": f(ex_l), "vol": f(vol_l), "md": f(md_l), "mkt": f(mk_l), "R": f(R_l)}


class PathwisePolicy:
    def __init__(self, num_assets, f_dim, g_dim=5, equivariant=True, gz_prior=True, safety=True,
                 alpha=0.2, ell=0.2, tc=0.001, eta=0.05, rho=2.0, r=0.02, dt=1 / 252, gross_max=2.0,
                 w_min=-0.5, w_max=2.0, lr=3e-3, seed=0, name="E-CBLPO", detach_prev=False,
                 residual=True, penalty=0.0):
        torch.manual_seed(seed)
        self.name = name
        self.actor = EquivariantActor(f_dim, g_dim) if equivariant else MLPActor(num_assets, f_dim, g_dim)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.gz_prior, self.safety = gz_prior, safety
        self.alpha, self.ell, self.tc, self.eta, self.rho = alpha, ell, tc, eta, rho
        self.rho_eff = (1 - alpha) + rho * alpha
        self.r, self.dt = r, dt
        self.gross_max, self.w_min, self.w_max = gross_max, w_min, w_max
        self.gamma = rho
        # detach_prev: the previous position enters the network and the safety budget as an
        # exogenous input (no gradient through the position recurrence u_{t-1} -> u_t).  The
        # recurrence can amplify gradients over 252 steps; used in the empirical study.
        self.detach_prev = detach_prev
        # ablations: residual=False drops the network (Grossman-Zhou direction only);
        # penalty > 0 adds penalty * E[(maxDD - alpha)_+] to the loss (used with safety=False)
        self.residual, self.penalty = residual, penalty

    # ----------------------------------------------------------------- policy map
    def weights(self, ex, vol, md, mkt, W, H, u_prev, t_frac):
        alpha = self.alpha
        if self.detach_prev:
            u_prev = u_prev.detach()
        h = W - (1 - alpha) * H
        D = 1 - W / H
        per = [ex * 10, vol * 5, md / 2]
        x = torch.stack(per, -1)
        if mkt.shape[-1] > 0:
            x = torch.cat([x, mkt], -1)
        g = torch.stack([D / alpha, (h / W) / alpha, torch.full_like(W, t_frac), torch.log(W), u_prev.abs().sum(1)], 1)
        a = self.actor(x, g) if self.residual else torch.zeros_like(md)
        if self.gz_prior:
            a = a + md / self.rho_eff
        u = a * (torch.clamp(h, min=0) / W)[:, None]
        # leverage limits (same as env.enforce_leverage, written to stay differentiable a.e.)
        u = torch.clamp(u, self.w_min, self.w_max)
        gross = u.abs().sum(1, keepdim=True)
        u = u * torch.clamp(self.gross_max / gross.clamp_min(1e-12), max=1.0)
        if self.safety:
            budget = (1 - self.eta) * h / W - self.tc * u_prev.abs().sum(1)
            need = (self.ell + 2 * self.tc) * u.abs().sum(1)
            s = torch.clamp(budget / need.clamp_min(1e-12), 0.0, 1.0)
            u = u * s[:, None]
        return u

    def rollout(self, ex_paths, track=False, track_weights_of=None):
        E = ex_paths
        T, B, N = E["R"].shape
        W = torch.ones(B); H = torch.ones(B); u_prev = torch.zeros(B, N)
        maxdd = torch.zeros(B); gross_sum = torch.zeros(B)
        Ws = [W]; Us = []
        for t in range(T):
            u = self.weights(E["ex"][t], E["vol"][t], E["md"][t], E["mkt"][t], W, H, u_prev, t / T)
            cost = self.tc * (u - u_prev).abs().sum(1)
            port = self.r * self.dt + (u * (E["R"][t] - self.r * self.dt)).sum(1)
            W = torch.clamp(W * (1 - cost) * (1 + port), min=1e-8)
            H = torch.maximum(H, W)
            maxdd = torch.maximum(maxdd, 1 - W / H)
            gross_sum = gross_sum + u.abs().sum(1).detach()
            u_prev = u
            if track:
                Ws.append(W.detach())
            if track_weights_of is not None:
                Us.append(u[track_weights_of].detach())
        out = {"W_T": W, "max_dd": maxdd, "mean_gross": gross_sum / T}
        if track:
            out["wealth_path"] = torch.stack(Ws).numpy()
        if track_weights_of is not None:
            out["weights"] = torch.stack(Us).numpy()
        return out

    def utility(self, W):
        g = self.gamma
        return torch.log(W) if g == 1.0 else (W ** (1 - g) - 1) / (1 - g)

    def train(self, market, iters=300, B=512, T=252, seed=0, log_every=50, verbose=True, guard=False,
              max_grad=1e3, ema=0.9):
        """Pathwise-gradient training.

        guard=True protects against exploding gradients through the wealth recursion
        (backpropagation through T steps): an update whose gradient norm is non-finite or
        above `max_grad` is skipped and the learning rate halved, and at the end the
        parameters with the best exponentially smoothed training CE are restored.  Used for
        the empirical study; the simulation experiments use guard=False.
        """
        rng = np.random.default_rng(seed)
        hist = []
        best, best_ce, ce_s = None, -np.inf, None
        for it in range(iters):
            E = simulate_exogenous(market, B, T, int(rng.integers(1 << 31)), self.alpha)
            out = self.rollout(E)
            loss = -self.utility(out["W_T"]).mean() * 100
            if self.penalty > 0:
                loss = loss + self.penalty * 100 * torch.relu(out["max_dd"] - self.alpha).mean()
            self.opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            skipped = guard and (not torch.isfinite(loss) or not torch.isfinite(gn) or float(gn) > max_grad)
            if skipped:
                for g in self.opt.param_groups:
                    g["lr"] *= 0.5
            else:
                self.opt.step()
            ce = float(torch.mean(out["W_T"].detach() ** (1 - self.gamma)) ** (1 / (1 - self.gamma)))
            hist.append({"iter": it, "ce_train": ce, "viol": float((out["max_dd"] > self.alpha + 1e-9).float().mean()),
                         "grad_norm": float(gn), "skipped": bool(skipped)})
            if guard and np.isfinite(ce):
                ce_s = ce if ce_s is None else ema * ce_s + (1 - ema) * ce
                if ce_s > best_ce:
                    best_ce = ce_s
                    best = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"    [{self.name}] it {it:4d}  CE(train batch)={ce:.4f}  viol={100*hist[-1]['viol']:.2f}%  gross={float(out['mean_gross'].mean()):.3f}", flush=True)
        if guard and best is not None:
            self.actor.load_state_dict(best)
        return hist

    @torch.no_grad()
    def evaluate(self, market, B, T, seed, track_weights_of=None):
        E = simulate_exogenous(market, B, T, seed, self.alpha)
        out = self.rollout(E, track=True, track_weights_of=track_weights_of)
        return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in out.items()}
