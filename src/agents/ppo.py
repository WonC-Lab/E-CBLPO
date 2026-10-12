"""
PPO-family agents trained on the vectorised drawdown environment.

Variants (all share features, PPO hyper-parameters and training budget):

  "E-CBLPO"            S_N-equivariant DeepSets actor, cushion-parameterised action
                       u = (h/W) * a(x), cushion-gap safety filter applied before execution.
  "E-CBLPO (MLP)"      ablation: same as E-CBLPO but with a non-equivariant MLP actor.
  "PPO"                MLP actor outputs weights directly, no safety mechanism.
  "PPO-Lagrangian"     PPO with a learned Lagrange multiplier on the drawdown excess cost.
  "E-CBLPO (no GZ prior)"  ablation: residual starts from zero instead of the GZ multiplier.

GZ prior: for cushion-parameterised variants the actor mean is
    a_i(x) = [Sigma^{-1}(mu - r)]_i / rho'  +  f_i(x),     rho' = (1 - alpha) + rho * alpha,
i.e. the network learns a residual correction to the Grossman-Zhou cushion multiplier
(f is initialised to zero, so training starts exactly at the GZ policy).  The prior is
an equivariant function of the per-asset features, so equivariance is preserved.

The safety filter is treated as part of the environment for the policy gradient: the
log-probability is that of the pre-filter action, which is a valid (if not minimal
variance) estimator.  No differentiation through the filter is required.
"""
import numpy as np
import torch
import torch.nn as nn

from ..safety.drawdown_filter import CushionGapFilter

torch.set_num_threads(1)


# ---------------------------------------------------------------- features
def build_features(o):
    """Per-asset features (B,N,F) and global features (B,G), all permutation-consistent."""
    alpha = o["alpha"]
    per = [o["ex"] * 10.0, o["vol"] * 5.0, o["merton_dir"] / 2.0]
    mk = o["mkt"]
    if mk.shape[-1] > 0:
        per.append(mk * o["mkt_scale"])
    per = np.concatenate([p[..., None] if p.ndim == 2 else p for p in per], axis=-1)
    g = np.stack([
        o["D"] / alpha,
        (o["h"] / o["W"]) / alpha,
        o["t_frac"],
        np.log(o["W"]),
        np.abs(o["u_prev"]).sum(axis=1),
    ], axis=1)
    return per.astype(np.float32), g.astype(np.float32)


# ---------------------------------------------------------------- networks
class DeepSetsLayer(nn.Module):
    def __init__(self, din, dout, dg):
        super().__init__()
        self.self_ = nn.Linear(din, dout)
        self.pool = nn.Linear(din, dout, bias=False)
        self.glob = nn.Linear(dg, dout, bias=False)

    def forward(self, x, g):
        return self.self_(x) + self.pool(x.mean(1, keepdim=True)) + self.glob(g).unsqueeze(1)


class EquivariantActor(nn.Module):
    def __init__(self, f, g, hidden=64):
        super().__init__()
        self.l1 = DeepSetsLayer(f, hidden, g)
        self.l2 = DeepSetsLayer(hidden, hidden, g)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, g):
        h = torch.tanh(self.l1(x, g))
        h = torch.tanh(self.l2(h, g))
        return self.out(h).squeeze(-1)


class InvariantCritic(nn.Module):
    def __init__(self, f, g, hidden=64, heads=1):
        super().__init__()
        self.l1 = DeepSetsLayer(f, hidden, g)
        self.head = nn.Sequential(nn.Linear(hidden + g, hidden), nn.Tanh(), nn.Linear(hidden, heads))

    def forward(self, x, g):
        h = torch.tanh(self.l1(x, g)).mean(1)
        return self.head(torch.cat([h, g], -1))


class MLPActor(nn.Module):
    def __init__(self, n, f, g, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n * f + g, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, g):
        return self.net(torch.cat([x.flatten(1), g], -1))


class MLPCritic(nn.Module):
    def __init__(self, n, f, g, hidden=64, heads=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n * f + g, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, heads))

    def forward(self, x, g):
        return self.net(torch.cat([x.flatten(1), g], -1))


# ---------------------------------------------------------------- agent
VARIANTS = {
    "E-CBLPO":                dict(equivariant=True,  cushion=True,  safety=True,  lagrangian=False, gz_prior=True),
    "E-CBLPO (MLP)":          dict(equivariant=False, cushion=True,  safety=True,  lagrangian=False, gz_prior=True),
    "E-CBLPO (no GZ prior)":  dict(equivariant=True,  cushion=True,  safety=True,  lagrangian=False, gz_prior=False),
    "PPO":                    dict(equivariant=False, cushion=False, safety=False, lagrangian=False, gz_prior=False),
    "PPO-Lagrangian":         dict(equivariant=False, cushion=False, safety=False, lagrangian=True,  gz_prior=False),
}


class PPOAgent:
    def __init__(self, variant, num_assets, f_dim, g_dim, alpha=0.2, ell=0.2, tc=0.001, eta=0.05,
                 lr=1e-3, init_std=None, seed=0, rho=2.0):
        cfg = VARIANTS[variant]
        self.name = variant
        self.cfg = cfg
        torch.manual_seed(seed)
        n = num_assets
        if cfg["equivariant"]:
            self.actor = EquivariantActor(f_dim, g_dim)
            self.critic = InvariantCritic(f_dim, g_dim, heads=2)
        else:
            self.actor = MLPActor(n, f_dim, g_dim)
            self.critic = MLPCritic(n, f_dim, g_dim, heads=2)
        # cushion multiplier is O(1-5); raw weights are O(0.1-1)
        s0 = init_std if init_std is not None else (0.3 if cfg["cushion"] else 0.15)
        self.log_std = nn.Parameter(torch.tensor(np.log(s0), dtype=torch.float32))   # shared scalar => equivariant
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr)
        self.filter = CushionGapFilter(alpha=alpha, ell=ell, tc=tc, eta=eta) if cfg["safety"] else None
        self.log_lambda = 0.0
        self.lambda_lr = 0.05
        self.rho_eff = (1.0 - alpha) + rho * alpha

    def prior(self, o):
        if not self.cfg["gz_prior"]:
            return np.zeros_like(o["merton_dir"], dtype=np.float32)
        return (o["merton_dir"] / self.rho_eff).astype(np.float32)

    def mean_action(self, xt, gt, prior_t):
        return self.actor(xt, gt) + prior_t

    @property
    def lam(self):
        return float(np.exp(self.log_lambda)) if self.cfg["lagrangian"] else 0.0

    # raw policy output -> executed weights
    def to_weights(self, a, o):
        u = a * (np.maximum(o["h"], 0.0) / o["W"])[:, None] if self.cfg["cushion"] else a
        info = None
        if self.filter is not None:
            u, info = self.filter(u, o["W"], o["H"], o["u_prev"])
        return u, info

    def act(self, o, deterministic=True):
        x, g = build_features(o)
        with torch.no_grad():
            mean = self.mean_action(torch.from_numpy(x), torch.from_numpy(g), torch.from_numpy(self.prior(o))).numpy()
        if deterministic:
            a = mean
        else:
            a = mean + np.exp(self.log_std.item()) * np.random.standard_normal(mean.shape)
        return self.to_weights(a.astype(np.float64), o)[0]

    # ------------------------------------------------------------ training
    def train(self, env, iters=200, epochs=4, minibatch=4096, gamma=1.0, lam_gae=0.95, clip=0.2,
              reward_scale=100.0, seed=0, log_every=25, verbose=True):
        rng = np.random.default_rng(seed)
        history = []
        for it in range(iters):
            o = env.reset(int(rng.integers(1 << 31)))
            X, G, P, A, LP, V, R, C = [], [], [], [], [], [], [], []
            for t in range(env.T):
                x, g = build_features(o)
                pr = self.prior(o)
                xt, gt = torch.from_numpy(x), torch.from_numpy(g)
                with torch.no_grad():
                    mean = self.mean_action(xt, gt, torch.from_numpy(pr))
                    std = torch.exp(self.log_std)
                    a = mean + std * torch.randn_like(mean)
                    lp = torch.distributions.Normal(mean, std).log_prob(a).sum(-1)
                    v = self.critic(xt, gt)
                u, _ = self.to_weights(a.numpy().astype(np.float64), o)
                o, rew, cost, done, _ = env.step(u)
                X.append(x); G.append(g); P.append(pr); A.append(a.numpy()); LP.append(lp.numpy()); V.append(v.numpy())
                R.append(rew * reward_scale); C.append(cost * reward_scale)
            T, B = env.T, env.B
            V = np.stack(V)                       # (T,B,2)
            R = np.stack(R); C = np.stack(C)      # (T,B)
            adv = np.zeros((T, B, 2)); ret = np.zeros((T, B, 2))
            gae = np.zeros((B, 2)); nxt = np.zeros((B, 2))
            rc = np.stack([R, C], -1)
            for t in reversed(range(T)):
                delta = rc[t] + gamma * nxt - V[t]
                gae = delta + gamma * lam_gae * gae
                adv[t] = gae; ret[t] = gae + V[t]; nxt = V[t]
            X = torch.from_numpy(np.concatenate(X)); G = torch.from_numpy(np.concatenate(G)); P = torch.from_numpy(np.concatenate(P))
            A = torch.from_numpy(np.concatenate(A)); LP = torch.from_numpy(np.concatenate(LP))
            adv = adv.reshape(T * B, 2); ret = torch.from_numpy(ret.reshape(T * B, 2)).float()
            a_r = (adv[:, 0] - adv[:, 0].mean()) / (adv[:, 0].std() + 1e-8)
            if self.cfg["lagrangian"]:
                a_c = (adv[:, 1] - adv[:, 1].mean()) / (adv[:, 1].std() + 1e-8)
                a_tot = (a_r - self.lam * a_c) / (1.0 + self.lam)
            else:
                a_tot = a_r
            a_tot = torch.from_numpy(a_tot).float()
            n = X.shape[0]
            for _ in range(epochs):
                perm = torch.randperm(n)
                for s in range(0, n, minibatch):
                    idx = perm[s:s + minibatch]
                    mean = self.mean_action(X[idx], G[idx], P[idx])
                    dist = torch.distributions.Normal(mean, torch.exp(self.log_std))
                    lp = dist.log_prob(A[idx]).sum(-1)
                    ratio = torch.exp(lp - LP[idx])
                    pl = -torch.min(ratio * a_tot[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * a_tot[idx]).mean()
                    vl = ((self.critic(X[idx], G[idx]) - ret[idx]) ** 2).mean()
                    loss = pl + 0.5 * vl
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], 0.5)
                    self.opt.step()
            ep_cost = C.sum(0).mean() / reward_scale
            if self.cfg["lagrangian"]:
                # dual ascent on log-lambda: grow while episodes still incur drawdown excess
                self.log_lambda = float(np.clip(self.log_lambda + self.lambda_lr * (1.0 if ep_cost > 1e-6 else -1.0), -5, 5))
            s = env.summary()
            rec = dict(iter=it, mean_W=float(s["W_T"].mean()), viol=float(s["violated"].mean()),
                       ep_cost=float(ep_cost), std=float(np.exp(self.log_std.item())), lam=self.lam)
            history.append(rec)
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"    [{self.name}] it {it:4d}  E[W_T]={rec['mean_W']:.4f}  viol={100*rec['viol']:.1f}%  std={rec['std']:.3f}  lam={rec['lam']:.3f}", flush=True)
        return history

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "log_std": self.log_std.detach().clone(), "log_lambda": self.log_lambda, "variant": self.name}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])
        with torch.no_grad():
            self.log_std.copy_(sd["log_std"])
        self.log_lambda = sd["log_lambda"]
