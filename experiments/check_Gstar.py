"""
Item 9: randomised counterexample search for Proposition prop:Gstar ((G*) certificate), using the exact
wealth recursion eq:rec. Checks, on random instances with |X_i| <= ell (including adversarial returns
X_i = -ell sign(u_i)):
  (i)   (G*) at k  =>  h_{k+1} >= 0;
  (ii)  (G*) at k  =>  u_{k+1} = 0 satisfies (G*) at k+1;
  (iii) (G) => (G*);
  (iv)  convexity: midpoints of (G*)-feasible points are feasible;
  (v)   multi-period simulation: policies that scale random directions to the (G*) boundary never violate;
and computes the multiplier caps of (G*) and (G) (eta = 0 and eta = 0.05) with u_k = u_{k-1}.
Output: results/extensions/gstar_check.json
"""
import json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALPHA, ELL, C, ETA = 0.20, 0.20, 0.001, 0.05
a = 1 - ALPHA
rng = np.random.default_rng(0)
n1 = lambda u: np.abs(u).sum(-1)


def gstar_lhs(u, up, rdt):
    return (1 - C * n1(u - up)) * (1 + rdt - ELL * n1(u)) * (1 - C * n1(u))


def gstar_ok(u, up, hw, rdt):
    return (ELL * n1(u) < 1) & (C * n1(u) <= ALPHA) & (gstar_lhs(u, up, rdt) >= 1 - hw)


def g_ok(u, up, hw, eta=ETA):
    return C * n1(up) + (ELL + 2 * C) * n1(u) <= (1 - eta) * hw


def step(W, H, u, up, X, rdt):
    W1 = W * (1 - C * n1(u - up)) * (1 + rdt + (u * X).sum(-1))
    return W1, np.maximum(H, W1)


res = {"violations": {}, "counts": {}}
viol = dict(i=0, ii=0, iii=0, iv=0, v=0)
cnt = dict(i=0, ii=0, iii=0, iv=0, v_steps=0)
for N in (1, 3, 10):
    B = 200_000
    for _ in range(5):
        H = np.ones(B); hw = rng.uniform(1e-4, ALPHA, B)          # cushion ratio h/W, with H = 1
        W = a / (1 - hw)                                             # so that h = W - aH = hw W
        rdt = rng.choice([0.0, 0.02 / 252], B)
        up = rng.normal(size=(B, N)) * rng.uniform(0, 3, (B, 1)) / N
        u = rng.normal(size=(B, N)) * rng.uniform(0, 6, (B, 1)) / N
        ok = gstar_ok(u, up, hw, rdt)
        cnt["i"] += int(ok.sum())
        for adv in (True, False):
            X = -ELL * np.sign(u) if adv else rng.uniform(-ELL, ELL, (B, N))
            W1, H1 = step(W, H, u, up, X, rdt)
            h1 = W1 - a * H1
            viol["i"] += int((ok & (h1 < -1e-12)).sum())
            # (ii): u_{k+1} = 0 feasible at k+1
            hw1 = np.where(ok, h1 / W1, 1.0)
            ok1 = gstar_ok(np.zeros_like(u), u, hw1, rdt)
            viol["ii"] += int((ok & ~ok1).sum()); cnt["ii"] += int(ok.sum())
        okg = g_ok(u, up, hw)
        viol["iii"] += int((okg & ~ok).sum()); cnt["iii"] += int(okg.sum())
        # (iv) convexity on pairs of feasible points sharing (up, hw, rdt)
        u2 = rng.normal(size=(B, N)) * rng.uniform(0, 6, (B, 1)) / N
        ok2 = gstar_ok(u2, up, hw, rdt)
        lam = rng.uniform(0, 1, (B, 1))
        both = ok & ok2
        viol["iv"] += int((both & ~gstar_ok(lam * u + (1 - lam) * u2, up, hw, rdt)).sum()); cnt["iv"] += int(both.sum())
    # (v) multi-period: scale a random direction to the (G*) boundary by bisection, 252 steps, many paths
    P = 20_000
    W = np.ones(P); H = np.ones(P); up = np.zeros((P, N)); rdt = 0.02 / 252
    for k in range(252):
        hw = (W - a * H) / W
        d = rng.normal(size=(P, N)); d /= n1(d)[:, None]
        lo, hi = np.zeros(P), np.full(P, 1 / ELL)
        for _ in range(40):
            mid = (lo + hi) / 2
            f = gstar_ok(mid[:, None] * d, up, hw, rdt)
            lo, hi = np.where(f, mid, lo), np.where(f, hi, mid)
        u = lo[:, None] * d
        X = np.where(rng.random((P, 1)) < 0.5, -ELL * np.sign(u), rng.uniform(-ELL, ELL, (P, N)))
        W, H = step(W, H, u, up, X, rdt)
        viol["v"] += int((W < a * H - 1e-12).sum()); cnt["v_steps"] += P
        up = u
res["violations"], res["counts"] = viol, cnt
# multiplier caps with u_k = u_{k-1}, r = 0, as h/W -> 0
hw = np.array([1e-4, 1e-3, 1e-2, 0.1, 0.2])
caps = {}
for name, f in [("Gstar", lambda K, h: gstar_ok(np.array([[K * h]]), np.array([[K * h]]), np.array([h]), 0.0)[0]),
                ("G_eta0", lambda K, h: g_ok(np.array([[K * h]]), np.array([[K * h]]), np.array([h]), eta=0.0)[0]),
                ("G_eta005", lambda K, h: g_ok(np.array([[K * h]]), np.array([[K * h]]), np.array([h]), eta=ETA)[0])]:
    out = []
    for h in hw:
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid, h) else (lo, mid)
        out.append(lo)
    caps[name] = out
res["caps_vs_hw"] = {"hw": hw.tolist(), **caps}
res["caps_limit"] = {"Gstar": 1 / (ELL + C), "G_eta0": 1 / (ELL + 3 * C), "G_eta005": (1 - ETA) / (ELL + 3 * C)}
json.dump(res, open(os.path.join(ROOT, "results", "extensions", "gstar_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
