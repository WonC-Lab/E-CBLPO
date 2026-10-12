"""
All paper figures, generated only from files written by run_benchmark.py /
run_extensions.py (and, for Fig. 2, from saved checkpoints evaluated on a grid).

  fig1_safety_efficiency.pdf   CE vs violation rate, every strategy, every market
  fig2_policy_shape.pdf        cushion multiplier pi W / h vs W/H: HJB, certified HJB, GZ, E-CBLPO
  fig3_drawdown_paths.pdf      drawdown distribution over time (constant-coefficient market)
  fig4_zeroshot_N50.pdf        weights of the N=5-trained policy executed on N=50 assets
  fig5_alpha_sweep.pdf         CE and violation rate vs drawdown limit alpha
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
FIG = os.path.join(RES, "figures")

from src.agents.hjb import solve_hjb
from src.agents.ppo import EquivariantActor

STYLE = {
    "Cash": ("#9aa0a6", "s"), "Constant mix (60% risky)": ("#bdbdbd", "D"),
    "Merton (unconstrained)": ("#d62728", "v"), "CPPI (m=3)": ("#8c564b", "<"),
    "TIPP (m=3)": ("#e377c2", ">"), "TIPP (m=5)": ("#bcbd22", ">"),
    "Grossman-Zhou": ("#1f77b4", "o"), "HJB optimum (continuous-time)": ("#000000", "*"),
    "HJB optimum (certified class)": ("#444444", "P"),
    "PPO": ("#ff7f0e", "x"), "PPO-Lagrangian": ("#ffbb78", "x"),
    "E-CBLPO (PPO-trained)": ("#98df8a", "^"), "E-CBLPO (no GZ prior)": ("#17becf", "^"),
    "E-CBLPO (MLP)": ("#9467bd", "^"), "E-CBLPO": ("#2ca02c", "^"),
}
TITLES = {"constant_bs": "Black–Scholes (1 asset)", "kim_omberg": "Kim–Omberg (1 asset)",
          "heston": "Heston (5 assets)", "jump": "Jump-diffusion (1 asset)"}


def fig1(summary):
    markets = [m for m in TITLES if m in summary]
    fig, axes = plt.subplots(1, len(markets), figsize=(4.2 * len(markets), 4.0), squeeze=False)
    for ax, m in zip(axes[0], markets):
        for name, r in summary[m].items():
            c, mk = STYLE.get(name, ("k", "."))
            ax.errorbar(100 * r["viol_rate"], r["CE"], yerr=1.96 * r["CE_se"], fmt=mk, color=c, ms=8 if name == "E-CBLPO" else 6,
                        mec="k" if name == "E-CBLPO" else c, label=name, capsize=2)
        ax.set_title(TITLES[m]); ax.set_xlabel("Paths violating the 20% limit (%)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Certainty equivalent of $W_T$")
    h, l = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h2, l2 = ax.get_legend_handles_labels()
        for hh, ll in zip(h2, l2):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, "fig1_safety_efficiency.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig1_safety_efficiency.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2():
    """Cushion multiplier as a function of x = W/H at mid-horizon, constant coefficients."""
    alpha, a, mu, sig, r, tc, eta, ell = 0.2, 0.8, 0.08, 0.2, 0.02, 0.001, 0.05, 0.2
    K = (1 - eta) / (ell + 2 * tc)
    t_frac = 0.5
    sol = solve_hjb(alpha=alpha)
    solc = solve_hjb(alpha=alpha, K_cap=K)
    n = int(t_frac * 252) * sol["steps_per_day"]
    x = np.linspace(a + 0.002, 1.0, 200)
    mult = lambda s: np.interp(x, s["x"], s["pi"][n]) / (1 - a / x)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, mult(sol), "k-", lw=2, label="HJB optimum (continuous-time)")
    ax.plot(x, mult(solc), color="#444444", ls="--", lw=2, label="HJB optimum (certified class)")
    md = (mu - r) / sig ** 2
    ax.axhline(md / (a + 2.0 * alpha), color="#1f77b4", lw=2, label="Grossman–Zhou")
    ax.axhline(K, color="r", ls=":", lw=1.5, label=f"certification cap K = {K:.2f}")
    cks = sorted(glob.glob(os.path.join(RES, "checkpoints", "constant_bs__E-CBLPO__seed*.pt")))
    for i, ck in enumerate(cks):
        actor = EquivariantActor(3, 5); actor.load_state_dict(torch.load(ck))
        H = 1.0; W = x * H; h = W - a * H
        feats = torch.tensor(np.stack([np.full_like(x, (mu - r) * 10), np.full_like(x, sig * 5), np.full_like(x, md / 2)], -1)[:, None, :], dtype=torch.float32)
        g = torch.tensor(np.stack([(1 - x) / alpha, (h / W) / alpha, np.full_like(x, t_frac), np.log(W), np.zeros_like(x)], 1), dtype=torch.float32)
        with torch.no_grad():
            m_raw = actor(feats, g).numpy()[:, 0] + md / (a + 2.0 * alpha)
        m_exec = np.minimum(m_raw, K)     # filter with u_prev = 0 caps the multiplier at K
        ax.plot(x, m_exec, color="#2ca02c", lw=1.2, alpha=0.8, label="E-CBLPO (5 training seeds)" if i == 0 else None)
    ax.set_xlabel("$x = W_t/H_t$"); ax.set_ylabel(r"cushion multiplier $\pi_t W_t / h_t$")
    ax.set_ylim(0, 15); ax.set_xlim(a, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Optimal and learned cushion multipliers ($t=T/2$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_policy_shape.pdf")); fig.savefig(os.path.join(FIG, "fig2_policy_shape.png"), dpi=200)
    plt.close(fig)


def fig3():
    names = ["Merton (unconstrained)", "HJB optimum (continuous-time)", "Grossman-Zhou", "E-CBLPO"]
    fig, axes = plt.subplots(1, len(names), figsize=(15, 3.4), sharey=True)
    for ax, nm in zip(axes, names):
        f = os.path.join(RES, "constant_bs", "raw", nm.replace(" ", "_") + "__eval0.npz")
        if not os.path.exists(f):
            continue
        Wp = np.load(f)["wealth_path_sample"]       # (T+1, 200)
        H = np.maximum.accumulate(Wp, axis=0); D = 100 * (1 - Wp / H)
        t = np.arange(D.shape[0])
        for q, al in [(50, 0.9), (95, 0.6), (99, 0.35)]:
            ax.plot(t, np.percentile(D, q, axis=1), color=STYLE[nm][0], alpha=al, label=f"{q}th pct")
        ax.plot(t, D.max(1), color=STYLE[nm][0], ls=":", label="max of 200 paths")
        ax.axhline(20, color="r", ls="--", lw=1)
        ax.set_title(nm, fontsize=10); ax.set_xlabel("trading day"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("drawdown $D_t$ (%)"); axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_drawdown_paths.pdf")); fig.savefig(os.path.join(FIG, "fig3_drawdown_paths.png"), dpi=200)
    plt.close(fig)


def fig4():
    f = os.path.join(RES, "extensions", "zeroshot_N50_path0.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    Wt = d["weights_path0"] * 100           # (T, 50)
    Wp = d["wealth_path0"]
    D = 100 * (1 - Wp / np.maximum.accumulate(Wp))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    im = a1.imshow(Wt.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-np.abs(Wt).max(), vmax=np.abs(Wt).max())
    fig.colorbar(im, ax=[a1, a2], label="weight (%)", shrink=0.75, anchor=(0, 1))
    a1.set_ylabel("asset index (sorted by base drift)")
    a1.set_title("E-CBLPO trained on N=5, executed without retraining on N=50 (one evaluation path)")
    a2.plot(D, color="#2ca02c"); a2.axhline(20, color="r", ls="--"); a2.set_xlim(-0.5, Wt.shape[0] - 0.5); a2.set_ylabel("$D_t$ (%)"); a2.set_xlabel("trading day"); a2.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig4_zeroshot_N50.pdf")); fig.savefig(os.path.join(FIG, "fig4_zeroshot_N50.png"), dpi=200)
    plt.close(fig)


def fig5():
    f = os.path.join(RES, "extensions", "alpha_sweep.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    alphas = sorted(float(a) for a in d["rules"])
    series = {nm: [d["rules"][str(a) if str(a) in d["rules"] else repr(a)][nm] for a in alphas] for nm in next(iter(d["rules"].values()))}
    series["E-CBLPO"] = [d["E-CBLPO"][str(a)] for a in alphas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for nm, vals in series.items():
        c, mk = STYLE.get(nm, ("k", "."))
        ce = np.array([[v["CE"] for v in vs] for vs in vals]); vi = np.array([[v["viol"] for v in vs] for vs in vals]) * 100
        x = np.array(alphas) * 100
        a1.errorbar(x, ce.mean(1), yerr=ce.std(1, ddof=1) / np.sqrt(ce.shape[1]) * 1.96, fmt=mk + "-", color=c, label=nm, capsize=2)
        a2.plot(x, vi.mean(1), mk + "-", color=c, label=nm)
    a1.set_xlabel(r"drawdown limit $\alpha$ (%)"); a1.set_ylabel("certainty equivalent"); a1.grid(alpha=0.3)
    a2.set_xlabel(r"drawdown limit $\alpha$ (%)"); a2.set_ylabel("paths violating (%)"); a2.grid(alpha=0.3)
    a2.legend(fontsize=7, loc="upper right")
    a1.set_title("Constant coefficients: CE vs $\\alpha$ (3 eval sets × 4000 paths)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_alpha_sweep.pdf")); fig.savefig(os.path.join(FIG, "fig5_alpha_sweep.png"), dpi=200)
    plt.close(fig)


def fig6():
    f = os.path.join(RES, "extensions", "price_of_safety.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    fits = {}
    for (al, rows), c in zip(d.items(), ["#1f77b4", "#d62728"]):
        rows_f = [r for r in rows if r["K"] != "inf"]
        K = np.array([r["K"] for r in rows_f], float); gap = np.array([r["gap_bp"] for r in rows_f])
        viol = np.array([r["viol"] for r in rows_f]) * 100
        a1.loglog(K, gap, "o-", color=c, label=f"alpha = {float(al):.0%}")
        sel = (K >= 3) & (K <= 15)
        slope = np.polyfit(np.log(K[sel]), np.log(gap[sel]), 1)[0]; fits[al] = slope
        a2.semilogx(K, viol, "o-", color=c, label=f"alpha = {float(al):.0%}")
        inf_v = [r["viol"] for r in rows if r["K"] == "inf"][0] * 100
        a2.axhline(inf_v, color=c, ls=":", lw=1)
    for ax in (a1, a2):
        ax.axvline(1.0, color="gray", ls="--", lw=1); ax.axvline(4.7, color="gray", ls=":", lw=1); ax.grid(alpha=0.3, which="both")
    a1.set_xlabel("cushion-multiplier cap K"); a1.set_ylabel("CE loss vs. continuous-time optimum (bp)")
    a1.set_title("Price of the certification constraint\n(fitted slope, K in [3,15]: " + ", ".join(f"{float(k):.0%}: {v:.2f}" for k, v in fits.items()) + ")", fontsize=10)
    a2.set_xlabel("cushion-multiplier cap K"); a2.set_ylabel("paths violating under daily execution (%)")
    a2.set_title("Violations of the K-capped optimum (dotted: K = inf)", fontsize=10)
    a1.text(1.05, a1.get_ylim()[0] * 1.5, "robust class\n(K = 1)", fontsize=8, color="gray")
    a1.legend(); a2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_price_of_safety.pdf")); fig.savefig(os.path.join(FIG, "fig6_price_of_safety.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    summary = json.load(open(os.path.join(RES, "tables", "summary.json")))
    fig1(summary); fig2(); fig3(); fig4(); fig5(); fig6()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
