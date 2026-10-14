# The Price of Discrete-Time Drawdown Safety

**Boundary behaviour of the finite-horizon optimum and learned certified policies**

Code, saved results and reproduction scripts for the manuscript of the same title
(WonChan Cho, Department of Mathematics, Sungkyunkwan University, `chln0124@skku.edu`).
Repository: <https://github.com/WonC-Lab/drawdown-safety-price> (formerly `E-CBLPO`; old links redirect). The code
keeps the earlier project name E-CBLPO for the learned method, which the paper calls ECP (equivariant certified policy).

## Problem

Maximise CRRA utility of terminal wealth, $\mathbb E[U(W_T)]$ with $U(w)=w^{1-\gamma}/(1-\gamma)$, subject to a
running-maximum drawdown limit

$$W_t \ge aH_t,\qquad H_t=\max_{s\le t}W_s,\qquad a=1-\alpha .$$

A policy is **$K$-certified** if its cushion multiplier $W_t\|u_t\|_1/(W_t-aH_t)$ never exceeds $K$. The paper asks
three questions:

1. What does the finite-horizon optimum do near the floor $W=aH$?
2. Why can it not be executed safely in discrete time, and what does discrete-time safety cost?
3. How can the certified optimum be computed when no Bellman reduction is available?

## Main results and their status

Every statement in the paper is labelled **proved**, **conditional** (proved under a stated extra hypothesis),
**numerical** or **conjecture**. Table 1 of the manuscript has the full list; the main entries are below.

### Boundary behaviour of the optimum (one risky asset, Black–Scholes, constant rate $r\ge0$)

| Result | Status |
|---|---|
| Value function is continuous and a viscosity solution of its HJB equation. No comparison principle is used, so uniqueness and convergence of numerical schemes are **not** claimed | proved |
| Reflection condition at the running maximum, $v_x(t,1-)=(1-\gamma)v(t,1)$ for every $t<T$, with no regularity assumed at $x=1$ | proved |
| Interior regularity $v\in C^{1,2}$, strict concavity; the Legendre dual cushion $Y$ solves a linear equation | proved |
| Floor marginal value: $v_x(t,a+)=\infty$ if $r=0$; if $r>0$, bounded at every date by an explicit barrier | proved |
| Floor equation $v(t,a)-v(s,a)=ra\int_t^s v_x(u,a+)\,du$ and smooth fit in time average ($r>0$) | proved |
| Strong obstacle formulation $\mathcal LY=ra\,\mathbf 1_{\{Y>0\}}$, $Y\in W^{2,1}_{p,\mathrm{loc}}$, supersolution property, optimal-stopping representation; **pointwise smooth fit at every date**, uniformly on compact time intervals ($r>0$) | proved |
| Optimal regularity $Y\in W^{2,1}_{\infty,\mathrm{loc}}$ (via Andersson–Lindgren–Shahgholian 2013, hypotheses checked); multiplier $m\le C(x-a)^{-1/2}$, an upper bound with no constant | proved |
| Unbounded optimal multiplier ($r>0$; and $r=0$ with $\gamma>1$) | proved |
| $r=0$, $\gamma>1$: near the floor the optimum is fixed-floor option-based insurance up to exponentially small terms | proved |
| Exact square-root law $m\sim\frac2\sigma\sqrt{ra/(x-a)}$ ($r>0$) | conditional on $Y_t\to0$ at the free boundary; confirmed numerically |

### Discrete-time safety and its price

| Result | Status |
|---|---|
| $K$-certified policies keep the drawdown limit almost surely in continuous time | proved |
| Exact one-step safety conditions with costs; robust safety for every scenario iff multiplier $\le1$ (known: De Franco–Nicolle–Pham 2020, Hsieh 2026) | proved / known |
| Stress-conditional guarantee: with costs, no violation while one-period moves stay below the stress bound $\ell$ | proved |
| $V_K\uparrow V$ as $K\to\infty$ | proved |
| Lower-bound deficit representation of $V-V_K$ | proved |
| Upper-bound deficit identity along the clamped optimum | conditional on $v\in C^{1,2}$ up to $x=1$ |
| Price of certification $\approx\mathrm{gap}(0,K)+c_rK^{-2}$, with $\mathrm{gap}(0,K)$ close to $K^{-3}$ | numerical (one calibration, error bars) and conjecture. Neither an upper bound $CK^{-2}$ nor a matching lower bound is proved; for $r=0$ the data do not distinguish $K^{-3}$ from $K^{-3}\ln K$ |

### Numbers (calibration $\alpha=20\%$, $\gamma=2$, one year, daily trading, cost 10 bp)

* Executed daily, the continuous-time optimum breaches the limit on **6.8%** of paths ($\alpha=20\%$) and **36%**
  ($\alpha=10\%$).
* The robust class ($K=1$) costs **123 bp** of certainty equivalent per year. $K=15$, which protects against a
  one-day loss of 6.1%, costs **0.6 bp** and has no violations in 20,000 simulated paths.
* Without costs, the learned policy is within **$1.1\pm0.6$ bp** of a numerical 2-D HJB benchmark in a Kim–Omberg market.
  The benchmark is a high-accuracy computation, grid-stable to 1 bp; convergence of the scheme is not proved.
* CRSP 1982–2025, out of sample: the plug-in continuous-time optimum violates the limit in **22%** of one-year
  episodes and in **70 of 75** crisis episodes; certified policies have no violation. Zero observed violations
  is an observation, not a guarantee. The guarantee holds while the stress bound holds.

## Method (learned certified policy)

* **Policy.** $u_t=(h_t/W_t)\,[\Sigma_t^{-1}(\mu_t-r)/\gamma'+f_\phi(x_t)]$, with cushion $h_t=W_t-aH_t$. The first
  term is the Grossman–Zhou / Cvitanić–Karatzas multiplier; $f_\phi$ is an $S_N$-equivariant DeepSets residual
  initialised at zero.
* **Safety layer.** A closed-form radial retraction onto
  $c\|u_{t-1}\|_1+(\ell+2c)\|u_t\|_1\le(1-\eta)h_t/W_t$. The same constraint gives almost-sure invariance in
  continuous time and the stress-conditional guarantee in discrete time.
* **Training.** Pathwise gradients through the simulated wealth recursion, with returns exogenous. PPO training is an ablation.

## Repository layout

```
src/sim/markets.py              market models (Black–Scholes, Kim–Omberg, Heston, Merton jumps, drift uncertainty)
src/sim/portfolio_env.py        drawdown-constrained wealth recursion with proportional costs
src/safety/drawdown_filter.py   certified cushion filter (closed-form retraction)
src/agents/hjb.py               1-D HJB solver, uniform grid (policies executed in simulations and data)
src/agents/hjb_graded.py        1-D HJB solver on a mesh graded towards the floor (near-floor analysis, large K)
src/agents/dual_solver.py       independent solver of the dual obstacle problem (cross-check of the primal solver)
src/agents/hjb_ko.py            2-D HJB solver for Kim–Omberg (with hedging demand)
src/agents/rules.py             Grossman–Zhou, Merton, CPPI, TIPP, constant mix, cash
src/agents/pathwise.py          learned certified policy, pathwise-gradient training
src/agents/ppo.py               PPO, PPO-Lagrangian, PPO-trained variant (baselines and ablation)
experiments/run_*.py            one script per experiment (resumable)
experiments/check_*.py          numerical checks of near-floor statements (floor equation, smooth fit, free boundary, (G*))
experiments/compare_primal_dual.py   primal vs dual solver comparison
experiments/make_*.py           tables and figures from saved results only
experiments/check_manuscript_numbers.py   recomputes every number quoted in the manuscript from saved results
tests/test_theory.py            numerical checks of the propositions (incl. the drift-only counterexample)
docs/REPRODUCE.md               step-by-step reproduction guide (script -> output -> data needed)
```

Saved results: `results/` (tables in `results/tables/`, figures in `results/figures/`, configuration in
`results/config.json`).

## Reproduce

```bash
pip install torch numpy scipy matplotlib pytest
pytest -q tests
bash experiments/run_all.sh                       # full pipeline, about 5-6 hours on 2 CPU cores; resumable
python experiments/check_manuscript_numbers.py    # compares every number quoted in the text with saved results
```

The near-floor checks run in seconds to minutes, for example
`python experiments/check_free_boundary.py` and `python experiments/compare_primal_dual.py`. Re-running them
reproduces the saved files exactly. `docs/REPRODUCE.md` maps every table and figure to its script and to the solver
settings that produced it.

### CRSP data (licensed)

`experiments/data/wrds_export.py` and `wrds_export_top50_history.py` extract the data on WRDS: Fama–French daily
factors, the CRSP market index, nine sector SPDR ETFs and the top-50 US stocks. Place the exports in `data/raw/wrds/`
and `data/raw/wrds_top50_history/`. Both directories are ignored by git; raw CRSP data are not redistributed. Then run
`run_empirical.py` (E1: US market 1982–2025; E2: sector ETFs 2004–2024), `run_empirical_zeroshot.py` (E3: ETF-trained
policies on 50 stocks), `run_empirical_tipp.py` and `make_empirical_tables.py`. Aggregated results are in
`results/empirical/`.

## History

* **v0.1** evaluated the reinforcement-learning agents without training them. Two of its figures were drawn from
  hand-written rules or typed-in numbers, and its drift-only barrier theorem was false; `tests/test_theory.py` checks
  a counterexample numerically. That code was removed and remains in the git history.
* **v0.2** made every number and figure come from `experiments/run_all.sh` and saved results.
* **v0.3 (current)** recentres the work on the theory of the finite-horizon optimum near the floor and on the price
  of discrete-time safety. It adds the graded and dual solvers, the near-floor checks and the manuscript-number
  check, and labels every result as proved, conditional, numerical or conjecture.

## License

MIT (see `LICENSE`).
