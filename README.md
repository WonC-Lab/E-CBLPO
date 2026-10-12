# E-CBLPO: Equivariant Certified-Cushion Policy Optimization for Drawdown-Constrained Portfolio Choice

Code for *Equivariant Control Barrier-Lyapunov Policy Optimization for Safe Dynamic Portfolio Choice*
(WonChan Cho, Department of Mathematics, Sungkyunkwan University — `chln0124@skku.edu`).

> **Revision note (v0.2, October 2026).** Version 0.1 of this repository evaluated reinforcement-learning
> agents **without training them**, and two figures (the 50-asset weight heat-map and the Pareto
> frontier) were produced from hand-written rules or typed-in numbers rather than from experiments.
> All of that code and its outputs have been removed. Every number and figure is now produced by
> `experiments/run_all.sh` from saved evaluation files, and the theory has been corrected
> (see `docs/theory_revision.tex`). A list of affected statements in the earlier manuscript is in
> `docs/manuscript_corrections.md`.

## What the method does

Problem: maximise CRRA utility of terminal wealth subject to a running-maximum drawdown limit
$W_t \ge (1-\alpha) H_t$, $H_t=\max_{s\le t}W_s$.

* **Policy.** Dollar exposure is the cushion $h_t = W_t-(1-\alpha)H_t$ times a multiplier:
  $u_t = (h_t/W_t)\,[\Sigma_t^{-1}(\mu_t-r)/\rho' + f_\phi(x_t)]$. The first term is the
  Grossman–Zhou / Cvitanić–Karatzas solution; $f_\phi$ is an $S_N$-equivariant DeepSets residual
  initialised at zero.
* **Safety.** A closed-form radial filter enforces
  $c\|u_{t-1}\|_1+(\ell+2c)\|u_t\|_1\le(1-\eta)h_t/W_t$. This gives almost-sure invariance in
  continuous time and pathwise invariance in discrete time whenever one-period moves stay
  within the stress bound $\ell$, with proportional costs included. Without such a bound,
  pathwise safety with Gaussian returns is impossible except in cash (Proposition 5.4).
* **Training.** Returns are exogenous (no price impact), so $\nabla_\phi\mathbb E[U(W_T)]$ is computed
  by back-propagating through the simulated wealth recursion. PPO training of the same policy is
  included as an ablation.

## Benchmarks

| Group | Strategies |
|---|---|
| Analytical / numerical | Grossman–Zhou (plug-in), Merton (unconstrained), finite-horizon HJB optimum (continuous-time and restricted to the certified class; constant coefficients only) |
| Portfolio insurance | CPPI (static floor, m = 3), TIPP (ratcheting floor, m = 3, 5), 60% constant mix, cash |
| Learned | PPO, PPO-Lagrangian, E-CBLPO trained with PPO, E-CBLPO without the GZ prior, E-CBLPO with an MLP instead of DeepSets |

Markets: Black–Scholes with constant coefficients (1 asset), Kim–Omberg (1 asset), multi-asset Heston
(5 assets; zero-shot transfer to 50 assets), Merton jump-diffusion with common crashes (1 asset).
Main metric: certainty-equivalent terminal wealth on common random paths, reported with path
violation rates and mean gross exposure.

## Layout

```
src/sim/markets.py            vectorised market models (B paths in parallel)
src/sim/portfolio_env.py      drawdown-constrained wealth recursion with costs
src/safety/drawdown_filter.py certified cushion-gap filter
src/agents/rules.py           GZ, Merton, CPPI, TIPP, constant mix, cash
src/agents/hjb.py             finite-horizon HJB solver (constant coefficients)
src/agents/pathwise.py        E-CBLPO, pathwise-gradient training
src/agents/ppo.py             PPO / PPO-Lagrangian / PPO-trained E-CBLPO
experiments/run_benchmark.py  main benchmark (resumable)
experiments/run_extensions.py 50-asset zero-shot transfer, alpha sweep
experiments/make_tables.py    tables from results/ only
experiments/make_figures.py   figures from results/ only
tests/test_theory.py          numerical checks of the propositions
docs/theory_revision.tex      revised Section 5 and proofs
```

## Reproduce

```bash
pip install torch numpy scipy matplotlib pytest
pytest -q tests
bash experiments/run_all.sh        # about 3 CPU-hours on 2 cores; safe to re-run after interruption
```

Outputs: `results/tables/` (LaTeX, Markdown, JSON), `results/figures/`, `results/checkpoints/`,
and per-path evaluation files in `results/<market>/raw/`.

## License

MIT.
