# Certified Cushion Policies for Drawdown-Constrained Portfolio Choice

Code and results for the revised version of *Equivariant Control Barrier-Lyapunov Policy Optimization
for Safe Dynamic Portfolio Choice* (WonChan Cho, Department of Mathematics, Sungkyunkwan University,
`chln0124@skku.edu`).

> **Revision note (v0.2, October 2026).** Version 0.1 of this repository evaluated the
> reinforcement-learning agents **without training them**, and two figures (the 50-asset weight
> heat-map and the Pareto frontier) were drawn from hand-written rules or typed-in numbers rather than
> from experiments. That code and its outputs have been removed; they remain visible in the git
> history. In v0.2 every number and figure is produced by `experiments/run_all.sh` from saved
> evaluation files, and the theory has been corrected: the drift-only barrier condition of the
> original Theorem 5.1 does not imply almost-sure invariance (a counterexample is checked numerically
> in `tests/test_theory.py`).

## Problem and main findings

Maximise CRRA utility of terminal wealth subject to a running-maximum drawdown limit
$W_t \ge (1-\alpha)H_t$, $H_t=\max_{s\le t}W_s$, with daily trading and proportional costs.

* **The finite-horizon optimum cannot be executed safely in discrete time.** Near the floor its cushion
  multiplier diverges, and under daily rebalancing it breaches the limit on 7–36% of paths.
* **Price of discrete-time safety.** Capping the cushion multiplier at $K$ costs, numerically, an
  interest-rate component close to $c_rK^{-2}$ plus a zero-interest component close to $K^{-3}$ (a conjecture;
  see the manuscript for what is proved). The robust class ($K=1$; De Franco, Nicolle & Pham 2020; Hsieh 2026) costs
  about 123 bp per year in our calibration, while $K=15$ costs 0.6 bp and produces no violations.
* **Learning the certified optimum where no Bellman reduction exists.** An $S_N$-equivariant residual on
  the Grossman–Zhou cushion multiplier, trained by pathwise gradients through the wealth recursion,
  is within 1 bp of the exact (2-D HJB) certified optimum in a Kim–Omberg market, beats
  portfolio-insurance rules by 31–90 bp per year in four markets with no violations, and transfers
  from 5 to 50 assets without retraining.

Tables: `results/tables/tables.md` · Figures: `results/figures/` · Configuration: `results/config.json`.

## Method

* **Policy.** $u_t = (h_t/W_t)\,[\Sigma_t^{-1}(\mu_t-r)/\rho' + f_\phi(x_t)]$ with cushion
  $h_t = W_t-(1-\alpha)H_t$: the Grossman–Zhou / Cvitanić–Karatzas multiplier plus an
  $S_N$-equivariant DeepSets residual initialised at zero.
* **Safety.** A closed-form radial filter enforces
  $c\|u_{t-1}\|_1+(\ell+2c)\|u_t\|_1\le(1-\eta)h_t/W_t$. This gives almost-sure invariance in
  continuous time and pathwise invariance in discrete time whenever one-period moves stay below the
  stress bound $\ell$, with costs included.
* **Training.** Returns are exogenous (no price impact), so $\nabla_\phi\mathbb E[U(W_T)]$ is obtained by
  back-propagating through simulated wealth paths. PPO training of the same policy is an ablation.

## Benchmarks and experiments

| Group | Contents |
|---|---|
| Exact / numerical optima | finite-horizon HJB, unrestricted, restricted to multiplier cap $K$, and the robust class ($K=1$): 1-D (Black–Scholes), 2-D (Kim–Omberg, incl. hedging demand) |
| Analytical rules | Grossman–Zhou (plug-in), Merton (unconstrained) |
| Portfolio insurance | CPPI (static floor), TIPP (ratcheting floor, m = 3, 5), 60% constant mix, cash |
| Learned | E-CBLPO; ablations: MLP instead of DeepSets, no GZ prior, PPO-trained; PPO and PPO-Lagrangian baselines |
| Markets | Black–Scholes (1 asset), Kim–Omberg (1), Heston (5; zero-shot to 50), Merton jump-diffusion (1) |
| Extra studies | price of safety vs $K$ and its rate for large $K$ ($r>0$ vs $r=0$); boundary layer of the optimum near the floor; zero-cost optimality gap; drift-estimation risk; $\alpha$ sweep; risk aversion and horizon |

In the paper and figures the learned method is called ECP (equivariant certified policy); the code keeps the name E-CBLPO.

Main metric: certainty-equivalent terminal wealth on common random paths, with path-violation
rates (Clopper–Pearson bounds) and gross exposure, to show that strategies are not simply holding cash.

## Empirical study (CRSP, licensed)

`experiments/data/wrds_export.py` and `wrds_export_top50_history.py` extract the data on WRDS
(Fama-French daily factors, CRSP market index, nine sector SPDR ETFs, top-50 US stocks). Place the
unzipped exports in `data/raw/wrds/` and `data/raw/wrds_top50_history/` (ignored by git), then run
`run_empirical.py` (E1 US market 1982-2025, E2 sector ETFs 2004-2024), `run_empirical_zeroshot.py`
(E3: ETF-trained policies on 50 stocks), `run_empirical_tipp.py` (TIPP with the multiplier chosen
ex ante) and `make_empirical_tables.py`. Aggregated results are in `results/empirical/`; raw CRSP
data are not redistributed.

## Layout

```
src/sim/markets.py                 vectorised market models (incl. drift-uncertainty market)
src/sim/portfolio_env.py           drawdown-constrained wealth recursion with costs
src/safety/drawdown_filter.py      certified cushion-gap filter
src/agents/rules.py                GZ, Merton, CPPI, TIPP, constant mix, cash
src/agents/hjb.py, hjb_ko.py       finite-horizon HJB solvers (1-D, 2-D Kim-Omberg)
src/agents/pathwise.py             E-CBLPO with pathwise-gradient training
src/agents/ppo.py                  PPO / PPO-Lagrangian / PPO-trained E-CBLPO
experiments/run_*.py               one script per experiment (all resumable)
experiments/make_tables.py         tables from results/ only
experiments/make_figures.py        figures from results/ only
tests/test_theory.py               numerical checks of the propositions
```

## Reproduce

```bash
pip install torch numpy scipy matplotlib pytest
pytest -q tests
bash experiments/run_all.sh        # about 5-6 hours on 2 CPU cores; re-run safely after interruption
```

## License

MIT (see `LICENSE`).
