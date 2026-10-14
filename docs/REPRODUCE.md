# Reproducing the results

All numbers in the manuscript are produced by the scripts below from saved result files;
`experiments/check_manuscript_numbers.py` recomputes every number quoted in the text from those files and
reports mismatches (currently 0). Default parameters: alpha = 20%, gamma = 2, c = 10 bp, ell = 20%
(25% in the jump market), eta = 0.05, K = (1-eta)/(ell+2c) = 4.70, one year of daily trading
(`results/config.json`). Solver settings by use are listed in Table "Solver settings by use" of the
appendix.

| Step | Script(s) | Output | Needs |
|---|---|---|---|
| Simulation benchmarks (4 markets, 5 seeds) | `run_benchmark.py` | `results/<market>/`, `results/tables/summary.json` | – |
| Robust class, zero-shot N=50, alpha sweep | `run_robust_class.py`, `run_extensions.py` | `results/extensions/` | benchmark checkpoints |
| Kim–Omberg 2-D HJB, zero-cost gap | `run_ko_hjb.py`, `run_zero_cost.py` | `results/extensions/zero_cost/` | – |
| Estimation risk, gamma/horizon sensitivity | `run_estimation_risk.py`, `run_sensitivity.py` | `results/extensions/` | – |
| Near-floor numerics (Section 3) | `run_boundary_layer.py`, `run_localisation.py`, `run_r_scaling.py`, `check_smooth_fit.py`, `check_free_boundary.py`, `check_floor_equation.py` | `results/extensions/` | – |
| Price of certification, rate table | `run_price_of_safety.py`, `run_price_rate.py` (SPD=32,64), `run_solver_checks.py` (PART=A,B), `make_rate_table.py` | `results/extensions/` | – |
| Independent dual solver | `compare_primal_dual.py` | `results/extensions/primal_dual.json` | – |
| (G*) certificate | `check_Gstar.py`, `run_gstar_value.py` | `results/extensions/gstar_*.json` | – |
| Ablations, N-scaling | `run_ablation.py` (and `ABL_MARKET=jump`), `run_nscaling.py`, `make_*_table.py` | `results/extensions/` | benchmark checkpoints |
| CRSP study E1/E2/E3 | `data/wrds_export.py`, `data/wrds_export_top50_history.py`, `run_empirical.py`, `run_empirical_zeroshot.py`, `run_empirical_tipp.py`, `make_empirical_tables.py` | `results/empirical/` | WRDS licence |
| Empirical diagnostics | `run_empirical_diagnostics.py`, `run_stress_exceedance.py`, `violation_ci.py`, block bootstrap (`EMP_BLOCK=20 RESULTS_DIR=results/block20 run_empirical.py E1`), `compare_bootstrap.py` | `results/empirical/`, `results/block20/` | WRDS licence |
| Cost-convention sensitivity | `run_cost_sensitivity.py` | `results/extensions/cost_sensitivity.json` | checkpoints, WRDS for E1 |
| Tables, figures, number check | `make_tables.py`, `make_figures.py`, `check_manuscript_numbers.py` | `results/tables/`, `results/figures/` | all of the above |

`experiments/run_all.sh` runs the steps in this order. CRSP data are licensed and are not in the repository;
the export scripts reproduce the files in `data/raw/`.
