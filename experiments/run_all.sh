#!/usr/bin/env bash
# Full reproduction pipeline. Every step is resumable: re-run after an interruption and
# completed jobs are skipped. Roughly 5-6 hours on 2 CPU cores.
set -e
cd "$(dirname "$0")/.."

# Main benchmark: 4 markets x 5 training seeds, rules + HJB + learned policies
python experiments/run_benchmark.py --seeds 5 --workers 2

# 50-asset zero-shot transfer and drawdown-limit sweep
python experiments/run_extensions.py

# Kim-Omberg 2-D HJB optimum on the benchmark's evaluation sets
[ -f "results/kim_omberg/raw/HJB_optimum_(certified_class)__eval4.npz" ] || python experiments/run_ko_hjb.py

# Optimum of the robustly safe class (multiplier at most one), Black-Scholes and Kim-Omberg
[ -f "results/kim_omberg/raw/HJB_optimum_(robust_class,_K=1)__eval4.npz" ] || python experiments/run_robust_class.py

# Price of discrete-time safety (certified HJB over multiplier caps K)
[ -f results/extensions/price_of_safety.json ] || python experiments/run_price_of_safety.py

# Boundary behaviour of the finite-horizon optimum, and the rate of the price for large K
[ -f results/extensions/boundary_layer.json ] || python experiments/run_boundary_layer.py
for spd in 32 64; do for c in "r=0.02" "r=0.005" "r=0"; do SPD=$spd CASE=$c python experiments/run_price_rate.py; done; done

# Empirical study on CRSP data (needs data/raw/wrds from experiments/data/wrds_export*.py; licensed)
if [ -d data/raw/wrds ]; then
  python experiments/run_empirical.py E1 E2
  [ -d data/raw/wrds_top50_history ] && python experiments/run_empirical_zeroshot.py
  python experiments/run_empirical_tipp.py E1 E2
  python experiments/make_empirical_tables.py
fi

# Optimality gap without transaction costs (Black-Scholes and Kim-Omberg)
python experiments/run_zero_cost.py

# Drift-estimation risk
python experiments/run_estimation_risk.py

# Sensitivity to risk aversion and horizon
python experiments/run_sensitivity.py

python experiments/make_tables.py > /dev/null
python experiments/make_figures.py
echo "ALL DONE"
