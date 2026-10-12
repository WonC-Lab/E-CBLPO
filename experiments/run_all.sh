#!/usr/bin/env bash
# Full reproduction pipeline (resumable: rerun after an interruption).
set -e
cd "$(dirname "$0")/.."
python experiments/run_benchmark.py --seeds 5 --workers 2
[ -f results/extensions/alpha_sweep.json ] || python experiments/run_extensions.py
python experiments/make_tables.py > /dev/null
python experiments/make_figures.py
echo "ALL DONE"
