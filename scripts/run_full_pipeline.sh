#!/usr/bin/env bash
# Run full pipeline 01 → 07 + visuals.
# Usage: ./scripts/run_full_pipeline.sh
# Quick test: ./scripts/run_full_pipeline.sh --quick

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

QUICK=""
if [[ "$1" == "--quick" ]]; then
  QUICK="--max-rows 10000"
fi

echo "=== [1/8] Forechecks ==="
python scripts/01_forechecks.py

echo ""
echo "=== [2/8] Features ==="
python scripts/02_features.py

echo ""
echo "=== [3/8] Simple attribution ==="
python scripts/03_simple-attribution.py

echo ""
echo "=== [4/8] Tuning (optional) ==="
python scripts/04_tuning.py

echo ""
echo "=== [5/8] Modeling ==="
python scripts/05_modeling.py $QUICK

echo ""
echo "=== [6/8] Ranking ==="
python scripts/06_ranking.py

echo ""
echo "=== [7/8] Evaluation ==="
python scripts/07_evaluation.py

echo ""
echo "=== [8/8] Visuals ==="
python scripts/_visuals.py

echo ""
echo "Done. Results in data/results/, plots in plots/"
