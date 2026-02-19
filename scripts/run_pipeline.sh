#!/usr/bin/env bash
# Run forecheck pipeline top to bottom.
# Requires raw data in data/raw/

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Use venv python if it exists (needed for nohup/caffeinate where shell rc isn't loaded)
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi

echo "=== 01 Forechecks ==="
"$PYTHON" scripts/01_forechecks.py

echo "=== 02 Features ==="
"$PYTHON" scripts/02_features.py

echo "=== 03 Simple attribution ==="
"$PYTHON" scripts/03_simple-attribution.py

echo "=== 04 Tuning (optional) ==="
"$PYTHON" scripts/04_tuning.py || true

echo "=== 05 Modeling ==="
"$PYTHON" scripts/05_modeling.py

echo "=== 06 Ranking ==="
"$PYTHON" scripts/06_ranking.py

echo "=== Visuals ==="
"$PYTHON" scripts/_visuals.py --all

echo "=== Done ==="
