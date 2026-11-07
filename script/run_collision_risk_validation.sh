#!/usr/bin/env bash
set -euo pipefail

source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

OUT_DIR="${1:-experiment_result/collision_risk_validation}"
python script/validate_collision_risk.py \
  --root experiment_result/validation \
  --out-dir "$OUT_DIR"

echo "collision-risk validation: $OUT_DIR/summary.csv"
