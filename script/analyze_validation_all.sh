#!/usr/bin/env bash
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
for tag in release stage1 diverse; do
  for mesh in wide_stairs narrow_stairs; do
    python script/analyze_validation.py       --root experiment_result/validation/$tag/$mesh       --mesh data/mesh/$mesh.pkl       --out experiment_result/validation/summary_${tag}_${mesh}.csv
  done
done
