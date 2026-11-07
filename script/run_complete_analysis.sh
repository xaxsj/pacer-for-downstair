#!/usr/bin/env bash
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
OUT=experiment_result/validation/complete_summary.csv
rm -f "$OUT"
for tag in complete_release complete_stage1 complete_diverse; do
  for mesh in wide_stairs narrow_stairs; do
    for seed in 0 1; do
      st="experiment_result/validation/${tag}/${mesh}/seed_${seed}/states.pkl"
      [ -f "$st" ] || continue
      python script/evaluate_complete.py --states "$st" --mesh "data/mesh/${mesh}.pkl" --out "$OUT" >/dev/null 2>&1 || true
    done
  done
done
cat "$OUT"
