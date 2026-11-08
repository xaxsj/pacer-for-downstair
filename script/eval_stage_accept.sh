#!/usr/bin/env bash
# Per-stage acceptance test: run the fixed eval matrix on one checkpoint and
# write a CSV (first_reset, route progress, climb) for that stage.
#
#   WEIGHTS=output/exp/pacer_curriculum/stage1 TAG=stage1_accept GPU=4
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

: "${TAG:?set TAG}"
WEIGHTS=${WEIGHTS:?set WEIGHTS}
SEED=${SEED:-0}
FRAMES=${FRAMES:-300}
MESHES=${MESHES:-"wide_stairs narrow_stairs eval_r010_t035 eval_r012_t028 eval_r018_t032"}

OUT_CSV="experiment_result/validation/${TAG}_summary.csv"
rm -f "$OUT_CSV"
for mesh in $MESHES; do
  WEIGHTS="$WEIGHTS" \
  CFG_ENV=pacer/data/cfg/pacer_stage1_stairs.yaml \
  CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  MESH="$mesh" PATH_FILE=data/paths/mixed_stairs.json TAG="$TAG" \
  START_XY=0.5,0.0 START_YAW=-1.4486 \
  SEEDS="$SEED" FRAMES="$FRAMES" bash script/run_validation_matrix.sh
  st="experiment_result/validation/${TAG}/${mesh}/seed_${SEED}/states.pkl"
  [ -f "$st" ] || continue
  python script/evaluate_complete.py --states "$st" \
    --mesh "data/mesh/${mesh}.pkl" --out "$OUT_CSV" >/dev/null 2>&1 || true
done
echo "=== ${TAG} acceptance ==="
cat "$OUT_CSV" 2>/dev/null || echo "no rows"
echo ACCEPT_DONE
