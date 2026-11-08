#!/usr/bin/env bash
# Stage-1 stair evaluation matrix (the official protocol used for the 32/40/100%
# checkpoints): wide 2.8 m + narrow 1.2 m stairs, seeds 0-2, 300 frames, start
# at x=0.5 facing the stairs. Ends with a first-failure summary.
#
#   TAG=<name>            required, output goes to experiment_result/validation/<TAG>
#   WEIGHTS=<ckpt dir>    default output/exp/pacer_stage1_10k  (uses <dir>/Humanoid.pth)
#   SEEDS="0 1 2"         default "0 1 2"
#   FRAMES=300            recorded frames per run
#   SX=0.5                start x on the approach line
#   GPU=0                 CUDA_VISIBLE_DEVICES
#
# NOTE: do not change the pose/termination code between checkpoints you intend
# to compare -- this matrix only makes sense on one fixed evaluator.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

: "${TAG:?set TAG}"
WEIGHTS=${WEIGHTS:-output/exp/pacer_stage1_10k}
SEEDS=${SEEDS:-"0 1 2"}
FRAMES=${FRAMES:-300}
SX=${SX:-0.5}
SEED_SLUG=$(echo "$SEEDS" | tr ' ' '_')

for mesh in wide_stairs narrow_stairs; do
  WEIGHTS="$WEIGHTS" \
  CFG_ENV=pacer/data/cfg/pacer_stage1_stairs.yaml \
  CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  MESH="$mesh" PATH_FILE=data/paths/mixed_stairs.json TAG="$TAG" \
  START_XY="${SX},0.0" START_YAW=-1.4486 \
  SEEDS="$SEEDS" FRAMES="$FRAMES" bash script/run_validation_matrix.sh
done

OUT="experiment_result/validation/${TAG}_summary.csv"
rm -f "$OUT"
for mesh in wide_stairs narrow_stairs; do
  for seed in $SEEDS; do
    st="experiment_result/validation/${TAG}/${mesh}/seed_${seed}/states.pkl"
    [ -f "$st" ] || continue
    python script/evaluate_complete.py \
      --states "$st" --mesh "data/mesh/${mesh}.pkl" --out "$OUT" >/dev/null 2>&1 || true
  done
done
echo "=== ${TAG} (seeds ${SEED_SLUG}, ${FRAMES} frames) ==="
cat "$OUT"
echo "EVAL_MATRIX_DONE"
