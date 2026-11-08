#!/usr/bin/env bash
# Official curriculum run (doc §7): four stages, one config each.
#
#   GPU=4 bash script/run_curriculum.sh
#
# For every stage it
#   1. seeds the stage directory with the previous stage's final weights,
#   2. trains to that stage's target epoch (checkpoints every 250 epochs),
#   3. copies the stage-final weights to <root>/stageN_final.pth,
#   4. runs the acceptance matrix on that checkpoint.
# No wall-clock timeout: each stage stops at its epoch target.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

ENVS=${ENVS:-2048}
ROOT=${ROOT:-output/exp/pacer_curriculum}
RELEASE=${RELEASE:-output/release/pacer/Humanoid.pth}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml}
EPOCHS=${EPOCHS:-"1500 2000 3000 3500"}

set -- $EPOCHS
START=41250
PREV="$RELEASE"
mkdir -p "$ROOT"
SUMMARY="$ROOT/curriculum_summary.txt"
: > "$SUMMARY"

i=0
for add in "$@"; do
  i=$((i + 1))
  TARGET=$((START + add))
  OUT="$ROOT/stage${i}"
  mkdir -p "$OUT"

  if [ ! -f "$OUT/Humanoid.pth" ]; then
    cp "$PREV" "$OUT/Humanoid.pth"
    echo "[curriculum] stage$i seeded from $PREV"
  fi

  echo "[curriculum] stage$i: $START -> $TARGET epochs ($add), cfg stage$i, out $OUT"
  python pacer/run.py \
    --task HumanoidPedestrianTerrain \
    --cfg_env "pacer/data/cfg/pacer_stage${i}_stairs.yaml" \
    --cfg_train "$CFG_TRAIN" \
    --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
    --network_path "$OUT" \
    --epoch -1 \
    --num_envs "$ENVS" \
    --max_iterations "$TARGET" \
    --headless --no_log \
    2>&1 | tee "/data/ljx/tmp/curriculum_stage${i}.log"

  # keep the stage's final weights separately
  cp "$OUT/Humanoid.pth" "$ROOT/stage${i}_final.pth"
  echo "[curriculum] stage$i final weights -> $ROOT/stage${i}_final.pth"

  # per-stage acceptance test
  WEIGHTS="$OUT" TAG="curriculum_stage${i}_accept" GPU="${GPU:-4}" \
    bash script/eval_stage_accept.sh > "$ROOT/stage${i}_accept.log" 2>&1 || true
  cp "experiment_result/validation/curriculum_stage${i}_accept_summary.csv" \
     "$ROOT/stage${i}_accept.csv" 2>/dev/null || true
  {
    echo "stage$i  epoch=$TARGET  weights=$ROOT/stage${i}_final.pth"
    cat "$ROOT/stage${i}_accept.csv" 2>/dev/null
    echo
  } >> "$SUMMARY"

  PREV="$ROOT/stage${i}_final.pth"
  START=$TARGET
done

echo "[curriculum] all stages done"
cat "$SUMMARY"
echo CURRICULUM_DONE
