#!/usr/bin/env bash
# Stage 5: same stair difficulty as Stage 4, finish at the end of the planned
# path, style weight raised to 0.5 (task 0.5 / AMP 0.5), 1000 epochs.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

EPOCHS=${EPOCHS:-1000}
START_EPOCH=${START_EPOCH:-51250}
ENVS=${ENVS:-2048}
OUT=${OUT:-output/exp/pacer_stage5_1000ep}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage5_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage5.yaml}
SEED_W=${SEED_W:-output/exp/pacer_curriculum/stage4_final.pth}

mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then
  cp "$SEED_W" "$OUT/Humanoid.pth"
  echo "[stage5] seeded from $SEED_W"
fi

TARGET=$((START_EPOCH + EPOCHS))
echo "[stage5] envs=$ENVS epochs=$START_EPOCH -> $TARGET out=$OUT"
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$OUT" \
  --epoch -1 \
  --num_envs "$ENVS" \
  --max_iterations "$TARGET" \
  --headless --no_log \
  2>&1 | tee "/data/ljx/tmp/stage5_train.log"
echo STAGE5_DONE
