#!/usr/bin/env bash
# Stage 1 probe run: short training from the official checkpoint so the reward
# breakdown and a rollout video can be measured on a policy that has actually
# seen the new reward, before committing to the full 1500-epoch stage.
#
#   GPU=4 EPOCHS=200 OUT=output/exp/pacer_stage1_200ep bash script/train_stage1_200.sh
#
# No wall-clock timeout: the run stops at START_EPOCH + EPOCHS.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

EPOCHS=${EPOCHS:-200}
START_EPOCH=${START_EPOCH:-41250}
ENVS=${ENVS:-2048}
OUT=${OUT:-output/exp/pacer_stage1_200ep}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml}
RELEASE=${RELEASE:-output/release/pacer/Humanoid.pth}

mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then
  cp "$RELEASE" "$OUT/Humanoid.pth"
  echo "[stage1_200] seeded $OUT/Humanoid.pth from $RELEASE"
fi

TARGET=$((START_EPOCH + EPOCHS))
echo "[stage1_200] envs=$ENVS epochs=$START_EPOCH -> $TARGET out=$OUT"
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
  2>&1 | tee "/data/ljx/tmp/${OUT##*/}.log"
echo STAGE1_200_DONE
