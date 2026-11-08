#!/usr/bin/env bash
# Roll out a checkpoint on the procedural training terrain with N envs and dump
# states for script/completion_stats.py.
#   WEIGHTS=... CFG_ENV=... TAG=... NENV=64 FRAMES=300 GPU=6
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"

: "${WEIGHTS:?set WEIGHTS}"
: "${TAG:?set TAG}"
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml}
NENV=${NENV:-64}
FRAMES=${FRAMES:-300}
SEED=${SEED:-0}
OUT="experiment_result/validation/${TAG}/seed_${SEED}"
mkdir -p "$OUT"

before_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$WEIGHTS" \
  --test --num_envs "$NENV" --epoch -1 --no_virtual_display \
  --seed "$SEED" --auto_record --record_frames "$FRAMES" \
  >"$OUT/run.log" 2>&1 || true

after_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
if [ -n "$after_s" ] && [ "$after_s" != "$before_s" ]; then
  cp -f "$after_s" "$OUT/states.pkl"
fi
echo "states=$(test -f "$OUT/states.pkl" && echo yes || echo no) dir=$OUT"
echo PROBE_DONE
