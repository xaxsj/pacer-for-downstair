#!/usr/bin/env bash
# Reward decomposition probe: short headless rollout that dumps per-term reward
# statistics to a CSV (see script/reward_report.py for the table/figure).
#
#   GPU=4 ENVS=256 STEPS=600 OUT=output/reward_debug.csv
#
# STEPS is in control steps (32 per PPO epoch), so MAX_IT is derived from it.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

ENVS=${ENVS:-256}
STEPS=${STEPS:-600}
OUT=${OUT:-output/reward_debug.csv}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1_smoke.yaml}
NETWORK=${NETWORK:-output/release/pacer}
MAX_IT=${MAX_IT:-$(( (STEPS + 31) / 32 + 1 ))}

mkdir -p "$(dirname "$OUT")"
echo "[reward_monitor] envs=$ENVS steps=$STEPS max_it=$MAX_IT -> $OUT"
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$NETWORK" \
  --epoch -1 \
  --num_envs "$ENVS" \
  --max_iterations "$MAX_IT" \
  --headless --no_log \
  --reward_debug --reward_debug_steps "$STEPS" --reward_debug_out "$OUT" \
  2>&1 | grep -a "reward_debug\|term\|mean\|Ep:" | tail -20
echo REWARD_MONITOR_DONE
