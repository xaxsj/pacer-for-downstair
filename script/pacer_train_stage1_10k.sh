#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-51250}"

OUT="${OUT:-output/exp/pacer_safe_initial_10k}"
mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then
  cp output/release/pacer/Humanoid.pth "$OUT/Humanoid.pth"
fi

# Start from the official PACER checkpoint at epoch 41,250 and run at most
# 10,000 additional epochs.  Stage 1 should override MAX_ITERATIONS=42750;
# later stages resume the same experiment with their corresponding configs.
# There is deliberately no wall-clock timeout: stop only at the epoch target.
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$OUT" \
  --epoch -1 \
  --num_envs 2048 \
  --max_iterations "$MAX_ITERATIONS" \
  --headless \
  --no_log

echo "EXIT=$?"
