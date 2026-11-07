#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

OUT=output/exp/pacer_stage1_10k
mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then
  cp output/release/pacer/Humanoid.pth "$OUT/Humanoid.pth"
fi

# Original weights + 10,000 additional epochs; reward function unchanged.
timeout 43200 python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$OUT" \
  --epoch -1 \
  --num_envs 2048 \
  --max_iterations 51250 \
  --headless \
  --no_log

echo "EXIT=$?"
