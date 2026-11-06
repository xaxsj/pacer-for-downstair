#!/usr/bin/env bash
# Stage 1: KIT stairs + KIT flat AMP motions, stairs/flat terrain only.
set -euo pipefail

source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

mkdir -p output/exp/pacer_stage1_stairs
if [ ! -f output/exp/pacer_stage1_stairs/Humanoid.pth ]; then
  cp output/release/pacer/Humanoid.pth output/exp/pacer_stage1_stairs/Humanoid.pth
fi

python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path output/exp/pacer_stage1_stairs \
  --epoch -1 \
  --num_envs 1024 \
  --max_iterations 41750 \
  --headless \
  --no_log
