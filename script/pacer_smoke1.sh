#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES=5
OUT=output/exp/pacer_smoke1
mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then cp output/release/pacer/Humanoid.pth "$OUT/Humanoid.pth"; fi
timeout 600 python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1_smoke1.yaml --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl --network_path "$OUT" --epoch -1 --num_envs 1 --max_iterations 41251 --headless --no_log
echo "EXIT=$?"
