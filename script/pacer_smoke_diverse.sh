#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
OUT=output/exp/pacer_diverse_smoke
mkdir -p "$OUT"
if [ ! -f "$OUT/Humanoid.pth" ]; then cp output/release/pacer/Humanoid.pth "$OUT/Humanoid.pth"; fi
timeout 1200 python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1_progress.yaml --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl --network_path "$OUT" --epoch -1 --num_envs 1024 --max_iterations 41251 --headless --no_log
echo "EXIT=$?"
