#!/bin/bash
# Render the mixed-stairs scene with episode resets disabled.
# --disable_fall_check turns off both fall and trajectory-deviation termination,
# so the character keeps moving (and falling) instead of being respawned.
set -u

cd "$(dirname "$0")/.." || exit 1

export PATH=/root/autodl-tmp/env/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python:/root/autodl-tmp/pacer/repo/pacer
export LD_LIBRARY_PATH=/root/autodl-tmp/env/lib:${LD_LIBRARY_PATH:-}

FRAMES=${1:-600}
TAG=${2:-no_reset}

mkdir -p "experiment_result/${TAG}"

xvfb-run -a -s "-screen 0 1280x720x24" python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml \
  --motion_file sample_data/amass_isaac_standing_upright_slim.pkl \
  --network_path output/release/pacer \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --real_mesh --mesh_name mixed_stairs \
  --path_file data/paths/mixed_stairs.json \
  --start_xy 0.5,0.0 --start_yaw -1.4486 \
  --disable_fall_check \
  --auto_record --record_frames "$FRAMES"
