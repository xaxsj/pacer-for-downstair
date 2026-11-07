#!/usr/bin/env bash
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES=4
SEEDS="0 1"
FRAMES=300
for mesh in wide_stairs narrow_stairs; do
  WEIGHTS=output/release/pacer CFG_ENV=pacer/data/cfg/pacer.yaml \
  CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml \
  MESH="$mesh" PATH_FILE=data/paths/mixed_stairs.json TAG=complete_release \
  SEEDS="$SEEDS" FRAMES="$FRAMES" bash script/run_validation_matrix.sh

  WEIGHTS=output/exp/pacer_stage1_stairs CFG_ENV=pacer/data/cfg/pacer_stage1_stairs.yaml \
  CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  MESH="$mesh" PATH_FILE=data/paths/mixed_stairs.json TAG=complete_stage1 \
  SEEDS="$SEEDS" FRAMES="$FRAMES" bash script/run_validation_matrix.sh

  WEIGHTS=output/exp/pacer_stage1_diverse CFG_ENV=pacer/data/cfg/pacer_stage1_stairs.yaml \
  CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  MESH="$mesh" PATH_FILE=data/paths/mixed_stairs.json TAG=complete_diverse \
  SEEDS="$SEEDS" FRAMES="$FRAMES" bash script/run_validation_matrix.sh
done
echo "COMPLETE_SUITE_DONE"
