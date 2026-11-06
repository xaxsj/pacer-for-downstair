#!/usr/bin/env bash
# Evaluate a PACER checkpoint on a mesh/path pair and record an mp4.
#
# Usage:
#   WEIGHTS=output/exp/pacer_stage1_stairs \
#   CFG_ENV=pacer/data/cfg/pacer_stage1_stairs.yaml \
#   CFG_TRAIN=pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
#   MESH=mixed_stairs PATH_FILE=data/paths/mixed_stairs.json TAG=stage1_mixed \
#   FRAMES=900 START_XY=0.5,0.0 START_YAW=-1.4486 \
#   bash script/eval_pacer.sh
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

WEIGHTS=${WEIGHTS:-output/exp/pacer_stage1_stairs}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml}
MESH=${MESH:-mixed_stairs}
PATH_FILE=${PATH_FILE:-data/paths/mixed_stairs.json}
TAG=${TAG:-stage1_mixed}
FRAMES=${FRAMES:-900}
START_XY=${START_XY:-0.5,0.0}
START_YAW=${START_YAW:--1.4486}
GPU=${CUDA_VISIBLE_DEVICES:-4}

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p experiment_result/eval_stage1 output/renderings output/recordings

BEFORE=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)

XVFB_BIN=/data/ljx/tools/xvfb-root/usr/bin/Xvfb
if [ ! -x "$XVFB_BIN" ]; then
  XVFB_BIN=$(command -v Xvfb || true)
fi
if [ -z "$XVFB_BIN" ]; then
  echo "ERROR: Xvfb not found" >&2
  exit 1
fi

for n in 90 91 92 93 94 95 96 97 98 99; do
  if [ ! -e "/tmp/.X${n}-lock" ]; then DISPNUM=$n; break; fi
done
: "${DISPNUM:=89}"

"$XVFB_BIN" :${DISPNUM} -screen 0 1280x720x24 +extension GLX +render -noreset -nolisten tcp \
  >"/data/ljx/tmp/eval_xvfb_${DISPNUM}.log" 2>&1 &
XVFB_PID=$!
trap 'kill ${XVFB_PID} 2>/dev/null || true' EXIT
sleep 2
export DISPLAY=:${DISPNUM}

echo "[eval] weights=$WEIGHTS cfg_env=$CFG_ENV mesh=$MESH tag=$TAG frames=$FRAMES gpu=$GPU"

python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file sample_data/amass_isaac_standing_upright_slim.pkl \
  --network_path "$WEIGHTS" \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --real_mesh --mesh_name "$MESH" \
  --path_file "$PATH_FILE" \
  --start_xy "$START_XY" --start_yaw "$START_YAW" \
  --disable_fall_check \
  --auto_record --record_frames "$FRAMES"

AFTER=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
if [ -n "$AFTER" ] && [ "$AFTER" != "$BEFORE" ]; then
  out="experiment_result/eval_stage1/${TAG}.mp4"
  cp -f "$AFTER" "$out"
  echo "[eval] video -> $out"
else
  echo "[eval] WARNING: no new mp4 produced" >&2
  exit 2
fi
