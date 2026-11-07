#!/usr/bin/env bash
# Run one checkpoint/mesh/seed matrix without disabling fall termination.
# The resulting states pkl contains the first reset; videos show the fall.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

WEIGHTS=${WEIGHTS:-output/release/pacer}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml}
MESH=${MESH:-wide_stairs}
PATH_FILE=${PATH_FILE:-data/paths/mixed_stairs.json}
TAG=${TAG:-release_wide}
SEEDS=${SEEDS:-$(seq 0 19)}
FRAMES=${FRAMES:-900}
GPU=${CUDA_VISIBLE_DEVICES:-4}
START_XY=${START_XY:-0.5,0.0}
START_YAW=${START_YAW:--1.4486}

export CUDA_VISIBLE_DEVICES="$GPU"
XVFB_BIN=/data/ljx/tools/xvfb-root/usr/bin/Xvfb
[ -x "$XVFB_BIN" ] || XVFB_BIN=$(command -v Xvfb || true)
[ -n "$XVFB_BIN" ] || { echo "no Xvfb" >&2; exit 1; }
for n in 90 91 92 93 94 95 96 97 98 99; do
  [ ! -e "/tmp/.X${n}-lock" ] && DISPNUM=$n && break
done
: "${DISPNUM:=89}"
"$XVFB_BIN" :${DISPNUM} -screen 0 1280x720x24 +extension GLX +render -noreset -nolisten tcp \
  >/data/ljx/tmp/validation_xvfb_${DISPNUM}.log 2>&1 &
XPID=$!
trap 'kill ${XPID} 2>/dev/null || true' EXIT
sleep 2
export DISPLAY=:${DISPNUM}

ROOT="experiment_result/validation/${TAG}/${MESH}"
mkdir -p "$ROOT"
for seed in $SEEDS; do
  run="$ROOT/seed_${seed}"
  mkdir -p "$run"
  before_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
  before_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
  echo "[validation] tag=$TAG mesh=$MESH seed=$seed weights=$WEIGHTS"
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
    --seed "$seed" \
    --auto_record --record_frames "$FRAMES" \
    >"$run/run.log" 2>&1 || true

  after_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
  after_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
  if [ -n "$after_s" ] && [ "$after_s" != "$before_s" ]; then cp -f "$after_s" "$run/states.pkl"; fi
  if [ -n "$after_v" ] && [ "$after_v" != "$before_v" ]; then cp -f "$after_v" "$run/video.mp4"; fi
  echo "  states=$(test -f "$run/states.pkl" && echo yes || echo no) video=$(test -f "$run/video.mp4" && echo yes || echo no)"
done
