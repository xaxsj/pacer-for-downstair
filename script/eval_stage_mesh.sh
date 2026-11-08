#!/usr/bin/env bash
# Evaluate a checkpoint on a stage-matched stair mesh: same route as training
# (approach -> up -> platform -> down), straight target path, no path file.
#   WEIGHTS=... MESH=s1_r10_t39_w28_0 TAG=... GPU=6 FRAMES=300
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"

: "${WEIGHTS:?set WEIGHTS}"
: "${MESH:?set MESH}"
TAG=${TAG:-stage_mesh}
SEED=${SEED:-0}
FRAMES=${FRAMES:-300}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml}
START_XY=${START_XY:-0.5,0.0}

OUT="experiment_result/validation/${TAG}/${MESH}/seed_${SEED}"
mkdir -p "$OUT"

XVFB_BIN=/data/ljx/tools/xvfb-root/usr/bin/Xvfb
[ -x "$XVFB_BIN" ] || XVFB_BIN=$(command -v Xvfb || true)
for n in 90 91 92 93 94 95 96 97 98 99; do
  [ ! -e "/tmp/.X${n}-lock" ] && DISPNUM=$n && break
done
: "${DISPNUM:=89}"
"$XVFB_BIN" :${DISPNUM} -screen 0 1280x720x24 +extension GLX +render -noreset -nolisten tcp \
  >/data/ljx/tmp/stagemesh_xvfb_${DISPNUM}.log 2>&1 &
XPID=$!
trap 'kill ${XPID} 2>/dev/null || true' EXIT
sleep 2
export DISPLAY=:${DISPNUM}

before_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
before_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$WEIGHTS" \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --real_mesh --mesh_name "$MESH" \
  --start_xy "$START_XY" \
  --seed "$SEED" \
  --auto_record --record_frames "$FRAMES" \
  >"$OUT/run.log" 2>&1 || true

after_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
after_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
[ -n "$after_s" ] && [ "$after_s" != "$before_s" ] && cp -f "$after_s" "$OUT/states.pkl"
[ -n "$after_v" ] && [ "$after_v" != "$before_v" ] && cp -f "$after_v" "$OUT/video.mp4"
echo "mesh=$MESH states=$(test -f "$OUT/states.pkl" && echo yes || echo no) video=$(test -f "$OUT/video.mp4" && echo yes || echo no)"
echo STAGE_MESH_DONE
