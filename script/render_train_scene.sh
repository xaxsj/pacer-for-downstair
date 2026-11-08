#!/usr/bin/env bash
# Render one env on the TRAINING terrain (procedural stairs) from a chosen spawn
# pose, so the camera actually follows a robot that climbs.
#
#   TAG=<name>  SX="x,y"  SYAW=<radians>  FRAMES=500  GPU=0
#
# Pick a pose with: python script/pick_climber.py <rollout tag>
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

: "${TAG:?set TAG}"
SX=${SX:-"71.03,358.79"}
SYAW=${SYAW:-3.0319}
FRAMES=${FRAMES:-500}

XVFB_BIN=/data/ljx/tools/xvfb-root/usr/bin/Xvfb
[ -x "$XVFB_BIN" ] || XVFB_BIN=$(command -v Xvfb || true)
for n in 90 91 92 93 94 95 96 97 98 99; do
  [ ! -e "/tmp/.X${n}-lock" ] && DISPNUM=$n && break
done
: "${DISPNUM:=89}"
"$XVFB_BIN" :${DISPNUM} -screen 0 1280x720x24 +extension GLX +render -noreset -nolisten tcp \
  >/data/ljx/tmp/render_${TAG}_xvfb.log 2>&1 &
XPID=$!
trap 'kill ${XPID} 2>/dev/null || true' EXIT
sleep 2
export DISPLAY=:${DISPNUM}

OUT="experiment_result/validation/${TAG}/seed_0"
mkdir -p "$OUT"
before_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path output/exp/pacer_stage1_10k \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --seed 0 --auto_record --record_frames "$FRAMES" \
  --start_xy "$SX" --start_yaw "$SYAW" \
  >"$OUT/run.log" 2>&1 || true
after_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
[ -n "$after_v" ] && [ "$after_v" != "$before_v" ] && cp -f "$after_v" "$OUT/video.mp4"
echo "video=$(test -f "$OUT/video.mp4" && echo yes || echo no) dir=$OUT"
echo RENDER_DONE
