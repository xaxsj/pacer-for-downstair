#!/usr/bin/env bash
# Roll out the current config on the new straight-stair terrain and record a
# video (single env, camera follows it).  GPU=<id> TAG=<name> FRAMES=300
# WEIGHTS=<ckpt dir> SEED=0 YAW=0 (radians; 0 faces +x, i.e. up the flight)
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"

: "${TAG:?set TAG}"
WEIGHTS=${WEIGHTS:-output/exp/smoke_straight}
FRAMES=${FRAMES:-300}
SEED=${SEED:-0}
CFG_ENV=${CFG_ENV:-pacer/data/cfg/pacer_stage1_stairs.yaml}
CFG_TRAIN=${CFG_TRAIN:-pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1_smoke.yaml}

# Only override the spawn yaw when asked: passing --start_yaw makes the env use
# a *relative* rotation on top of the clip heading, which defeats the
# stair_heading_mode: straight per-env correction.
EXTRA_ARGS=()
if [ -n "${YAW:-}" ]; then
  EXTRA_ARGS+=(--start_yaw "$YAW")
fi
if [ -n "${START_XY:-}" ]; then
  EXTRA_ARGS+=(--start_xy "$START_XY")
fi

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

OUT="experiment_result/validation/${TAG}/seed_${SEED}"
mkdir -p "$OUT"
before_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
before_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$WEIGHTS" \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --seed $SEED "${EXTRA_ARGS[@]}" \
  --auto_record --record_frames "$FRAMES" \
  >"$OUT/run.log" 2>&1 || true
after_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
after_v=$(ls -t output/renderings/auto-*.mp4 2>/dev/null | head -1 || true)
[ -n "$after_s" ] && [ "$after_s" != "$before_s" ] && cp -f "$after_s" "$OUT/states.pkl"
[ -n "$after_v" ] && [ "$after_v" != "$before_v" ] && cp -f "$after_v" "$OUT/video.mp4"
echo "states=$(test -f "$OUT/states.pkl" && echo yes || echo no) video=$(test -f "$OUT/video.mp4" && echo yes || echo no)"
echo RENDER_SMOKE_DONE

