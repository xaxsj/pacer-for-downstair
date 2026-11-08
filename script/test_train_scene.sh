#!/usr/bin/env bash
# Roll out the latest stage1_10k checkpoint ON THE TRAINING SCENE
# (procedural terrain from pacer_stage1_stairs.yaml, no --real_mesh mesh).
# Envs spawn on random walkable locations with random heading, exactly as in training.
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

NENV=${NENV:-64}
FRAMES=${FRAMES:-600}
SEED=${SEED:-0}
TAG=${TAG:-train_scene}
OUT="experiment_result/validation/${TAG}/seed_${SEED}"
mkdir -p "$OUT"

XVFB_BIN=/data/ljx/tools/xvfb-root/usr/bin/Xvfb
[ -x "$XVFB_BIN" ] || XVFB_BIN=$(command -v Xvfb || true)
[ -n "$XVFB_BIN" ] || { echo "no Xvfb" >&2; exit 1; }
for n in 90 91 92 93 94 95 96 97 98 99; do
  [ ! -e "/tmp/.X${n}-lock" ] && DISPNUM=$n && break
done
: "${DISPNUM:=89}"
"$XVFB_BIN" :${DISPNUM} -screen 0 1280x720x24 +extension GLX +render -noreset -nolisten tcp \
  >/data/ljx/tmp/trainscene_xvfb_${DISPNUM}.log 2>&1 &
XPID=$!
trap 'kill ${XPID} 2>/dev/null || true' EXIT
sleep 2
export DISPLAY=:${DISPNUM}

before_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path output/exp/pacer_stage1_10k \
  --test --num_envs "$NENV" --epoch -1 --no_virtual_display \
  --seed "$SEED" --auto_record --record_frames "$FRAMES" \
  >"$OUT/run.log" 2>&1 || true

after_s=$(ls -t output/recordings/states-*.pkl 2>/dev/null | head -1 || true)
if [ -n "$after_s" ] && [ "$after_s" != "$before_s" ]; then cp -f "$after_s" "$OUT/states.pkl"; fi
echo "states=$(test -f "$OUT/states.pkl" && echo yes || echo no) dir=$OUT"
echo TRAIN_SCENE_DONE
