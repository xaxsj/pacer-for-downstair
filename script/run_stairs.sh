#!/bin/bash
# PACER experiment launcher (ASCII only: non-ASCII bytes have corrupted
# line endings during scp on this setup, so keep this file plain ASCII).
#
# Usage (run from repo root):
#   bash script/run_stairs.sh --mesh stairs05_neg --path data/paths/stairs05_neg.json \
#        --yaw 1.5708 --traj_dt 1.0 --frames 300 --tag stairs05_dt1.0
#
# Products go to experiment_result/<tag>/ :
#   states.pkl   raw state dump (re-render with script/render_path.py)
#   auto.mp4     in-sim camera recording (GROUND TRUTH for posture)
#   run.log      full log

set -u

MESH="stairs05_neg"
PATHFILE="data/paths/stairs05_neg.json"
YAW="1.5708"
FRAMES="200"
TAG=""
START_XY="0.0,0.0"
NO_FALL_CHECK=""
FLAT=""
TRAJ_DT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mesh)      MESH="$2"; shift 2 ;;
    --path)      PATHFILE="$2"; shift 2 ;;
    --yaw)       YAW="$2"; shift 2 ;;
    --frames)    FRAMES="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    --start_xy)  START_XY="$2"; shift 2 ;;
    --traj_dt)   TRAJ_DT="$2"; shift 2 ;;
    --no_fall_check) NO_FALL_CHECK="--disable_fall_check"; shift ;;
    --flat)      FLAT="1"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [ -n "$FLAT" ]; then
  MESH_OPTS=""
  MESH_DESC="flat(default terrain)"
else
  MESH_OPTS="--real_mesh --mesh_name $MESH"
  MESH_DESC="mesh:$MESH"
fi

[ -z "$TAG" ] && TAG="run_$(date +%Y%m%d-%H%M%S)"

cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
OUTDIR="$REPO/experiment_result/$TAG"
mkdir -p "$OUTDIR"

export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python
export LD_LIBRARY_PATH=/root/miniconda3/lib:/usr/local/cuda-11.8/lib64
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:$PATH
mkdir -p /tmp/pacer_tmp/smpl
[ -e /tmp/smpl ] || ln -s /tmp/pacer_tmp/smpl /tmp/smpl

LOG="$OUTDIR/run.log"

# trajSampleTimestep override. Baseline copy is kept so the repo is restored.
CFG="pacer/data/cfg/pacer.yaml"
[ -f "$CFG.base" ] || cp "$CFG" "$CFG.base"
cp -f "$CFG.base" "$CFG"
if [ -n "$TRAJ_DT" ]; then
  sed -i "s/^  trajSampleTimestep: .*/  trajSampleTimestep: $TRAJ_DT/" "$CFG"
fi

echo "=== PACER run: $TAG ==="             | tee "$LOG"
echo "terrain=$MESH_DESC"                 | tee -a "$LOG"
echo "path=$PATHFILE"                     | tee -a "$LOG"
echo "yaw=$YAW frames=$FRAMES xy=$START_XY" | tee -a "$LOG"
grep -E "trajSampleTimestep|numTrajSamples|has_upright_start|stateInit" "$CFG" | tee -a "$LOG"

python - "$PATHFILE" "$CFG" <<'PY' | tee -a "$LOG"
import json, re, sys
p = json.load(open(sys.argv[1]))
pts = p["path"] if isinstance(p, dict) else p
L = sum(((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2
         + (pts[i+1][2]-pts[i][2])**2)**0.5
        for i in range(len(pts)-1))
dt = 0.5
for line in open(sys.argv[2]):
    m = re.match(r"\s+trajSampleTimestep:\s*([0-9.]+)", line)
    if m:
        dt = float(m.group(1))
speed = L / 10.0
print("path length = %.2f m -> implied speed = %.2f m/s (episode 10 s)" % (L, speed))
print("waypoint spacing = %.3f m  (speed %.2f x traj_dt %.2f s)" % (speed * dt, speed, dt))
PY

nohup xvfb-run -a -s "-screen 0 1280x720x24" python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml \
  --motion_file sample_data/amass_isaac_standing_upright_slim.pkl \
  --network_path output/release/pacer \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  $MESH_OPTS \
  --path_file "$PATHFILE" \
  --start_xy "$START_XY" --start_yaw "$YAW" \
  $NO_FALL_CHECK \
  --auto_record --record_frames "$FRAMES" >> "$LOG" 2>&1 &
PID=$!
echo "pid=$PID log=$LOG"
wait $PID
RC=$?
echo "exit=$RC" | tee -a "$LOG"

# Collect products by parsing THIS run's log (mtime-based picking is unreliable).
S=$(grep -oE 'states-[0-9-]+\.pkl' "$LOG" | tail -1)
V=$(grep -oE 'auto-[0-9-]+\.mp4' "$LOG" | tail -1)
if [ -n "$S" ] && [ -f "output/recordings/$S" ]; then
  cp -f "output/recordings/$S" "$OUTDIR/states.pkl"
  echo "states -> $OUTDIR/states.pkl"
else
  echo "WARN: no states produced"
fi
if [ -n "$V" ] && [ -f "output/renderings/$V" ]; then
  cp -f "output/renderings/$V" "$OUTDIR/auto.mp4"
  echo "video  -> $OUTDIR/auto.mp4"
else
  echo "WARN: no video produced"
fi

echo "=== done: $OUTDIR ==="
# Restore pacer.yaml so --traj_dt does not leak into later runs.
[ -f "$CFG.base" ] && cp -f "$CFG.base" "$CFG"
exit $RC
