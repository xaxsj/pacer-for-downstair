#!/bin/bash
# Mixed stair scene launcher for PACER (Isaac Gym).
#
#   approach flat -> up 10 steps -> top flat -> 90 deg turn -> down 10 -> bottom
#
# Usage (from repo root):
#   bash script/run_mixed_stairs.sh
#   bash script/run_mixed_stairs.sh --frames 900 --tag mixed_stairs_try2
#
# It regenerates data/mesh/mixed_stairs.pkl and data/paths/mixed_stairs.json,
# then calls script/run_stairs.sh (which owns the Isaac Gym environment).
set -u

cd "$(dirname "$0")/.." || exit 1

MESH="mixed_stairs"
PATHFILE="data/paths/mixed_stairs.json"
START_XY="0.5,0.0"
YAW="0.0"
FRAMES="900"
TAG="mixed_stairs"
REBUILD="1"

while [ $# -gt 0 ]; do
  case "$1" in
    --frames)   FRAMES="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --start_xy) START_XY="$2"; shift 2 ;;
    --yaw)      YAW="$2"; shift 2 ;;
    --no_rebuild) REBUILD="0"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [ "$REBUILD" = "1" ]; then
  echo "=== rebuild mixed stair scene ==="
  python script/make_mixed_stairs_scene.py --out "data/mesh/$MESH.pkl" || exit 1
  python script/make_mixed_stairs_path.py --out "$PATHFILE" || exit 1
fi

echo "=== run PACER on $MESH ==="
bash script/run_stairs.sh \
  --mesh "$MESH" \
  --path "$PATHFILE" \
  --yaw "$YAW" \
  --start_xy "$START_XY" \
  --frames "$FRAMES" \
  --tag "$TAG"
