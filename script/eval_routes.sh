#!/usr/bin/env bash
# Evaluate a checkpoint on the Stage-3 style multi-turn route meshes.
#   WEIGHTS=... TAG=s4route bash script/eval_routes.sh
set -u
cd /data/ljx/code/pacer-main
: "${WEIGHTS:?set WEIGHTS}"
TAG=${TAG:-s4route}
GPU=${GPU:-6}
FRAMES=${FRAMES:-600}
SUM="experiment_result/validation/${TAG}_summary.txt"
: > "$SUM"

for m in s3route_r15_t37_w28_0 s3route_r12_t31_w28_1 s3route_r10_t36_w28_2 s3route_r14_t31_w32_3; do
  WEIGHTS="$WEIGHTS" MESH="$m" PATH_FILE="data/paths/${m}.json" TAG="$TAG" \
    GPU="$GPU" FRAMES="$FRAMES" \
    bash script/eval_stage_mesh.sh >"/data/ljx/tmp/${TAG}_${m}.log" 2>&1
  st="experiment_result/validation/${TAG}/${m}/seed_0/states.pkl"
  {
    echo "=== $m ==="
    /data/ljx/envs/pacer/bin/python script/inspect_recording.py "$st" 2>/dev/null | head -1
    /data/ljx/envs/pacer/bin/python script/route_progress.py "$st" 2>/dev/null | tail -5
  } >> "$SUM"
done
cat "$SUM"
echo ROUTES_DONE
