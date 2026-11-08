#!/usr/bin/env bash
# Run the four Stage-3 route meshes with the Stage 3 checkpoint and summarise.
set -u
cd /data/ljx/code/pacer-main
W=/data/ljx/tmp/s3w
SUM=experiment_result/validation/s3route_summary.txt
: > "$SUM"

for m in s3route_r15_t37_w28_0 s3route_r12_t31_w28_1 s3route_r10_t36_w28_2 s3route_r14_t31_w32_3; do
  WEIGHTS=$W MESH=$m PATH_FILE="data/paths/${m}.json" TAG=s3route GPU=6 FRAMES=600 \
    bash script/eval_stage_mesh.sh >"/data/ljx/tmp/route_${m}.log" 2>&1
  st="experiment_result/validation/s3route/${m}/seed_0/states.pkl"
  {
    echo "=== $m ==="
    /data/ljx/envs/pacer/bin/python script/route_progress.py "$st" 2>/dev/null \
      | tail -6
  } >> "$SUM"
done
cat "$SUM"
echo S3ROUTE_DONE
