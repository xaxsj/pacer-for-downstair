#!/usr/bin/env bash
# Wait for Stage 5 training, then validate it on the Stage-4 mesh set:
#   A) the four multi-turn route meshes (path-defined finish)
#   B) the five fixed acceptance meshes (no path file)
#   C) torso pitch statistics on one route rollout
set -u
cd /data/ljx/code/pacer-main

echo "[validate5] waiting for stage 5 to finish..."
for i in $(seq 1 240); do
  grep -q STAGE5_DONE /data/ljx/tmp/stage5_wrap.log 2>/dev/null && break
  sleep 30
done

W=output/exp/pacer_stage5_1000ep
SUM=experiment_result/validation/stage5_validation.txt
: > "$SUM"

{
  echo "=== stage5 weights ==="
  ls -la --time-style=long-iso "$W"/Humanoid.pth
  echo
  echo "=== A) multi-turn route meshes (path-defined finish) ==="
} >> "$SUM"
WEIGHTS="$W" TAG=s5route GPU=6 FRAMES=600 \
  bash script/eval_routes.sh >> "$SUM" 2>&1

{
  echo
  echo "=== B) fixed acceptance meshes (no path file) ==="
} >> "$SUM"
WEIGHTS="$W" TAG=s5fixed GPU=6 \
  bash script/eval_fixed_fair.sh >> "$SUM" 2>&1

{
  echo
  echo "=== C) torso pitch by slope (route r15) ==="
} >> "$SUM"
/data/ljx/envs/pacer/bin/python script/pitch_by_slope.py \
  experiment_result/validation/s5route/s3route_r15_t37_w28_0/seed_0/states.pkl \
  2>/dev/null | head -8 >> "$SUM"

cat "$SUM"
echo STAGE5_VALIDATED
