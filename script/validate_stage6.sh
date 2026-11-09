#!/usr/bin/env bash
# After Stage 6 finishes: run the four route meshes with the stage-6 config so
# the clean-finish rule (stop + upright) is active, and report the outcome.
set -u
cd /data/ljx/code/pacer-main

echo "[validate6] waiting for stage 6..."
for i in $(seq 1 120); do
  grep -q STAGE6_DONE /data/ljx/tmp/stage6_wrap.log 2>/dev/null && break
  sleep 30
done

W=output/exp/pacer_stage6_500ep
export CFG_ENV=pacer/data/cfg/pacer_stage6_stairs.yaml
SUM=experiment_result/validation/stage6_validation.txt
: > "$SUM"

echo "=== stage6 route meshes (stop + upright required) ===" | tee -a "$SUM"
WEIGHTS="$W" TAG=s6route GPU=6 FRAMES=700 \
  bash script/eval_routes.sh 2>&1 | tee -a "$SUM"

echo | tee -a "$SUM"
echo "=== termination detail ===" | tee -a "$SUM"
for m in s3route_r15_t37_w28_0 s3route_r12_t31_w28_1 s3route_r10_t36_w28_2 s3route_r14_t31_w32_3; do
  echo "-- $m" | tee -a "$SUM"
  /data/ljx/envs/pacer/bin/python script/inspect_recording.py \
    "experiment_result/validation/s6route/$m/seed_0/states.pkl" 2>/dev/null \
    | head -5 | tee -a "$SUM"
done
echo STAGE6_VALIDATED
