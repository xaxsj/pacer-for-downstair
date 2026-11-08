#!/usr/bin/env bash
# Run every stage-matched mesh with its own stage checkpoint and summarise.
set -u
cd /data/ljx/code/pacer-main
SUMMARY=experiment_result/validation/stage_matched_summary.txt
: > "$SUMMARY"

run_one() {
  local w="$1" mesh="$2" tag="$3" gpu="$4"
  WEIGHTS="$w" MESH="$mesh" TAG="$tag" GPU="$gpu" FRAMES=300 \
    bash script/eval_stage_mesh.sh >"/data/ljx/tmp/${tag}_${mesh}.log" 2>&1
  local st="experiment_result/validation/${tag}/${mesh}/seed_0/states.pkl"
  [ -f "$st" ] || { echo "$tag,$mesh,NO_STATES"; return; }
  /data/ljx/envs/pacer/bin/python script/inspect_recording.py "$st" 2>/dev/null \
    | head -3 | tail -1
}

mkdir -p /data/ljx/tmp/s1w /data/ljx/tmp/s2w
cp -f output/exp/pacer_curriculum/stage1_final.pth /data/ljx/tmp/s1w/Humanoid.pth
cp -f output/exp/pacer_curriculum/stage2_final.pth /data/ljx/tmp/s2w/Humanoid.pth

for m in s1_r10_t39_w28_0 s1_r10_t36_w28_1 s1_r10_t41_w24_2 s1_r09_t42_w28_3; do
  echo "S1 $m" >> "$SUMMARY"
  run_one /data/ljx/tmp/s1w "$m" s1_matched 6 >> "$SUMMARY"
done
for m in s2_r12_t37_w30_0 s2_r13_t33_w18_1 s2_r12_t32_w28_2 s2_r10_t38_w30_3; do
  echo "S2 $m" >> "$SUMMARY"
  run_one /data/ljx/tmp/s2w "$m" s2_matched 6 >> "$SUMMARY"
done
cat "$SUMMARY"
echo MATCHED_DONE
