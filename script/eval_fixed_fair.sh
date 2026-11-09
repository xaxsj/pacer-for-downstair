#!/usr/bin/env bash
# Re-run the fixed acceptance mesh list with the straight corridor path (no
# fixed path file, which does not match these meshes) and record videos.
set -u
cd /data/ljx/code/pacer-main
: "${WEIGHTS:?set WEIGHTS}"
TAG=${TAG:-accept_fair}
GPU=${GPU:-6}
SUM="experiment_result/validation/${TAG}_summary.txt"
: > "$SUM"

for m in wide_stairs narrow_stairs eval_r010_t035 eval_r012_t028 eval_r018_t032; do
  WEIGHTS="$WEIGHTS" MESH="$m" TAG="$TAG" GPU="$GPU" FRAMES=300 \
    bash script/eval_stage_mesh.sh >"/data/ljx/tmp/${TAG}_${m}.log" 2>&1
  st="experiment_result/validation/${TAG}/${m}/seed_0/states.pkl"
  {
    printf "%-18s " "$m"
    /data/ljx/envs/pacer/bin/python script/inspect_recording.py "$st" 2>/dev/null \
      | sed -n 2p
  } >> "$SUM"
done
cat "$SUM"
echo FAIR_ACCEPT_DONE
