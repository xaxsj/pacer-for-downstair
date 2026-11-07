#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES=4
timeout 180 python script/probe_obs_dims.py > /data/ljx/tmp/probe_obs_dims_gpu4.log 2>&1 || true
echo EXIT=$?
tail -50 /data/ljx/tmp/probe_obs_dims_gpu4.log
