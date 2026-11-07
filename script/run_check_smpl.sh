#!/usr/bin/env bash
set -euo pipefail
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export CUDA_VISIBLE_DEVICES=""
python script/check_smpl_model.py
