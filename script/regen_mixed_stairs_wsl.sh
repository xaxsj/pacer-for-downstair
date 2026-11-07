#!/bin/bash
# Regenerate the mixed stair pkl/path with the WSL (Isaac Gym) Python.
# Use this when the pkl was produced on Windows/numpy 2 and must run under
# the server's numpy 1.x environment.
set -u

export PATH=/home/lvjianxiong0505/miniconda3/envs/mimickit/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/lvjianxiong0505/miniconda3/envs/mimickit/lib

cd "$(dirname "$0")/.." || exit 1
python script/make_mixed_stairs_scene.py --out data/mesh/mixed_stairs.pkl
python script/make_mixed_stairs_path.py --out data/paths/mixed_stairs.json
