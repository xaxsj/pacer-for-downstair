#!/bin/bash
# Build a WIDER 20-step staircase (width 2.4 m) and verify it.
set -u
cd /root/autodl-tmp/pacer/repo || exit 1

export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python
export LD_LIBRARY_PATH=/root/miniconda3/lib:/usr/local/cuda-11.8/lib64
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:$PATH

NAME=stairs20w24
echo "===== 1. generate width=2.4 m ====="
python scripts/make_stairs_mesh.py --steps 20 --riser 0.15 --tread 0.32 --width 2.4 --name $NAME 2>&1 | tail -8

echo ""
echo "===== 2. mirror along x ====="
python scripts/mirror_mesh_x.py data/mesh/$NAME.pkl data/mesh/${NAME}_neg.pkl 2>&1 | tail -5

echo ""
echo "===== 3. geometry ====="
python script/inspect_mesh.py data/mesh/${NAME}_neg.pkl 2>&1 | head -8
echo "--- platform (top layer) ---"
python script/inspect_mesh.py data/mesh/${NAME}_neg.pkl 2>&1 | grep -A 4 "顶部平台"

echo ""
echo "===== 4. collision vs heightmap ====="
xvfb-run -a python script/check_collision_vs_heightmap.py \
  --mesh data/mesh/${NAME}_neg.pkl \
  --xs=0.20,0.00,-0.40,-1.60,-3.60,-5.60,-7.40 \
  --y 0.0 2>&1 | tail -12
