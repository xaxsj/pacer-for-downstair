#!/bin/bash
# Build a 5.0 m wide 20-step staircase and verify it.
set -u
cd /root/autodl-tmp/pacer/repo || exit 1

export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python
export LD_LIBRARY_PATH=/root/miniconda3/lib:/usr/local/cuda-11.8/lib64
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:$PATH

NAME=stairs20w50
echo "===== 1. generate width=5.0 m ====="
python scripts/make_stairs_mesh.py --steps 20 --riser 0.15 --tread 0.32 --width 5.0 --name $NAME 2>&1 | tail -8

echo ""
echo "===== 2. mirror along x ====="
python scripts/mirror_mesh_x.py data/mesh/$NAME.pkl data/mesh/${NAME}_neg.pkl 2>&1 | tail -5

echo ""
echo "===== 3. geometry ====="
python script/inspect_mesh.py data/mesh/${NAME}_neg.pkl 2>&1 | head -7

echo ""
echo "===== 4. collision vs heightmap (y=0 and y=+2.0) ====="
xvfb-run -a python script/check_collision_vs_heightmap.py \
  --mesh data/mesh/${NAME}_neg.pkl \
  --xs=0.00,-2.00,-4.00,-6.00,-7.40 \
  --y 0.0 2>&1 | tail -10
xvfb-run -a python script/check_collision_vs_heightmap.py \
  --mesh data/mesh/${NAME}_neg.pkl \
  --xs=0.00,-4.00,-7.40 \
  --y 2.0 2>&1 | tail -8
