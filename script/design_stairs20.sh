#!/bin/bash
# Design the 20-step staircase for the -x descent direction, then verify it.
set -u
cd /root/autodl-tmp/pacer/repo || exit 1

export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python
export LD_LIBRARY_PATH=/root/miniconda3/lib:/usr/local/cuda-11.8/lib64
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:$PATH

echo "===== 0. backup existing meshes ====="
mkdir -p data/mesh_backup
cp -n data/mesh/stairs20.pkl data/mesh_backup/ 2>/dev/null && echo "  backed up stairs20.pkl"
cp -n data/mesh/stairs20_neg.pkl data/mesh_backup/ 2>/dev/null && echo "  backed up stairs20_neg.pkl"

echo ""
echo "===== 1. generate 20-step stairs (riser 0.15 / tread 0.32 / width 1.2) ====="
python scripts/make_stairs_mesh.py --steps 20 --riser 0.15 --tread 0.32 --width 1.2 --name stairs20 2>&1 | tail -8

echo ""
echo "===== 2. mirror along x (descent becomes -x) ====="
python scripts/mirror_mesh_x.py data/mesh/stairs20.pkl data/mesh/stairs20_neg.pkl 2>&1 | tail -6

echo ""
echo "===== 3. geometry check: stairs20_neg ====="
python script/inspect_mesh.py data/mesh/stairs20_neg.pkl 2>&1 | head -14

echo ""
echo "===== 4. collision vs heightmap (11 probes) ====="
xvfb-run -a python script/check_collision_vs_heightmap.py \
  --mesh data/mesh/stairs20_neg.pkl \
  --xs=0.20,0.00,-0.40,-0.90,-1.60,-2.60,-3.60,-4.60,-5.60,-6.60,-7.40 \
  --y 0.0 2>&1 | tail -16
