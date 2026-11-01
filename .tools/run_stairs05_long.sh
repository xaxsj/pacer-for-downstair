#!/bin/bash
cd /root/autodl-tmp/pacer/repo || exit 1

# --- 1. 备份并写入 8.5m 路径（旧 2.4m 版本保留）---
cp -n data/paths/stairs05_neg.json data/paths/stairs05_neg.json.bak2p4m 2>/dev/null
cat > data/paths/stairs05_neg.json <<'EOF'
{"name": "stairs05_neg", "path": [[0.0, 0.0, 0.75], [-2.4, 0.0, 0.0], [-8.5, 0.0, 0.0]]}
EOF
echo "=== new path (=8.5m -> 0.85 m/s) ==="
cat data/paths/stairs05_neg.json
echo ""

# --- 2. 环境 ---
export PYTHONPATH=/root/autodl-tmp/isaacgym_pkg/isaacgym/python
export LD_LIBRARY_PATH=/root/miniconda3/lib:/usr/local/cuda-11.8/lib64
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:$PATH
mkdir -p /tmp/pacer_tmp/smpl
[ -e /tmp/smpl ] || ln -s /tmp/pacer_tmp/smpl /tmp/smpl
mkdir -p log

LOG="log/run_stairs05_long.log"
: > "$LOG"

nohup xvfb-run -a -s "-screen 0 1280x720x24" python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer.yaml \
  --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml \
  --motion_file sample_data/amass_isaac_standing_upright_slim.pkl \
  --network_path output/release/pacer \
  --test --num_envs 1 --epoch -1 --no_virtual_display \
  --real_mesh --mesh_name stairs05_neg \
  --path_file data/paths/stairs05_neg.json \
  --start_xy "0.0,0.0" --start_yaw 1.57 \
  --auto_record --record_frames 200 >> "$LOG" 2>&1 &

PID=$!
echo "started pid=$PID log=$LOG"
sleep 8
echo "=== early log ==="
tail -n 25 "$LOG"
