echo "=== disk ==="; df -h /root/autodl-tmp | tail -1
echo "=== py ==="; /root/miniconda3/bin/python -c "import torch,sys;print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())" 2>&1 | tail -3
echo "=== isaacgym ==="; ls -d /root/autodl-tmp/isaacgym_pkg/isaacgym/python/isaacgym 2>&1; /root/miniconda3/bin/python -c "import sys;sys.path.insert(0,'/root/autodl-tmp/isaacgym_pkg/isaacgym/python');import isaacgym;print('isaacgym import OK')" 2>&1 | tail -2
echo "=== repo ==="; ls /root/autodl-tmp/pacer/repo 2>&1
echo "=== meshes/paths ==="; ls -1 /root/autodl-tmp/pacer/repo/data/mesh/ 2>&1; ls -1 /root/autodl-tmp/pacer/repo/data/paths/ 2>&1
echo "=== recordings ==="; ls -lt /root/autodl-tmp/pacer/repo/output/recordings/ 2>&1 | head -6
echo "=== results ==="; ls -lt /root/autodl-tmp/pacer/repo/experiment_result/ 2>&1 | head -6
echo "=== running procs ==="; ps aux | grep -E "python|run.py" | grep -v grep | head -5