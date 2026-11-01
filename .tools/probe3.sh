cd /root/autodl-tmp/pacer/repo || exit 1
echo "--- where flags defined ---"
grep -rn "mesh_name" pacer/utils/*.py pacer/run.py 2>/dev/null | head -10
echo "--- path_file arg ---"
grep -rn "path_file" pacer/utils/config.py pacer/run.py 2>/dev/null | head -10
echo "--- auto_record ---"
grep -rn "auto_record\b" pacer/env/tasks/base_task.py 2>/dev/null | head -8
echo "--- load_path usage ---"
grep -rn "load_path\|fixed_path" pacer/env/util/traj_generator.py | head -10
