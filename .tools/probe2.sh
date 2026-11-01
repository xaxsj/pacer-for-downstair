cd /root/autodl-tmp/pacer/repo || exit 1
echo "--- current path json ---"
cat data/paths/stairs05_neg.json
echo ""
echo "--- mesh ---"
ls -l data/mesh/stairs05_neg.pkl
echo "--- flags ---"
grep -nE 'add_argument\("--(mesh_name|path_file|start_xy|start_yaw|auto_record|record_frames|disable_fall_check)' pacer/utils/config.py | head -20
echo "--- git status ---"
git status --short | head -20
