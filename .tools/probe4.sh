cd /root/autodl-tmp/pacer/repo || exit 1
echo "=== start_yaw occurrences ==="
grep -rn "start_yaw" --include=*.py . | head -20
echo ""
echo "=== flags.start_yaw body ==="
grep -rn -A 40 "flags.start_yaw" pacer/env/tasks/base_task.py | head -70
