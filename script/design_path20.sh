#!/bin/bash
# Design the path for the 20-step staircase (ascent along -x) and report its speed.
set -u
cd /root/autodl-tmp/pacer/repo || exit 1

export PATH=/root/miniconda3/bin:$PATH

cp -n data/paths/stairs20_neg.json data/paths/stairs20_neg.json.bak 2>/dev/null && echo "backed up old path"

python - <<'PY'
import json, math
# Platform top z=3.00, stairs run x: -0.8 -> -7.2 (20 x 0.32)
# Then 0.70 m of ground runway.  Total length targeted at 8.50 m -> 0.85 m/s.
pts = [[0.0, 0.0, 3.00], [-7.20, 0.0, 0.0], [-7.90, 0.0, 0.0]]
payload = {"name": "stairs20_neg",
           "note": "20 steps, riser 0.15 / tread 0.32 / width 1.2, descending along -x",
           "path": pts}
open("data/paths/stairs20_neg.json", "w").write(json.dumps(payload, indent=2))

L = sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))
speed = L / 10.0
print("path =", json.dumps(pts))
print("length = %.3f m -> implied speed = %.3f m/s" % (L, speed))
for target in (2.0, 2.5):
    print("  waypoint spacing %.1f m -> traj_dt = %.2f s" % (target, target / speed))
PY
