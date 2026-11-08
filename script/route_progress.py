"""Summarise a route rollout: where the robot went and how high it got."""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"])[:, 0, :13]
n = len(rs)
print(f"frames={n} ({n / 30.0:.1f}s)")
for f in range(0, n, max(n // 8, 1)):
    print(f"  f{f:4d}  x={rs[f, 0]:6.2f} y={rs[f, 1]:6.2f} z={rs[f, 2]:5.2f}")
p = rs[-1]
print(f"final   x={p[0]:6.2f} y={p[1]:6.2f} z={p[2]:5.2f}")
terr = rs[:, 2] - 0.95
print(f"terrain height: start {terr[0]:.2f}  max {terr.max():.2f}  "
      f"final {terr[-1]:.2f}")
print(f"y range {rs[:, 1].min():.2f} .. {rs[:, 1].max():.2f}")
