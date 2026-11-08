"""Print root height along a single-env recording."""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"])[:, 0, :13]
frames = [0, 40, 80, 120, 140, 160, 180, 220, 260, 300]
for f in frames:
    if f < len(rs):
        print(f"frame {f:3d}  x={rs[f, 0]:7.2f}  z={rs[f, 2]:6.2f}")
print("z min/max", round(float(rs[:, 2].min()), 2), round(float(rs[:, 2].max()), 2))
