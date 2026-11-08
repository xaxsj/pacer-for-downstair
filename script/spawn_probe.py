"""Print the spawn pose of a single-env recording."""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"])[:, 0, :13]
print("frame0 root xyz:", np.round(rs[0, :3], 3))
print("frame0 heading deg:", round(float(np.degrees(np.arctan2(
    2 * (rs[0, 3] * rs[0, 4] + rs[0, 6] * rs[0, 5]),
    1 - 2 * (rs[0, 4] ** 2 + rs[0, 5] ** 2)))), 1))
print("z over first 10 frames:", np.round(rs[:10, 2], 2))
print("x over first 10 frames:", np.round(rs[:10, 0], 2))
print("z range:", round(float(rs[:, 2].min()), 2), "->", round(float(rs[:, 2].max()), 2))
