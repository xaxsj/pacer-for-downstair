"""Speed and torso pitch around the start of the first flight."""
import pickle
import sys

import numpy as np

st = pickle.load(open(sys.argv[1], "rb"))
root = np.asarray(st["root_states"])[:, 0, :13]
bpos = np.asarray(st["rigid_body_pos"])[:, 0]

pelvis, chest = bpos[:, 0], bpos[:, 11]
axis = chest - pelvis
tilt = np.degrees(np.arccos(np.clip(axis[:, 2] / np.linalg.norm(axis, axis=1), -1, 1)))
horiz = np.hypot(axis[:, 0], axis[:, 1])
back = np.degrees(np.arctan2(-horiz * np.sign(axis[:, 0] * 0 + 1), axis[:, 2]))

xy = root[:, :2]
speed = np.zeros(len(xy))
speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) * 30.0

lo, hi = int(sys.argv[2]), int(sys.argv[3])
print(f"{'f':>4} {'x':>6} {'z':>5} {'speed':>6} {'tilt':>6}")
for f in range(lo, hi):
    print(f"{f:4d} {root[f,0]:6.2f} {root[f,2]:5.2f} {speed[f]:6.2f} {tilt[f]:6.1f}")

seg = speed[lo:hi]
if len(seg):
    print(f"\nspeed in [{lo},{hi}): min {seg.min():.2f} mean {seg.mean():.2f} "
          f"max {seg.max():.2f}")
    pause = int(np.argmin(seg)) + lo
    print(f"slowest frame {pause} at x={root[pause,0]:.2f} "
          f"z={root[pause,2]:.2f} tilt={tilt[pause]:.1f} deg")
