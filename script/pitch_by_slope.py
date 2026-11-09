"""Torso pitch grouped by the local slope the robot is on.

pitch > 0 = torso leaning BACK relative to the direction of travel,
pitch < 0 = leaning FORWARD.
"""
import pickle
import sys

import numpy as np

st = pickle.load(open(sys.argv[1], "rb"))
root = np.asarray(st["root_states"])[:, 0, :13]
bpos = np.asarray(st["rigid_body_pos"])[:, 0]
n = len(root)

pelvis, chest = bpos[:, 0], bpos[:, 11]
axis = chest - pelvis
axis_n = axis / np.linalg.norm(axis, axis=1, keepdims=True)

d = np.zeros((n, 2))
d[1:-1] = root[2:, :2] - root[:-2, :2]
d[0] = d[1]
d[-1] = d[-2]
sp = np.linalg.norm(d, axis=1)
sp[sp < 1e-6] = 1e-6
dirv = d / sp[:, None]

dz = np.zeros(n)
dz[1:-1] = root[2:, 2] - root[:-2, 2]
dz[0] = dz[1]
dz[-1] = dz[-2]
slope = dz / sp

horiz = (axis_n[:, 0] * dirv[:, 0] + axis_n[:, 1] * dirv[:, 1])
pitch = np.degrees(np.arctan2(-horiz, axis_n[:, 2]))

moving = sp > 0.005
flat = moving & (np.abs(slope) < 0.12)
up = moving & (slope >= 0.12)
down = moving & (slope <= -0.12)

for name, m in (("FLAT   ", flat), ("ASCEND ", up), ("DESCEND", down)):
    if m.sum() == 0:
        print(f"{name}: no frames")
        continue
    p = pitch[m]
    print(f"{name}: n={m.sum():4d}  mean {p.mean():+6.1f}  median "
          f"{np.median(p):+6.1f}  p90 {np.percentile(p, 90):+6.1f}  "
          f"max {p.max():+6.1f}  min {p.min():+6.1f}")

print(f"\nall frames: mean {pitch.mean():+.1f}  "
      f"frames with pitch > +12 (clear lean back): {(pitch > 12).sum()}")
print("frames leaning back (>12 deg):")
for f in np.where(pitch > 12)[0]:
    kind = "FLAT" if flat[f] else ("UP" if up[f] else ("DOWN" if down[f] else "?"))
    print(f"  f{f:4d} pitch={pitch[f]:+6.1f} slope={slope[f]:+.2f} {kind} "
          f"z={root[f,2]:.2f}")
