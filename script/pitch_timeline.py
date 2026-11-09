"""Timeline of torso pitch, position and speed for a route rollout."""
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
d[0], d[-1] = d[1], d[-2]
sp = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
dirv = d / sp[:, None]
speed = sp * 30.0
pitch = np.degrees(np.arctan2(
    -(axis_n[:, 0] * dirv[:, 0] + axis_n[:, 1] * dirv[:, 1]), axis_n[:, 2]))

dz = np.zeros(n)
dz[1:-1] = root[2:, 2] - root[:-2, 2]
dz[0], dz[-1] = dz[1], dz[-2]
slope = dz / sp

step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
print(f"{'f':>4} {'t(s)':>6} {'x':>6} {'y':>6} {'z':>5} {'spd':>5} "
      f"{'slope':>6} {'pitch':>7}  phase")
for f in range(0, n, step):
    if slope[f] > 0.12:
        ph = "UP"
    elif slope[f] < -0.12:
        ph = "DOWN"
    else:
        ph = "FLAT"
    if abs(pitch[f]) > 10:
        ph += "  <<"
    print(f"{f:4d} {f/30:6.2f} {root[f,0]:6.2f} {root[f,1]:6.2f} "
          f"{root[f,2]:5.2f} {speed[f]:5.2f} {slope[f]:+6.2f} "
          f"{pitch[f]:+7.1f}  {ph}")
