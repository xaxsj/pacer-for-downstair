"""Torso pitch measured against the body's own forward axis (robust at low speed).

forward  = root quaternion applied to +X (the body axis), projected to XY
axis     = chest - pelvis
pitch    = atan2(dot(axis_xy, forward), axis_z)
            positive -> chest ahead of pelvis  (leaning FORWARD)
            negative -> leaning BACK
"""
import pickle
import sys

import numpy as np


def quat_rotate(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


st = pickle.load(open(sys.argv[1], "rb"))
root = np.asarray(st["root_states"])[:, 0, :13]
bpos = np.asarray(st["rigid_body_pos"])[:, 0]
n = len(root)

pelvis, chest = bpos[:, 0], bpos[:, 11]
axis = chest - pelvis
forward = np.zeros((n, 2))
for i in range(n):
    f = quat_rotate(root[i, 3:7], np.array([1.0, 0.0, 0.0]))[:2]
    nrm = np.linalg.norm(f)
    forward[i] = f / nrm if nrm > 1e-6 else np.array([1.0, 0.0])

horiz = (axis[:, 0] * forward[:, 0] + axis[:, 1] * forward[:, 1])
pitch = np.degrees(np.arctan2(horiz, axis[:, 2]))

# local slope from the vertical motion per unit horizontal displacement
d = np.zeros((n, 2))
d[1:-1] = root[2:, :2] - root[:-2, :2]
d[0], d[-1] = d[1], d[-2]
sp = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
dz = np.zeros(n)
dz[1:-1] = root[2:, 2] - root[:-2, 2]
dz[0], dz[-1] = dz[1], dz[-2]
slope = dz / sp
speed = sp * 30.0

moving = sp > 0.004
flat = moving & (np.abs(slope) < 0.12)
up = moving & (slope >= 0.12)
down = moving & (slope <= -0.12)

print(f"{'class':8s} {'n':>5s} {'mean':>7s} {'median':>7s} {'p10':>7s} "
      f"{'p90':>7s} {'max':>7s}")
for name, m in (("FLAT", flat), ("ASCEND", up), ("DESCEND", down)):
    if m.sum() == 0:
        continue
    p = pitch[m]
    print(f"{name:8s} {m.sum():5d} {p.mean():+7.1f} {np.median(p):+7.1f} "
          f"{np.percentile(p,10):+7.1f} {np.percentile(p,90):+7.1f} "
          f"{p.max():+7.1f}")
print(f"mean speed {speed.mean():.2f} m/s   frames >0.5 m/s: "
      f"{(speed>0.5).sum()}/{n}")
