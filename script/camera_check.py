"""Check the camera's relation to the body's own forward axis.

For selected frames it computes the body +X direction from the root quaternion,
the direction from the robot to the camera (the same path-relative rule the
renderer uses), and the angle between them:
    0 deg   -> camera in front of the body
   90 deg   -> camera on the body's left/right side
  180 deg   -> camera behind the body
"""
import json
import pickle
import sys

import numpy as np


def quat_rotate(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def left_of(poly, xy):
    a = poly[:-1, :2]
    b = poly[1:, :2]
    ab = b - a
    seg = np.linalg.norm(ab, axis=1)
    keep = seg > 1e-6
    a, ab, seg = a[keep], ab[keep], seg[keep]
    t = np.clip(((xy - a) * ab).sum(axis=1) / (seg ** 2), 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = np.linalg.norm(xy - proj, axis=1)
    i = int(np.argmin(d))
    dx, dy = ab[i] / seg[i]
    return np.array([-dy, dx])


states = pickle.load(open(sys.argv[1], "rb"))
path = np.asarray(json.load(open(sys.argv[2]))["path"], dtype=float)
rs = np.asarray(states["root_states"])[:, 0, :13]

print(f"{'frame':>5} {'body_fwd(deg)':>13} {'cam_dir(deg)':>13} {'angle':>7}")
for f in range(0, len(rs), max(len(rs) // 10, 1)):
    q = rs[f, 3:7]
    fwd = quat_rotate(q, np.array([1.0, 0.0, 0.0]))[:2]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
    left = left_of(path, rs[f, :2])
    ang = np.degrees(np.arccos(np.clip(fwd @ left, -1, 1)))
    print(f"{f:5d} {np.degrees(np.arctan2(fwd[1], fwd[0])):13.1f} "
          f"{np.degrees(np.arctan2(left[1], left[0])):13.1f} {ang:7.1f}")
