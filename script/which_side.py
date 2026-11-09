"""Which side is the camera really on?  Uses the anatomical left-right axis.

left_axis = L_Hip - R_Hip (world XY).  The dot product of (camera - root) with
that axis is positive only if the camera is on the robot's left side.
"""
import json
import pickle
import sys

import numpy as np


def quat_rotate(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def path_side(poly, xy):
    a, b = poly[:-1, :2], poly[1:, :2]
    ab = b - a
    seg = np.linalg.norm(ab, axis=1)
    keep = seg > 1e-6
    a, ab, seg = a[keep], ab[keep], seg[keep]
    t = np.clip(((xy - a) * ab).sum(axis=1) / (seg ** 2), 0.0, 1.0)
    proj = a + ab * t[:, None]
    i = int(np.argmin(np.linalg.norm(xy - proj, axis=1)))
    dx, dy = ab[i] / seg[i]
    return np.array([-dy, dx])


st = pickle.load(open(sys.argv[1], "rb"))
path = np.asarray(json.load(open(sys.argv[2]))["path"], dtype=float)
root = np.asarray(st["root_states"])[:, 0, :13]
bpos = np.asarray(st["rigid_body_pos"])[:, 0]        # (frames, bodies, 3)

print("body_names check: assuming L_Hip=1, R_Hip=5")
for f in range(0, len(root), max(len(root) // 12, 1)):
    xy = root[f, :2]
    left_axis = bpos[f, 1, :2] - bpos[f, 5, :2]
    n = np.linalg.norm(left_axis)
    if n < 1e-6:
        continue
    left_axis /= n
    body_side = quat_rotate(root[f, 3:7], np.array([0.0, 1.0, 0.0]))[:2]
    body_side /= max(np.linalg.norm(body_side), 1e-6)
    pside = path_side(path, xy)
    print(f"f{f:4d}  dot(anat_left, body+Y)={left_axis @ body_side:+.2f}"
          f"   dot(anat_left, path_left)={left_axis @ pside:+.2f}")
