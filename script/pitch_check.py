"""Quantify the torso pitch during a rollout.

Body order: 0 Pelvis, 9 Torso, 11 Chest, 13 Head.
Pitch is measured in the direction of travel: positive = leaning back
(shoulder behind the pelvis relative to the walking direction).
"""
import pickle
import sys

import numpy as np

BODY = {"pelvis": 0, "torso": 9, "chest": 11, "head": 13}

st = pickle.load(open(sys.argv[1], "rb"))
root = np.asarray(st["root_states"])[:, 0, :13]
bpos = np.asarray(st["rigid_body_pos"])[:, 0]

n = len(root)
step = max(n // 16, 1)
print(f"{'frame':>5} {'x':>6} {'y':>6} {'z':>5} {'pitch_deg':>9} {'tilt_from_vert':>14}")
for f in range(0, n, step):
    pelvis = bpos[f, BODY["pelvis"]]
    chest = bpos[f, BODY["chest"]]
    axis = chest - pelvis
    # travel direction from the next frames
    j = min(f + 5, n - 1)
    d = root[j, :2] - root[f, :2]
    if np.linalg.norm(d) < 1e-4:
        d = np.array([1.0, 0.0])
    d = d / np.linalg.norm(d)
    horiz = axis[0] * d[0] + axis[1] * d[1]
    pitch = np.degrees(np.arctan2(-horiz, axis[2]))  # + = chest behind pelvis
    tilt = np.degrees(np.arccos(np.clip(axis[2] / np.linalg.norm(axis), -1, 1)))
    print(f"{f:5d} {root[f,0]:6.2f} {root[f,1]:6.2f} {root[f,2]:5.2f} "
          f"{pitch:9.1f} {tilt:14.1f}")

pelvis = bpos[:, BODY["pelvis"]]
chest = bpos[:, BODY["chest"]]
axis = chest - pelvis
tilt = np.degrees(np.arccos(np.clip(axis[:, 2] / np.linalg.norm(axis, axis=1), -1, 1)))
print(f"\ntilt from vertical: mean {tilt.mean():.1f}  max {tilt.max():.1f} deg")
print(f"frames with tilt > 20 deg: {(tilt > 20).sum()} / {n}")
