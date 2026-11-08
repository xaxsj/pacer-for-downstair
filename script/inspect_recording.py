"""Analyse a single-env recording: resets, torso tilt, climb (render check).

Usage: python script/inspect_recording.py <states.pkl>
"""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"], dtype=np.float64)[:, 0, :13]
rot = np.asarray(d["rigid_body_rot"])[:, 0]
cf = np.asarray(d["contact_forces"], dtype=float)[:, 0]
prog = np.asarray(d["progress"])[:, 0]
dt = float(d["dt"])
names = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee",
         "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
         "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
         "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
feet = set(np.asarray(d["contact_body_ids"]).ravel().tolist())


def tilt(q):
    up = np.array([2 * (q[0] * q[2] + q[3] * q[1]),
                   2 * (q[1] * q[2] - q[3] * q[0]),
                   1 - 2 * (q[0] ** 2 + q[1] ** 2)])
    return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))


resets = np.where(np.diff(prog) < 0)[0]
print(f"frames={len(rs)} ({len(rs) * dt:.1f}s)  resets={len(resets)}")
print("reset  time    rise_before  tilt_before  trigger")
for r in resets:
    j = r
    rise = float(rs[:j + 1, 2].max() - rs[0, 2])
    mag = np.linalg.norm(cf[j], axis=-1)
    hit = [(names[b], int(mag[b])) for b in range(24)
           if b not in feet and mag[b] > 50]
    print(f"  {r:4d}  {j * dt:5.2f}s   {rise:+6.2f} m   {tilt(rot[j, 0]):5.1f}      {hit}")
print(f"max rise overall: {rs[:, 2].max() - rs[0, 2]:+.2f} m")
print("tilt over time (max):", round(max(tilt(q) for q in rot[:, 0]), 1), "deg")
