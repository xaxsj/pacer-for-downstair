"""Analyse a PACER auto_record dump, truncating the post-fall free-fall tail.

A walk that runs past the terrain border falls into the void: z keeps decreasing
for the rest of the episode (seen as z = -3242 m). Such frames are not gait
failures, so we cut the trace at the first frame where z drops clearly below the
stair-run bottom.

Usage: python script/analyze_eval_states.py <states.pkl> [...]
"""

import os
import pickle
import sys

import numpy as np

FLOOR_SLACK = 0.30     # a real staircase bottom is well above z = -0.3


def heading_of(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.cross(xyz, v)
    d = v + w * t + np.cross(xyz, t)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def analyse(path, env=0):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])[:, env].astype(np.float64)
    dt = float(np.asarray(d.get("dt", 0.0333)).ravel()[0])
    traj = np.asarray(d.get("traj_verts", np.zeros((0, 3))), dtype=np.float64)
    T = root.shape[0]
    z = root[:, 2]
    xy = root[:, 0:2]

    # fall out of the world?
    below = np.where(z < -FLOOR_SLACK)[0]
    fell_out = int(below[0]) if below.size else None
    n = fell_out if fell_out is not None else T

    # genuine re-rise (a fall + respawn) inside the valid window
    jumps = [int(f) for f in (np.where(np.diff(z[:n]) > 0.4)[0] + 1) if f != n - 1]

    seg = root[:n]
    travel = float(np.linalg.norm(np.diff(seg[:, 0:2], axis=0), axis=1).sum()) if n > 1 else 0.0
    path_len = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()) if len(traj) > 1 else 0.0

    return {
        "file": os.path.basename(path),
        "frames": T,
        "seconds": round((T - 1) * dt, 1),
        "valid_frames": n,
        "valid_seconds": round(max(n - 1, 0) * dt, 1),
        "fell_out_of_world": fell_out,
        "falls_during_walk": len(jumps),
        "start_xyz": tuple(np.round(root[0, 0:3], 2)),
        "end_xyz_valid": tuple(np.round(root[max(n - 1, 0), 0:3], 2)),
        "descended_m": round(float(z[0] - z[:n].min()), 2),
        "travel_m": round(travel, 2),
        "path_len_m": round(path_len, 2),
        "path_progress_pct": round(100.0 * travel / path_len, 1) if path_len else None,
        "lateral_max_m": round(float(np.abs(root[:n, 1] - root[0, 1]).max()), 2),
        "heading_start": round(heading_of(root[0, 3:7]), 1),
        "heading_end_valid": round(heading_of(root[max(n - 1, 0), 3:7]), 1),
        "z_min_valid": round(float(z[:n].min()), 2),
    }


def trace(path, env=0, stride=30):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])[:, env].astype(np.float64)
    T = root.shape[0]
    print("  帧     x       y       z      航向")
    for f in list(range(0, min(T, 210), stride)) + list(range(300, T, 150)):
        print("  %-5d %7.2f %7.2f %8.2f %7.1f" % (
            f, root[f, 0], root[f, 1], root[f, 2], heading_of(root[f, 3:7])))


def main():
    for p in sys.argv[1:]:
        m = analyse(p)
        print("=" * 64)
        print(m["file"])
        print("=" * 64)
        print("  时长            %.1f s（有效 %.1f s / %d 帧）" % (
            m["seconds"], m["valid_seconds"], m["valid_frames"]))
        print("  走完下降        %.2f m" % m["descended_m"])
        print("  行走距离        %.2f m / 路径 %.2f m（%.0f%%）" % (
            m["travel_m"], m["path_len_m"], m["path_progress_pct"] or 0))
        print("  横向漂移        max|dy| = %.2f m" % m["lateral_max_m"])
        print("  行走中摔倒      %d 次" % m["falls_during_walk"])
        print("  走出地形坠落    %s" % (
            "第 %d 帧（%.1f s）" % (m["fell_out_of_world"], m["fell_out_of_world"] * 0.0333)
            if m["fell_out_of_world"] is not None else "没有"))
        print("  航向            起 %.1f -> 末 %.1f deg" % (
            m["heading_start"], m["heading_end_valid"]))
        print()
        trace(p)
        print()


if __name__ == "__main__":
    main()
