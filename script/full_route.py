"""Full-route report: max path progress (not just survival) per rollout.

Usage: python script/full_route.py <tag>
"""
import json
import os
import pickle
import sys

import numpy as np

PATH_JSON = "data/paths/mixed_stairs.json"
SEGMENTS = ["approach", "up10", "top-flat", "90deg-turn", "down10", "bottom"]


def load_path():
    with open(PATH_JSON) as fh:
        return np.array(json.load(fh)["path"], dtype=float)[:, :2]


def arc_lengths(path):
    seg = path[1:] - path[:-1]
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])


def project(points, path):
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = arc_lengths(path)
    out = np.zeros(len(points))
    for i, p in enumerate(points):
        t = np.clip(((p - path[:-1]) * seg).sum(axis=1) / (seg_len ** 2), 0, 1)
        proj = path[:-1] + seg * t[:, None]
        k = int(np.argmin(np.linalg.norm(proj - p, axis=1)))
        out[i] = cum[k] + t[k] * seg_len[k]
    return out, cum


def tilt_deg(q):
    up = np.array([2 * (q[0] * q[2] + q[3] * q[1]),
                   2 * (q[1] * q[2] - q[3] * q[0]),
                   1 - 2 * (q[0] ** 2 + q[1] ** 2)])
    return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))


def main(tag):
    path = load_path()
    cum = arc_lengths(path)
    total = cum[-1]
    print(f"route length = {total:.2f} m;  segment boundaries (m): "
          + ", ".join(f"{SEGMENTS[i]} {cum[i]:.2f}-{cum[i + 1]:.2f}"
                      for i in range(len(SEGMENTS))))
    print()
    base = f"experiment_result/validation/{tag}"
    print("mesh          seed  resets  first_fall  max_arc  max_%   reached_where")
    for mesh in ("wide_stairs", "narrow_stairs"):
        for seed in range(3):
            p = f"{base}/{mesh}/seed_{seed}/states.pkl"
            if not os.path.exists(p):
                continue
            d = pickle.load(open(p, "rb"))
            rs = np.asarray(d["root_states"])[:, 0, :13]
            rot = np.asarray(d["rigid_body_rot"])[:, 0]
            prog = np.asarray(d["progress"])[:, 0]
            dt = float(d["dt"])
            arc, _ = project(rs[:, :2], path)
            resets = int((np.diff(prog) < 0).sum())
            tilts = np.array([tilt_deg(q) for q in rot[:, 0]])
            fell = np.where(tilts > 45)[0]
            first_fall = f"{fell[0] * dt:.2f}s" if len(fell) else "-"
            k = int(np.argmax(arc))
            frac = 100 * arc[k] / total
            seg = "END" if arc[k] >= total - 0.35 else next(
                (SEGMENTS[i] for i in range(len(SEGMENTS)) if arc[k] < cum[i + 1]),
                "END")
            print(f"{mesh:13s} {seed}    {resets:4d}   {first_fall:9s}   "
                  f"{arc[k]:5.2f}   {frac:5.1f}   {seg}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fullroute")
