#!/usr/bin/env python3
"""Detailed first-failure diagnosis for one validation states.pkl.

Prints per-foot box-bottom clearance around the first failure and labels the
lowest foot as swing/stance from the other foot's contact state.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

FEET = [
    (3, "L_Ankle", -0.0487),
    (4, "L_Toe", +0.0070),
    (7, "R_Ankle", -0.0487),
    (8, "R_Toe", +0.0070),
]


def sampler(md):
    hm = np.asarray(md["heigthmap"], dtype=np.float64)
    cam = np.asarray(md["cam_pos"], dtype=np.float64)
    xs, ys = float(md["x_scale"]), float(md["y_scale"])
    nx, ny = hm.shape
    x_min = float(cam[0]) - (nx / 2.0) / xs
    y_min = float(cam[1]) - (ny / 2.0) / ys

    def sample(x, y):
        px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
        py = int(np.clip(np.floor((y - y_min) * ys), 0, ny - 1))
        return float(hm[px, py])

    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--halfwidth", type=float, default=0.9)
    ap.add_argument("--window", type=int, default=45)
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    height = sampler(md)
    with open(args.states, "rb") as fh:
        d = pickle.load(fh)

    root = np.asarray(d["root_states"])[:, 0]
    rb = np.asarray(d["rigid_body_pos"])[:, 0]
    T = root.shape[0]
    z = root[:, 2]
    progress = np.asarray(d.get("progress", np.array([]))).reshape(T, -1)[:, 0] \
        if "progress" in d and np.asarray(d["progress"]).size else np.zeros(T)
    resets = np.where(np.diff(z) > 0.4)[0] + 1
    if progress.size == T:
        resets = np.unique(np.concatenate([resets, np.where(np.diff(progress) < -5)[0] + 1]))
    first_reset = int(resets[0]) if len(resets) else -1

    foot = []
    for idx, name, off in FEET:
        x, y = rb[:, idx, 0], rb[:, idx, 1]
        zb = rb[:, idx, 2] + off
        h = np.array([height(float(a), float(b)) for a, b in zip(x, y)])
        foot.append({"idx": idx, "name": name, "x": x, "y": y, "zb": zb, "h": h, "cl": zb - h})
    cl = np.stack([f["cl"] for f in foot], axis=1)
    lowest = cl.argmin(axis=1)
    first_sink = int(np.where(cl.min(axis=1) < -0.02)[0][0]) if np.any(cl.min(axis=1) < -0.02) else -1
    first_air = int(np.where(cl.min(axis=1) > 0.05)[0][0]) if np.any(cl.min(axis=1) > 0.05) else -1

    print("frames", T, "first_reset", first_reset, "first_sink", first_sink, "first_airborne", first_air)
    print("frame root_xyz progress | foot: name (x,y,bottom,h,cl)")
    events = [x for x in (first_sink, first_air, first_reset) if x >= 0]
    first_event = min(events) if events else 0
    print("first_event", first_event)
    lo = max(0, first_event - 12)
    hi = min(args.window, first_event + 15)
    for i in range(lo, hi):
        fi = int(lowest[i])
        f = foot[fi]
        print("%3d root=(%+.2f,%+.2f,%+.2f) prog=%.2f | low=%s x=%+.2f y=%+.2f zb=%+.3f h=%.3f cl=%+.3f" %
              (i, root[i, 0], root[i, 1], root[i, 2], progress[i], f["name"],
               f["x"][i], f["y"][i], f["zb"][i], f["h"][i], f["cl"][i]))

    if first_sink >= 0:
        i = first_sink
        fi = int(lowest[i])
        other = [j for j in range(4) if j != fi]
        print("\n=== first sink ===")
        print("frame", i, "lowest", foot[fi]["name"], "cl", float(foot[fi]["cl"][i]))
        for j in other:
            print(" %-8s cl=%+.3f x=%+.2f y=%+.2f h=%.3f" %
                  (foot[j]["name"], foot[j]["cl"][i], foot[j]["x"][i], foot[j]["y"][i], foot[j]["h"][i]))
        other_min = float(min(foot[j]["cl"][i] for j in other))
        if other_min > 0.02:
            print("other foot clearance >2cm -> both feet off/airborne at first sink")
        elif other_min > -0.02:
            print("other foot is near contact -> lowest foot is the swing foot")
        else:
            print("other foot also below surface -> check stance foot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
