#!/usr/bin/env python3
"""Supplementary riser-hook / swing-clearance detector.

Heightfield clearance alone cannot see a vertical riser. This script probes
forward from each toe, finds the first rising step edge, and reports:
- distance from toe to the riser;
- toe bottom relative to the riser top;
- forward speed and root speed;
- first hook / low-clearance / stall frame.

It is diagnostic only; it does not change the policy or reward.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

FEET = [(3, "L_Ankle", -0.0487), (4, "L_Toe", 0.0070),
        (7, "R_Ankle", -0.0487), (8, "R_Toe", 0.0070)]


def height_sampler(md):
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


def quat_forward(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.cross(xyz, v)
    d = v + w * t + np.cross(xyz, t)
    return np.array([float(d[0]), float(d[1])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--window", type=int, default=70)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    h = height_sampler(md)
    with open(args.states, "rb") as fh:
        d = pickle.load(fh)

    root = np.asarray(d["root_states"])[:, 0]
    rb = np.asarray(d["rigid_body_pos"])[:, 0]
    T = root.shape[0]
    dt = 1.0 / 30.0
    rxy = root[:, :2]
    rvel = np.zeros_like(rxy)
    rvel[1:] = np.diff(rxy, axis=0) / dt
    rvel[0] = rvel[1]
    rheading = np.array([quat_forward(root[i, 3:7]) for i in range(T)])

    feet = []
    for idx, name, off in FEET:
        xy = rb[:, idx, :2]
        vel = np.zeros_like(xy)
        vel[1:] = np.diff(xy, axis=0) / dt
        vel[0] = vel[1]
        bottom = rb[:, idx, 2] + off
        feet.append({"idx": idx, "name": name, "xy": xy, "vel": vel, "bottom": bottom})

    # For each frame, find the first rising edge ahead of each toe/ankle.
    rows = []
    for i in range(T):
        speed = float(np.linalg.norm(rvel[i]))
        if speed > 0.05:
            direction = rvel[i] / speed
        else:
            direction = rheading[i]
        for f in feet:
            base_h = h(float(f["xy"][i, 0]), float(f["xy"][i, 1]))
            riser_dist = -1.0
            riser_top = base_h
            for dstep in np.arange(0.0, 0.31, 0.01):
                q = f["xy"][i] + direction * dstep
                hh = h(float(q[0]), float(q[1]))
                if hh > base_h + 0.03:
                    riser_dist = float(dstep)
                    riser_top = float(hh)
                    break
            gap = float(f["bottom"][i] - riser_top)
            along = float(np.dot(f["vel"][i], direction))
            root_along = float(np.dot(rvel[i], direction))
            low = (riser_dist >= 0.0 and riser_dist < 0.12 and gap < 0.03)
            hook = (riser_dist >= 0.0 and riser_dist < 0.08 and gap < -0.02
                    and along < 0.05 and root_along > 0.15)
            stall = (riser_dist >= 0.0 and riser_dist < 0.08 and along < 0.03
                     and root_along > 0.15 and gap < 0.05)
            if low or hook or stall:
                rows.append({
                    "frame": i, "foot": f["name"], "riser_dist": round(riser_dist, 3),
                    "riser_top": round(riser_top, 3), "toe_bottom": round(float(f["bottom"][i]), 3),
                    "gap": round(gap, 3), "foot_along": round(along, 3),
                    "root_along": round(root_along, 3), "hook": int(hook),
                    "stall": int(stall),
                })

    print("frames", T, "events", len(rows))
    for r in rows[:args.window]:
        print(r)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["frame"])
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
