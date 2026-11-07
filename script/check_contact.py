#!/usr/bin/env python3
"""Validate terrain contact/first-failure from a PACER states pickle.

Fixes the earlier version:
- sample the height field with cell-center coordinates instead of rounded
  camera offsets;
- use foot/toe rigid bodies, not every body, for clearance;
- compare lateral drift against the actual stair half-width;
- report the first failure frame/reason, not only the 30-second video tail.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

DEFAULT_FEET = "3,4,7,8"  # L_Ankle, L_Toe, R_Ankle, R_Toe in the SMPL XML


def parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def make_height_sampler(md):
    hm = np.asarray(md["heigthmap"], dtype=np.float64)
    cam = np.asarray(md["cam_pos"], dtype=np.float64)
    xs, ys = float(md["x_scale"]), float(md["y_scale"])
    nx, ny = hm.shape
    x_min = float(cam[0]) - (nx / 2.0) / xs
    y_min = float(cam[1]) - (ny / 2.0) / ys

    def sample(x: float, y: float) -> float:
        px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
        py = int(np.clip(np.floor((y - y_min) * ys), 0, ny - 1))
        return float(hm[px, py])

    return sample, (x_min, y_min, nx, ny, xs, ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--halfwidth", type=float, default=0.9,
                    help="stair half-width in metres")
    ap.add_argument("--feet", default=DEFAULT_FEET,
                    help="comma-separated rigid-body indices for feet/toes")
    ap.add_argument("--airborne", type=float, default=0.05)
    ap.add_argument("--sink", type=float, default=-0.02)
    ap.add_argument("--verbose-every", type=int, default=10)
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    height_at, meta = make_height_sampler(md)

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])[:, args.env]
    rb = np.asarray(d["rigid_body_pos"])[:, args.env]
    feet = parse_ints(args.feet)
    feet = [i for i in feet if 0 <= i < rb.shape[1]]
    if not feet:
        raise SystemExit("no valid feet indices")

    T = root.shape[0]
    z = root[:, 2]
    progress = np.asarray(d.get("progress", np.array([]))).reshape(T, -1)[:, 0] \
        if "progress" in d and np.asarray(d["progress"]).size else np.zeros(T)
    resets = np.where(np.diff(z) > 0.4)[0] + 1
    if progress.size == T:
        resets = np.unique(np.concatenate([resets, np.where(np.diff(progress) < -5)[0] + 1]))
    first_reset = int(resets[0]) if len(resets) else -1

    foot_z = rb[:, feet, 2].min(axis=1)
    terrain_h = np.array([height_at(float(x), float(y)) for x, y in root[:, :2]])
    clearance = foot_z - terrain_h
    lateral = np.abs(root[:, 1])

    first_air = int(np.where(clearance > args.airborne)[0][0]) if np.any(clearance > args.airborne) else -1
    first_sink = int(np.where(clearance < args.sink)[0][0]) if np.any(clearance < args.sink) else -1
    first_out = int(np.where(lateral > args.halfwidth)[0][0]) if np.any(lateral > args.halfwidth) else -1

    reasons = []
    if first_reset >= 0:
        reasons.append(("reset", first_reset))
    if first_sink >= 0:
        reasons.append(("sink", first_sink))
    if first_out >= 0:
        reasons.append(("outside", first_out))
    first_failure = min((f for _, f in reasons), default=-1)
    first_reason = next((r for r, f in reasons if f == first_failure), "")

    print("frames", T, "feet", feet, "halfwidth", args.halfwidth)
    print("height grid meta x_min,y_min,nx,ny,xs,ys:", meta)
    for i in range(0, T, max(1, args.verbose_every)):
        flag = ""
        if clearance[i] > args.airborne:
            flag += " airborne"
        if clearance[i] < args.sink:
            flag += " sink"
        if lateral[i] > args.halfwidth:
            flag += " outside"
        print(" %3d root=(%+.2f,%+.2f,%+.2f) h=%.3f foot=%.3f cl=%+.3f |y|=%.3f%s" %
              (i, root[i, 0], root[i, 1], root[i, 2], terrain_h[i], foot_z[i],
               clearance[i], lateral[i], flag))

    print("=== summary ===")
    print("first_reset", first_reset, "resets", resets.tolist()[:10])
    print("first_airborne", first_air, "first_sink", first_sink, "first_outside", first_out)
    print("first_failure", first_failure, "reason", first_reason)
    print("clearance mean/min/max", float(clearance.mean()), float(clearance.min()), float(clearance.max()))
    print("lateral max", float(lateral.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
