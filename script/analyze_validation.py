#!/usr/bin/env python3
"""Summarize validation runs with a corrected terrain/foot mapping."""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

DEFAULT_FEET = "3,4,7,8"


def parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment_result/validation")
    ap.add_argument("--mesh", default="data/mesh/wide_stairs.pkl")
    ap.add_argument("--halfwidth", type=float, default=1.4)
    ap.add_argument("--feet", default=DEFAULT_FEET)
    ap.add_argument("--airborne", type=float, default=0.05)
    ap.add_argument("--sink", type=float, default=-0.02)
    ap.add_argument("--out", default="experiment_result/validation/summary.csv")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    sample = height_sampler(md)
    feet_all = parse_ints(args.feet)

    rows = []
    for p in sorted(Path(args.root).rglob("states.pkl")):
        try:
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            root = np.asarray(d["root_states"])[:, 0]
            rb = np.asarray(d["rigid_body_pos"])[:, 0]
            feet = [i for i in feet_all if i < rb.shape[1]]
            T = root.shape[0]
            z = root[:, 2]
            progress = np.asarray(d.get("progress", np.array([]))).reshape(T, -1)[:, 0] \
                if "progress" in d and np.asarray(d["progress"]).size else np.zeros(T)

            resets = np.where(np.diff(z) > 0.4)[0] + 1
            if progress.size == T:
                resets = np.unique(np.concatenate([resets, np.where(np.diff(progress) < -5)[0] + 1]))
            first_reset = int(resets[0]) if len(resets) else -1

            foot_z = rb[:, feet, 2].min(axis=1)
            h = np.array([sample(float(x), float(y)) for x, y in root[:, :2]])
            clearance = foot_z - h
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
            reason = next((r for r, f in reasons if f == first_failure), "")

            rows.append({
                "tag": p.parts[-4] if len(p.parts) >= 4 else "",
                "mesh": p.parts[-3] if len(p.parts) >= 3 else "",
                "seed": p.parent.name,
                "frames": T,
                "first_failure": first_failure,
                "reason": reason,
                "first_reset": first_reset,
                "n_resets": len(resets),
                "first_airborne": first_air,
                "first_sink": first_sink,
                "first_outside": first_out,
                "min_clearance": round(float(clearance.min()), 4),
                "max_lateral": round(float(lateral.max()), 4),
            })
        except Exception as exc:
            print("skip", p, type(exc).__name__, exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["tag"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows), "->", out)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
