#!/usr/bin/env python3
"""Summarize validation runs: first reset/failure, clearance, lateral drift."""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def height_at(md, x, y):
    hm = np.asarray(md["heigthmap"], dtype=np.float64)
    cam = np.asarray(md["cam_pos"], dtype=np.float64)
    xs, ys = float(md["x_scale"]), float(md["y_scale"])
    xc = (hm.shape[0] / 2.0) / xs
    yc = (hm.shape[1] / 2.0) / ys
    px = int(np.clip(round((x - (cam[0] - xc)) * xs), 0, hm.shape[0] - 1))
    py = int(np.clip(round((y - (cam[1] - yc)) * ys), 0, hm.shape[1] - 1))
    return float(hm[px, py])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment_result/validation")
    ap.add_argument("--mesh", default="data/mesh/wide_stairs.pkl")
    ap.add_argument("--out", default="experiment_result/validation/summary.csv")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)

    rows = []
    for p in sorted(Path(args.root).rglob("states.pkl")):
        try:
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            root = np.asarray(d["root_states"])[:, 0]
            rb = np.asarray(d["rigid_body_pos"])[:, 0]
            z = root[:, 2]
            resets = list((np.where(np.diff(z) > 0.4)[0] + 1).tolist())
            first = resets[0] if resets else -1
            h = np.array([height_at(md, float(x), float(y)) for x, y in root[:, :2]])
            clearance = rb[:, :, 2].min(axis=1) - h
            lateral = np.abs(root[:, 1])
            # first frame with obvious airborne or lateral departure
            airborne = np.where(clearance > 0.05)[0]
            outside = np.where(lateral > 1.2)[0]
            rows.append({
                "tag": p.parts[-4] if len(p.parts) >= 4 else "",
                "mesh": p.parts[-3] if len(p.parts) >= 3 else "",
                "seed": p.parent.name,
                "frames": len(root),
                "first_reset": first,
                "n_resets": len(resets),
                "first_airborne": int(airborne[0]) if len(airborne) else -1,
                "first_outside": int(outside[0]) if len(outside) else -1,
                "min_clearance": round(float(clearance.min()), 4),
                "max_lateral": round(float(lateral.max()), 4),
            })
        except Exception as exc:
            print("skip", p, type(exc).__name__, exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["tag"])
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows), "->", out)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
