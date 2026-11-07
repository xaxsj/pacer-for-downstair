#!/usr/bin/env python3
"""Write the centerline path for the mixed stair scene.

The path is a polyline of [x, y, z] points. PACER only uses XY; z is for
visualization/debugging (see pacer/env/util/fixed_path.py).

Path shape (world frame):
  start (0.5, 0) -> up stairs -> top flat -> 90 deg turn -> down stairs -> bottom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/paths/mixed_stairs.json")
    ap.add_argument("--riser", type=float, default=0.15)
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    top_z = args.steps * args.riser
    down_y = 4.0 + args.steps * args.tread

    # z values are for visualization only; XY is what PACER tracks.
    path = [
        [0.50, 0.0, 0.0],          # start on the approach flat
        [2.00, 0.0, 0.0],          # foot of the up stairs
        [2.00 + args.steps * args.tread, 0.0, top_z],  # top of the up stairs
        [7.60, 0.0, top_z],        # top flat
        [8.20, 0.60, top_z],       # rounded 90 deg turn
        [8.20, 4.00, top_z],       # head of the down stairs
        [8.20, down_y, 0.0],       # bottom of the down stairs
        [8.20, 8.60, 0.0],         # goal
    ]

    payload = {
        "name": "mixed_stairs",
        "note": "approach -> up 10 -> top flat -> 90 turn -> down 10 -> bottom",
        "path": path,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    length = 0.0
    for a, b in zip(path[:-1], path[1:]):
        length += ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    print("[mixed_stairs] wrote", out)
    print("  points      %d" % len(path))
    print("  2D length   %.2f m" % length)
    print("  implied v   %.2f m/s over 30 s episode" % (length / 30.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
