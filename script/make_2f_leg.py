"""Build a 2F-only path: bottom of stair A -> north landing -> mouth of stair B.

Used to check whether the flat second floor is walkable at all, independently of
any stair descent.

    python script/make_2f_leg.py
"""

import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "script")
from make_multi_floor_switchback import astar, grid_axes   # noqa: E402

START_2F = (1.80, 7.20)     # foot of stair A, on the 2F slab
GOAL_2F = (9.00, 8.60)      # mouth of stair B, on the 2F slab


def main():
    mesh = pickle.load(open("data/mesh/mfsb2.pkl", "rb"))
    names = [b["name"] for b in mesh["blocks"]]
    i2 = names.index("floor_2F")
    g = mesh["heigthmap"][i2]
    nx, ny, xc, yc = grid_axes()

    pts = astar(g, xc, yc, START_2F, GOAL_2F, max_climb=0.28, clearance=1.00)
    length = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum()
    print("2F leg %.2f m, %d waypoints" % (length, len(pts)))
    for p in pts:
        print("   (%6.2f, %6.2f, %5.2f)" % (p[0], p[1], p[2]))

    with open("data/paths/mfsb2_2f.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "mfsb2_2f",
                   "note": "flat 2F only: stair A foot -> north landing -> stair B mouth",
                   "path": [[round(float(v), 3) for v in p] for p in pts]}, fh, indent=2)
    print("written data/paths/mfsb2_2f.json")
    print("START_XY=%.2f,%.2f  GOAL_XY=%.2f,%.2f" % (START_2F + GOAL_2F))


if __name__ == "__main__":
    main()
