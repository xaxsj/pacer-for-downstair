#!/usr/bin/env python3
"""Build the mixed stair scene used by PACER's MeshTerrain.

Layout (world frame, meters, +x forward, +y left, +z up):
  approach flat      x [0.0, 2.0]    y [-0.9, 0.9]   z = 0
  up stairs          x [2.0, 5.0]    y [-0.9, 0.9]   z 0 -> 1.5   (10 x 0.15/0.30)
  top flat           x [5.0, 8.2]    y [-0.9, 0.9]   z = 1.5
  turn flat          x [7.3, 9.1]    y [-0.9, 4.0]   z = 1.5
  down stairs        x [7.3, 9.1]    y [4.0, 7.0]    z 1.5 -> 0   (10 x 0.15/0.30)
  bottom flat        x [7.3, 9.1]    y [7.0, 9.0]    z = 0

Output pickle matches MeshTerrain:
  vertices (V,3) float32, faces (F,3) int32,
  heigthmap (W,H) float32, walkable_map (W,H) int8,
  cam_pos (2,) float32, x_scale, y_scale
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def build_profile(riser, tread, steps):
    return {
        "x_start": 2.0,
        "x_top": 2.0 + steps * tread,
        "x_flat_end": 8.2,
        "turn_x0": 7.3,
        "turn_x1": 9.1,
        "turn_y0": -0.9,
        "turn_y1": 4.0,
        "down_y1": 4.0 + steps * tread,
        "bottom_y1": 9.0,
        "half_w": 0.9,
        "riser": riser,
        "tread": tread,
        "steps": steps,
        "top_z": steps * riser,
    }


def z_at(x, y, p):
    hw = p["half_w"]
    top_z = p["top_z"]

    # approach
    if 0.0 <= x <= 2.0 and -hw <= y <= hw:
        return 0.0

    # up stairs
    if p["x_start"] <= x <= p["x_top"] and -hw <= y <= hw:
        i = int(np.floor((x - p["x_start"]) / p["tread"]))
        i = max(0, min(i, p["steps"] - 1))
        return p["riser"] * (i + 1)

    # top flat
    if p["x_top"] <= x <= p["x_flat_end"] and -hw <= y <= hw:
        return top_z

    # turn + upper corridor
    if p["turn_x0"] <= x <= p["turn_x1"] and p["turn_y0"] <= y <= p["turn_y1"]:
        return top_z

    # down stairs
    if p["turn_x0"] <= x <= p["turn_x1"] and p["down_y1"] >= y >= 4.0:
        i = int(np.floor((y - 4.0) / p["tread"]))
        i = max(0, min(i, p["steps"] - 1))
        return top_z - p["riser"] * (i + 1)

    # bottom flat
    if p["turn_x0"] <= x <= p["turn_x1"] and p["bottom_y1"] >= y >= p["down_y1"]:
        return 0.0

    return np.nan


def build_grid(p, res, x_min=0.0, x_max=9.4, y_min=-1.0, y_max=9.2):
    nx = int(round((x_max - x_min) / res))
    ny = int(round((y_max - y_min) / res))
    xs = x_min + (np.arange(nx) + 0.5) * res
    ys = y_min + (np.arange(ny) + 0.5) * res
    Z = np.full((nx, ny), np.nan, dtype=np.float64)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[i, j] = z_at(float(x), float(y), p)
    mask = ~np.isnan(Z)
    return xs, ys, Z, mask


def build_mesh(xs, ys, Z, mask, res):
    """Top quads plus vertical walls wherever the height drops."""
    verts = []
    faces = []

    def add_quad(a, b, c, d):
        base = len(verts)
        verts.extend([a, b, c, d])
        faces.append((base + 0, base + 1, base + 2))
        faces.append((base + 0, base + 2, base + 3))

    nx, ny = Z.shape
    for i in range(nx):
        x0 = xs[i] - 0.5 * res
        x1 = xs[i] + 0.5 * res
        for j in range(ny):
            if not mask[i, j]:
                continue
            y0 = ys[j] - 0.5 * res
            y1 = ys[j] + 0.5 * res
            z = float(Z[i, j])

            # top
            add_quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))

            # four neighbours: wall if lower (or outside)
            nbrs = (
                (i - 1, j, "x0", (x0, y0), (x0, y1)),
                (i + 1, j, "x1", (x1, y0), (x1, y1)),
                (i, j - 1, "y0", (x0, y0), (x1, y0)),
                (i, j + 1, "y1", (x0, y1), (x1, y1)),
            )
            for ni, nj, _side, q0, q1 in nbrs:
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    nz = float(Z[ni, nj])
                else:
                    nz = 0.0
                if nz < z - 1e-9:
                    add_quad((q0[0], q0[1], nz), (q1[0], q1[1], nz),
                             (q1[0], q1[1], z), (q0[0], q0[1], z))

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mesh/mixed_stairs.pkl")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--riser", type=float, default=0.15)
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--half_width", type=float, default=0.9)
    args = ap.parse_args()

    p = build_profile(args.riser, args.tread, args.steps)
    p["half_w"] = args.half_width
    xs, ys, Z, mask = build_grid(p, args.res)
    verts, faces = build_mesh(xs, ys, Z, mask, args.res)

    nx, ny = Z.shape
    cam_pos = np.array([(0.0 + 9.4) / 2.0, (-1.0 + 9.2) / 2.0], dtype=np.float32)
    heights = np.where(mask, Z, 0.0).astype(np.float32)

    payload = {
        "vertices": verts,
        "faces": faces,
        "heigthmap": heights,
        "walkable_map": mask.astype(np.int8),
        "cam_pos": cam_pos,
        "x_scale": float(1.0 / args.res),
        "y_scale": float(1.0 / args.res),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)

    print("[stair_width] wrote", out)
    print("  grid        %d x %d  @ %.2f m" % (nx, ny, args.res))
    print("  vertices    %d" % len(verts))
    print("  faces       %d" % len(faces))
    print("  walkable    %d cells" % int(mask.sum()))
    print("  z range     %.2f .. %.2f m" % (float(np.nanmin(Z)), float(np.nanmax(Z))))
    print("  cam_pos     %s" % cam_pos.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
