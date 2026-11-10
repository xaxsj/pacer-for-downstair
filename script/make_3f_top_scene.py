"""3F + stair A + 2F scene (no first floor, no stair B).

Used to isolate "a slab overhead" from "a slab below": the 3F slab sits 2.4 m
above the second floor, the 1F slab is absent.

    sheets: floor_3F, A, floor_2F
    stair A: x[0.30,3.30]  y 2.00 -> 6.32 (+y)   z 4.80 -> 2.40
    3F slab hole: x[0.30,3.30] y[2.00,8.32]   (2 m run-out past the flight bottom)
    2F slab: solid

    python script/make_3f_top_scene.py --name top3
"""

import argparse
import json
import os
import pickle

import numpy as np

from make_multi_floor_switchback import (          # noqa: E402
    EXTENT, NSTEP, OFFSET, PIT, RES, RISER, SLAB_T, TREAD,
    Mesh, U_RECTS, XA, A_Y0,
    empty, grid_axes, rect_mask, slab_map,
)
from make_2f_scene import tess_box               # noqa: E402

H = NSTEP * RISER          # 2.40
Z3, Z2 = 2 * H, H          # 4.80, 2.40
RUNOUT = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="top3")
    ap.add_argument("--no-top", action="store_true",
                    help="drop the 3F slab but keep stair A; start on its top step")
    args = ap.parse_args()
    no_top = args.no_top
    z_start = Z3 - RISER if no_top else Z3

    nx, ny, xc, yc = grid_axes()
    ya_top = A_Y0
    ya_bot = A_Y0 + NSTEP * TREAD
    hole_a = (XA[0], XA[1], ya_top, ya_bot + RUNOUT)

    # stair A treads
    ma = empty(nx, ny)
    y = ya_top
    for i in range(NSTEP):
        ma[rect_mask(xc, yc, (XA[0], XA[1], y, y + TREAD))] = Z3 - (i + 1) * RISER
        y += TREAD

    top_land = (XA[0], XA[1], 0.60, ya_top)   # small landing flush with the top step
    if no_top:
        m_land = empty(nx, ny)
        m_land[rect_mask(xc, yc, top_land)] = Z3 - RISER
        sheets = [
            ("top_land", m_land),
            ("A", ma),
            ("floor_2F", slab_map(xc, yc, Z2)),
        ]
    else:
        sheets = [
            ("floor_3F", slab_map(xc, yc, Z3, (hole_a,))),
            ("A", ma),
            ("floor_2F", slab_map(xc, yc, Z2)),
        ]
    names = [n for n, _ in sheets]
    maps = [m for _, m in sheets]
    print("storey H=%.2f  stair A y %.2f -> %.2f  hole %s" % (H, ya_top, ya_bot, hole_a))
    print("sheets:", names)

    # ---- physics mesh ------------------------------------------------------
    m = Mesh()

    def subtract(rects, hole):
        hx0, hx1, hy0, hy1 = hole
        out = []
        for r in rects:
            ix0, ix1 = max(r[0], hx0), min(r[1], hx1)
            iy0, iy1 = max(r[2], hy0), min(r[3], hy1)
            if ix0 < ix1 and iy0 < iy1:
                if r[2] < iy0:
                    out.append((r[0], r[1], r[2], iy0))
                if iy1 < r[3]:
                    out.append((r[0], r[1], iy1, r[3]))
                if r[0] < ix0:
                    out.append((r[0], ix0, iy0, iy1))
                if ix1 < r[1]:
                    out.append((ix1, r[1], iy0, iy1))
            else:
                out.append(r)
        return out

    if not no_top:
        for r in subtract(U_RECTS, hole_a):
            tess_box(m, r[0], r[1], r[2], r[3], Z3)
    else:
        tess_box(m, top_land[0], top_land[1], top_land[2], top_land[3], Z3 - RISER)
    y = ya_top
    for i in range(NSTEP):
        tess_box(m, XA[0], XA[1], y, y + TREAD, Z3 - (i + 1) * RISER,
                 thick=0.16, step=0.12)
        y += TREAD
    for r in U_RECTS:
        tess_box(m, r[0], r[1], r[2], r[3], Z2)

    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts %d tris  z %.2f..%.2f"
          % (len(verts), len(faces), verts[:, 2].min(), verts[:, 2].max()))

    # ---- query check -------------------------------------------------------
    ok = True
    print("surface query check (RISE=0.35):")
    for label, (px, py), foot, exp in [
        ("top step", (1.80, 2.18), Z3 - RISER, Z3 - RISER) if no_top else ("3F slab", (1.80, 1.20), Z3, Z3),
        ("A mid", (1.80, 4.00), Z3 - 6 * RISER, Z3 - 6 * RISER),
        ("A bottom", (1.80, 6.30), Z2, Z2),
        ("2F after stair", (1.80, 7.00), Z2, Z2),
        ("2F north", (5.00, 10.50), Z2, Z2),
        ("2F at goal", (8.70, 8.30), Z2, Z2),
    ]:
        ix, iy = int((px - xc[0]) / RES), int((py - yc[0]) / RES)
        vals = [float(mm[ix, iy]) for mm in maps]
        usable = [v for v in vals if v <= foot + 0.35]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-16s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
              % (label, px, py, foot, got, exp, flag))

    os.makedirs("data/mesh", exist_ok=True)
    stack = np.stack(maps).astype(np.float32)
    walk = (stack > PIT + 0.1).astype(np.int8)
    cam = np.array([OFFSET[0] + EXTENT[0] / 2.0, OFFSET[1] + EXTENT[1] / 2.0],
                   dtype=np.float32)
    with open("data/mesh/%s.pkl" % args.name, "wb") as fh:
        pickle.dump(dict(vertices=verts, faces=faces, heigthmap=stack,
                         walkable_map=walk, cam_pos=cam,
                         x_scale=1.0 / RES, y_scale=1.0 / RES,
                         start_block=0, start_z=z_start, rise_tol=0.35,
                         start_yaw=1.5707963267948966,
                         blocks=[dict(id=i, name=n,
                                      kind="floor" if n.startswith("floor") else "stair")
                                 for i, n in enumerate(names)]), fh)
    print("written data/mesh/%s.pkl (%d sheets)" % (args.name, len(names)))
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
