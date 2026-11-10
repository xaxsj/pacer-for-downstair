"""Big-floor probe: stair A embedded in a large continuous 2F slab.

Temporary experiment.  The 2F floor is a 28 x 28 m plane, so the staircase has
continuous ground on both sides instead of the deep stair-well pits of the
multi-floor scenes.  No 3F slab, no 1F slab, no stair B.

    top landing  x[0.30,3.30]  y[0.60,2.00]   z 4.60
    stair A      x[0.30,3.30]  y 2.00 -> 6.32  z 4.80 -> 2.40
    big 2F slab  x[-12,16]     y[-12,16]       z 2.40

    python script/make_bigfloor_scene.py --name big2f
"""

import argparse
import os
import pickle

import numpy as np

from make_multi_floor_switchback import (
    NSTEP, PIT, RES, RISER, SLAB_T, TREAD, XA, A_Y0, Mesh, empty, rect_mask,
)
from make_2f_scene import tess_box

H = NSTEP * RISER           # 2.40
Z3, Z2 = 2 * H, H           # 4.80, 2.40
FLOOR = (-12.0, 16.0, -12.0, 16.0)      # x0, x1, y0, y1
LAND = (XA[0], XA[1], 0.60, A_Y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="big2f")
    args = ap.parse_args()

    x0, x1, y0, y1 = FLOOR
    nx = int(round((x1 - x0) / RES))
    ny = int(round((y1 - y0) / RES))
    xc = x0 + (np.arange(nx) + 0.5) * RES
    yc = y0 + (np.arange(ny) + 0.5) * RES
    print("grid %dx%d over %.0fx%.0f m" % (nx, ny, x1 - x0, y1 - y0))

    ya_top = A_Y0
    ya_bot = A_Y0 + NSTEP * TREAD

    land = empty(nx, ny)
    land[rect_mask(xc, yc, LAND)] = Z3 - RISER

    stair = empty(nx, ny)
    y = ya_top
    for i in range(NSTEP):
        stair[rect_mask(xc, yc, (XA[0], XA[1], y, y + TREAD))] = Z3 - (i + 1) * RISER
        y += TREAD

    floor = empty(nx, ny)
    floor[rect_mask(xc, yc, FLOOR)] = Z2

    sheets = [("top_land", land), ("A", stair), ("floor_2F", floor)]
    names = [n for n, _ in sheets]
    maps = [m for _, m in sheets]
    print("sheets:", names)
    print("stair A y %.2f -> %.2f   z %.2f -> %.2f" % (ya_top, ya_bot, Z3, Z2))

    # ---- mesh --------------------------------------------------------------
    m = Mesh()
    tess_box(m, LAND[0], LAND[1], LAND[2], LAND[3], Z3 - RISER)
    y = ya_top
    for i in range(NSTEP):
        tess_box(m, XA[0], XA[1], y, y + TREAD, Z3 - (i + 1) * RISER,
                 thick=0.16, step=0.12)
        y += TREAD
    tess_box(m, FLOOR[0], FLOOR[1], FLOOR[2], FLOOR[3], Z2, thick=0.30, step=0.25)

    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts %d tris  z %.2f..%.2f"
          % (len(verts), len(faces), verts[:, 2].min(), verts[:, 2].max()))

    # ---- query check -------------------------------------------------------
    ok = True
    print("surface query check (RISE=0.35):")
    for label, (px, py), foot, exp in [
        ("top landing", (1.80, 1.20), Z3 - RISER, Z3 - RISER),
        ("top step", (1.80, 2.18), Z3 - RISER, Z3 - RISER),
        ("A mid", (1.80, 4.00), Z3 - 6 * RISER, Z3 - 6 * RISER),
        ("A bottom", (1.80, 6.30), Z2, Z2),
        ("floor after stair", (1.80, 6.60), Z2, Z2),
        ("big floor centre", (5.00, 10.00), Z2, Z2),
        ("goal", (9.00, 8.20), Z2, Z2),
    ]:
        ix, iy = int((px - x0) / RES), int((py - y0) / RES)
        vals = [float(mm[ix, iy]) for mm in maps]
        usable = [v for v in vals if v <= foot + 0.35]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-18s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
              % (label, px, py, foot, got, exp, flag))

    os.makedirs("data/mesh", exist_ok=True)
    stack = np.stack(maps).astype(np.float32)
    walk = (stack > PIT + 0.1).astype(np.int8)
    cam = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float32)
    with open("data/mesh/%s.pkl" % args.name, "wb") as fh:
        pickle.dump(dict(vertices=verts, faces=faces, heigthmap=stack,
                         walkable_map=walk, cam_pos=cam,
                         x_scale=1.0 / RES, y_scale=1.0 / RES,
                         start_block=0, start_z=Z3 - RISER, rise_tol=0.35,
                         start_yaw=1.5707963267948966,
                         blocks=[dict(id=i, name=n,
                                      kind="floor" if n.startswith("floor") else "stair")
                                 for i, n in enumerate(names)]), fh)
    print("written data/mesh/%s.pkl (%d sheets, %.1f MB)"
          % (args.name, len(names),
             os.path.getsize("data/mesh/%s.pkl" % args.name) / 1e6))
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
