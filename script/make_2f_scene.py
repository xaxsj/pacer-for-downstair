"""Two-storey scene: second floor + one flight down to the first floor.

The third floor is removed entirely (no slab in the heightmap and no slab in the
physics mesh) so that a flat second-floor walk can be tested without anything
overhead.

Layout:

    2F slab z=2.40, U corridor, hole for stair B (x 7.7-10.7, y 1.68-8.0)
    stair B: 12 steps x riser 0.20 x tread 0.36, y 8.00 -> 3.68, z 2.40 -> 0.00
    1F slab z=0.00

    python script/make_2f_scene.py --name flat2
"""

import argparse
import json
import os
import pickle

import numpy as np

from make_multi_floor_switchback import (          # noqa: E402
    EXTENT, NSTEP, OFFSET, PIT, RES, RISER, SLAB_T, TREAD,
    Mesh, U_RECTS, WEST, NORTH, EAST, XB, B_Y0,
    empty, grid_axes, rect_mask, slab_map,
)

H = NSTEP * RISER          # 2.40 m storey
Z2, Z1 = H, 0.0
H_LAND = 2.0


def tess_box(m, x0, x1, y0, y1, z_top, thick=SLAB_T, step=0.10):
    """Box whose top and bottom faces are finely tessellated.

    Isaac Gym maps the default ground texture through vertex positions, so a slab
    made of two huge triangles renders as a smeared uniform surface.  The
    training/eval terrains are built on a fine grid and therefore show the
    speckled texture; tessellating the slabs here reproduces the same look.
    """
    nx = max(1, int(round((x1 - x0) / step)))
    ny = max(1, int(round((y1 - y0) / step)))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    z0 = z_top - thick
    for z in (z_top, z0):
        base = len(m.v)
        for i in range(nx + 1):
            for j in range(ny + 1):
                m.v.append((float(xs[i]), float(ys[j]), float(z)))
        for i in range(nx):
            for j in range(ny):
                a = base + i * (ny + 1) + j
                b = base + (i + 1) * (ny + 1) + j
                c = base + (i + 1) * (ny + 1) + j + 1
                d = base + i * (ny + 1) + j + 1
                if z == z_top:
                    m.f += [(a, b, c), (a, c, d)]
                else:
                    m.f += [(a, c, b), (a, d, c)]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for k in range(4):
        ax, ay = corners[k]
        bx, by = corners[(k + 1) % 4]
        base = len(m.v)
        m.v += [(ax, ay, z0), (bx, by, z0), (bx, by, z_top), (ax, ay, z_top)]
        m.f += [(base, base + 1, base + 2), (base, base + 2, base + 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="flat2")
    ap.add_argument("--single", action="store_true",
                    help="keep only the 2F slab: no stair, no first floor")
    args = ap.parse_args()
    single = args.single

    nx, ny, xc, yc = grid_axes()
    yb_top = B_Y0
    yb_bot = B_Y0 - NSTEP * TREAD
    hole_b = (XB[0], XB[1], yb_bot - H_LAND, yb_top)

    # stair B treads (the only stair in this scene)
    mb = empty(nx, ny)
    y = yb_top
    for i in range(NSTEP):
        mb[rect_mask(xc, yc, (XB[0], XB[1], y - TREAD, y))] = Z2 - (i + 1) * RISER
        y -= TREAD

    if single:
        sheets = [("floor_2F", slab_map(xc, yc, Z2, (hole_b,)))]
    else:
        sheets = [
            ("floor_2F", slab_map(xc, yc, Z2, (hole_b,))),
            ("B", mb),
            ("floor_1F", slab_map(xc, yc, Z1)),
        ]
    names = [n for n, _ in sheets]
    maps = [m for _, m in sheets]
    print("storey H=%.2f  stair B y %.2f -> %.2f  hole %s" % (H, yb_top, yb_bot, hole_b))
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

    for r in subtract(U_RECTS, hole_b):
        tess_box(m, r[0], r[1], r[2], r[3], Z2)
    if not single:
        y = yb_top
        for i in range(NSTEP):
            tess_box(m, XB[0], XB[1], y - TREAD, y, Z2 - (i + 1) * RISER, thick=0.16, step=0.12)
            y -= TREAD
        for r in U_RECTS:
            tess_box(m, r[0], r[1], r[2], r[3], Z1)

    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts %d tris  z %.2f..%.2f"
          % (len(verts), len(faces), verts[:, 2].min(), verts[:, 2].max()))

    # ---- query check -------------------------------------------------------
    ok = True
    print("surface query check (RISE=0.35):")
    for label, (px, py), foot, exp in [
        ("2F slab", (1.80, 7.20), Z2, Z2),
        ("2F north", (5.00, 10.50), Z2, Z2),
        ("2F at B mouth", (9.00, 8.60), Z2, Z2),
        ("B mid", (9.00, 5.90), 1.00, 1.00),
        ("1F under B", (9.00, 5.90), Z1, Z1),
        ("1F goal", (9.00, 1.00), Z1, Z1),
    ]:
        ix, iy = int((px - xc[0]) / RES), int((py - yc[0]) / RES)
        vals = [float(mm[ix, iy]) for mm in maps]
        usable = [v for v in vals if v <= foot + 0.35]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-14s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
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
                         start_block=0, start_z=Z2, rise_tol=0.35,
                         start_yaw=1.5707963267948966,
                         blocks=[dict(id=i, name=n, kind="floor" if n.startswith("floor") else "stair")
                                 for i, n in enumerate(names)]), fh)
    print("written data/mesh/%s.pkl (%d sheets)" % (args.name, len(names)))
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
