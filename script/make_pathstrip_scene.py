"""Path-strip probe: only a 1.6 m wide walkway along the route exists.

Everything off the route is a void, so the heightmap patch always contains the
path edges.  Stair A and its top landing are kept so the descent still works.

    python script/make_pathstrip_scene.py --name strip
"""

import argparse
import json
import os
import pickle

import numpy as np

from make_multi_floor_switchback import (
    NSTEP, PIT, RES, RISER, SLAB_T, TREAD, XA, A_Y0, Mesh, empty, rect_mask,
)
from make_2f_scene import tess_box

H = NSTEP * RISER
Z3, Z2 = 2 * H, H
LAND = (XA[0], XA[1], 0.60, A_Y0)
WIDTH = 1.6


def route_2f():
    """2F part of the route: stair bottom -> north -> arc -> east -> arc -> goal."""
    pts = [(1.80, 6.32), (1.80, 8.60)]
    cx, cy, R = 3.30, 8.60, 1.5
    for a in np.linspace(180, 90, 7)[1:]:
        t = np.radians(a)
        pts.append((cx + R * np.cos(t), cy + R * np.sin(t)))
    pts.append((7.20, 10.10))
    cx, cy, R = 7.20, 8.60, 1.5
    for a in np.linspace(90, 0, 7)[1:]:
        t = np.radians(a)
        pts.append((cx + R * np.cos(t), cy + R * np.sin(t)))
    pts.append((8.70, 8.60))
    pts.append((9.00, 8.20))
    return np.asarray(pts, dtype=float)


def dist_to_polyline(xc, yc, pts):
    """Distance from every grid point to the polyline (metres)."""
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    best = np.full(X.shape, 1e9)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        if L2 < 1e-12:
            continue
        t = ((X - ax) * abx + (Y - ay) * aby) / L2
        t = np.clip(t, 0.0, 1.0)
        px, py = ax + t * abx, ay + t * aby
        d = np.hypot(X - px, Y - py)
        best = np.minimum(best, d)
    return best


def tess_quad(m, p0, p1, p2, p3, z, step=0.15):
    """Tessellate the quad p0-p1-p2-p3 (in order) on plane z."""
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    nu = max(1, int(round(np.linalg.norm(p1 - p0) / step)))
    nv = max(1, int(round(np.linalg.norm(p3 - p0) / step)))
    base = len(m.v)
    for i in range(nu + 1):
        for j in range(nv + 1):
            u, v = i / nu, j / nv
            a = (1 - u) * p0 + u * p1
            b = (1 - u) * p3 + u * p2
            q = (1 - v) * a + v * b
            m.v.append((float(q[0]), float(q[1]), float(z)))
    for i in range(nu):
        for j in range(nv):
            A = base + i * (nv + 1) + j
            B = base + (i + 1) * (nv + 1) + j
            C = base + (i + 1) * (nv + 1) + j + 1
            D = base + i * (nv + 1) + j + 1
            m.f += [(A, B, C), (A, C, D)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="strip")
    args = ap.parse_args()

    x0, x1, y0, y1 = -2.0, 13.0, -1.0, 14.0
    nx = int(round((x1 - x0) / RES))
    ny = int(round((y1 - y0) / RES))
    xc = x0 + (np.arange(nx) + 0.5) * RES
    yc = y0 + (np.arange(ny) + 0.5) * RES
    print("grid %dx%d" % (nx, ny))

    pts = route_2f()
    d = dist_to_polyline(xc, yc, pts)
    strip = empty(nx, ny)
    strip[d <= WIDTH / 2.0] = Z2

    land = empty(nx, ny)
    land[rect_mask(xc, yc, LAND)] = Z3 - RISER

    stair = empty(nx, ny)
    y = A_Y0
    for i in range(NSTEP):
        stair[rect_mask(xc, yc, (XA[0], XA[1], y, y + TREAD))] = Z3 - (i + 1) * RISER
        y += TREAD

    sheets = [("top_land", land), ("A", stair), ("floor_2F", strip)]
    names = [n for n, _ in sheets]
    maps = [m for _, m in sheets]
    print("sheets:", names)
    print("strip width %.2f m, path %d points, %.2f m long"
          % (WIDTH, len(pts), np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))

    # ---- mesh --------------------------------------------------------------
    m = Mesh()
    tess_box(m, LAND[0], LAND[1], LAND[2], LAND[3], Z3 - RISER)
    y = A_Y0
    for i in range(NSTEP):
        tess_box(m, XA[0], XA[1], y, y + TREAD, Z3 - (i + 1) * RISER,
                 thick=0.16, step=0.12)
        y += TREAD
    hw = WIDTH / 2.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        L = np.linalg.norm(seg)
        if L < 1e-9:
            continue
        n = np.array([-seg[1], seg[0]]) / L
        tess_quad(m, a + n * hw, b + n * hw, b - n * hw, a - n * hw, Z2, step=0.15)

    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts %d tris" % (len(verts), len(faces)))

    ok = True
    print("surface query check:")
    for label, (px, py), foot, exp in [
        ("top landing", (1.80, 1.20), Z3 - RISER, Z3 - RISER),
        ("A mid", (1.80, 4.00), Z3 - 6 * RISER, Z3 - 6 * RISER),
        ("A bottom", (1.80, 6.30), Z2, Z2),
        ("strip at bottom", (1.80, 6.45), Z2, Z2),
        ("strip north", (1.80, 8.60), Z2, Z2),
        ("strip east", (5.00, 10.10), Z2, Z2),
        ("strip goal", (9.00, 8.20), Z2, Z2),
        ("off strip", (5.00, 5.00), Z2, PIT),
    ]:
        ix, iy = int((px - x0) / RES), int((py - y0) / RES)
        vals = [float(mm[ix, iy]) for mm in maps]
        usable = [v for v in vals if v <= foot + 0.35]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-17s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
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
    print("written data/mesh/%s.pkl" % args.name)
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
