"""Three-storey building with switchback flights of 6 steps each.

The controller keeps its balance for roughly six consecutive down steps (measured:
a 12-step flight collapses at step 6-7, an 8-step flight just barely passes), so
every flight here is 6 steps.  A storey is therefore two flights with a landing:

    storey H = 2 * 6 * 0.15 = 1.8 m

Layout (plan view, corridor width C = 3.4):

        y
        ^
     12 +-----------------------------+
        |        north landing         |
    8.6 +--------+----------+---------+
        |  west  |          |  east   |
        |  corr. |          |  corr.  |
      0 +--------+----------+---------+
        0      3.4 4.0     7.4      10.8  x

    stair A (3F -> 2F), west corridor, x[0.20,3.20]:
        flight 1  y 2.00 -> 4.16   z 3.60 -> 2.70
        landing   y 4.16 -> 5.66   z 2.70
        flight 2  y 5.66 -> 7.82   z 2.70 -> 1.80
    stair B (2F -> 1F), east corridor, x[7.60,10.40]:
        flight 1  y 8.00 -> 5.84   z 1.80 -> 0.90
        landing   y 5.84 -> 4.34   z 0.90
        flight 2  y 4.34 -> 2.18   z 0.90 -> 0.00

Nine surface sheets in route order:
    floor_3F, A1, A_land, A2, floor_2F, B1, B_land, B2, floor_1F

The engine queries `max{ sheet <= foot + rise_tol }` per cell, so no sheet has to
describe another sheet's geometry and there is no block id anywhere.

    python script/make_multi_floor_switchback.py --name mfsb
"""

import argparse
import json
import os
import pickle

import numpy as np

RES = 0.05
PIT = -1.5
SLAB_T = 0.20

RISER = 0.20
TREAD = 0.36
NSTEP = 12              # continuous flight, no landing
H = NSTEP * RISER       # storey height
C = 3.4
LAND = 0.0              # no mid landing: one continuous flight per storey

# -------- layout ------------------------------------------------------------
WEST = (0.0, 3.6, 0.0, 14.0)
NORTH = (0.0, 11.0, 9.6, 14.0)
EAST = (7.4, 11.0, 0.0, 14.0)
U_RECTS = [WEST, NORTH, EAST]

XA = (0.30, 3.30)       # stair A lanes (west corridor)
XB = (7.70, 10.70)      # stair B lanes (east corridor)

A_Y0 = 2.00
B_Y0 = 8.00

START = (1.80, 1.20)
GOAL = (9.20, 1.00)

OFFSET = (-2.0, -2.0)
EXTENT = (15.0, 16.0)


class Mesh:
    def __init__(self):
        self.v, self.f = [], []

    def box(self, x0, x1, y0, y1, z_top, thick=SLAB_T):
        z0, z1 = z_top - thick, z_top
        b = len(self.v)
        self.v += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for a, b2, c, d in [(4, 5, 6, 7), (0, 1, 2, 3), (0, 1, 5, 4),
                            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            self.f += [(b + a, b + b2, b + c), (b + a, b + c, b + d)]


def grid_axes():
    nx = int(round(EXTENT[0] / RES))
    ny = int(round(EXTENT[1] / RES))
    xc = OFFSET[0] + (np.arange(nx) + 0.5) * RES
    yc = OFFSET[1] + (np.arange(ny) + 0.5) * RES
    return nx, ny, xc, yc


def rect_mask(xc, yc, r):
    return ((xc[:, None] >= r[0]) & (xc[:, None] <= r[1])
            & (yc[None, :] >= r[2]) & (yc[None, :] <= r[3]))


def empty(nx, ny):
    return np.full((nx, ny), PIT, dtype=np.float32)


def slab_map(xc, yc, z, holes=()):
    m = empty(len(xc), len(yc))
    for r in U_RECTS:
        m[rect_mask(xc, yc, r)] = z
    for h in holes:
        m[rect_mask(xc, yc, h)] = PIT
    return m


def flight(xc, yc, xr, y_start, n, d, h, z_top, dr):
    """One flight; returns the map, the end y and the end z."""
    m = empty(len(xc), len(yc))
    y = y_start
    for i in range(n):
        if dr > 0:
            r = (xr[0], xr[1], y, y + d)
        else:
            r = (xr[0], xr[1], y - d, y)
        m[rect_mask(xc, yc, r)] = z_top - (i + 1) * h
        y += dr * d
    return m, y, z_top - n * h


def landing(xc, yc, xr, y0, y1, z):
    m = empty(len(xc), len(yc))
    m[rect_mask(xc, yc, (xr[0], xr[1], min(y0, y1), max(y0, y1)))] = z
    return m


def astar(hmap, xc, yc, start_xy, goal_xy, max_climb=0.28, clearance=1.20):
    import heapq
    nx, ny = hmap.shape
    walk = hmap > PIT + 0.1
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(walk) * RES
        er = walk & (dist > clearance)
        if er.sum() > 0.5 * walk.sum():
            walk = er
    except Exception:
        pass

    def idx(xy):
        return (int(np.clip(round((xy[0] - xc[0]) / RES), 0, nx - 1)),
                int(np.clip(round((xy[1] - yc[0]) / RES), 0, ny - 1)))

    def snap(ix, iy):
        if walk[ix, iy]:
            return ix, iy
        for rad in range(1, 60):
            best, bd = None, 1e9
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    a, b = ix + dx, iy + dy
                    if 0 <= a < nx and 0 <= b < ny and walk[a, b]:
                        dd = dx * dx + dy * dy
                        if dd < bd:
                            bd, best = dd, (a, b)
            if best:
                return best
        raise RuntimeError("no walkable cell near %s" % (start_xy,))

    s, g = snap(*idx(start_xy)), snap(*idx(goal_xy))
    openq = [(0.0, 0.0, s)]
    gs, came, seen = {s: 0.0}, {}, set()
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    while openq:
        _f, gc, cur = heapq.heappop(openq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == g:
            break
        a, b = cur
        for da, db, st in nbrs:
            na, nb = a + da, b + db
            if not (0 <= na < nx and 0 <= nb < ny) or not walk[na, nb]:
                continue
            if abs(float(hmap[na, nb]) - float(hmap[a, b])) > max_climb:
                continue
            ng = gc + st
            if ng < gs.get((na, nb), 1e18):
                gs[(na, nb)] = ng
                came[(na, nb)] = cur
                dd = (na - g[0]) ** 2 + (nb - g[1]) ** 2
                heapq.heappush(openq, (ng + RES * np.sqrt(dd), ng, (na, nb)))
    if g not in came and g != s:
        raise RuntimeError("A* failed")
    cells, cur = [], g
    while cur != s:
        cells.append(cur)
        cur = came[cur]
    cells.append(s)
    cells.reverse()
    pts = np.array([[xc[a], yc[b], float(hmap[a, b])] for a, b in cells])
    keep = [0]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts):
            p, q = pts[i], pts[j + 1]
            seg = q - p
            ln = np.linalg.norm(seg[:2])
            if ln < 1e-9:
                j += 1
                continue
            rel = pts[i + 1:j + 1] - p
            cr = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / ln
            if cr.max() > 0.10:
                break
            j += 1
        keep.append(j)
        i = j
    return pts[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mfsb")
    args = ap.parse_args()

    nx, ny, xc, yc = grid_axes()
    z3, z2, z1 = 2 * H, H, 0.0
    print("grid %dx%d  storey H=%.2f m  riser=%.2f tread=%.2f  %d steps/flight"
          % (nx, ny, H, RISER, TREAD, NSTEP))

    # ---- stair A: 3F -> 2F (descending +y) --------------------------------
    a1, ya1, za1 = flight(xc, yc, XA, A_Y0, NSTEP, TREAD, RISER, z3, +1)
    la = landing(xc, yc, XA, ya1, ya1 + LAND, za1)
    ya2 = ya1 + LAND
    if LAND > 0:
        a2, ya2b, za2 = flight(xc, yc, XA, ya2, NSTEP, TREAD, RISER, za1, +1)
    else:
        a2, ya2b, za2 = a1, ya1, za1

    # ---- stair B: 2F -> 1F (descending -y) --------------------------------
    b1, yb1, zb1 = flight(xc, yc, XB, B_Y0, NSTEP, TREAD, RISER, z2, -1)
    lb = landing(xc, yc, XB, yb1, yb1 - LAND, zb1)
    yb2 = yb1 - LAND
    if LAND > 0:
        b2, yb2b, zb2 = flight(xc, yc, XB, yb2, NSTEP, TREAD, RISER, zb1, -1)
    else:
        b2, yb2b, zb2 = b1, yb1, zb1

    print("A: f1 y %.2f->%.2f z %.2f->%.2f | landing -> %.2f | f2 y %.2f->%.2f z ->%.2f"
          % (A_Y0, ya1, z3, za1, ya1 + LAND, ya2, ya2b, za2))
    print("B: f1 y %.2f->%.2f z %.2f->%.2f | landing -> %.2f | f2 y %.2f->%.2f z ->%.2f"
          % (B_Y0, yb1, z2, zb1, yb1 - LAND, yb2, yb2b, zb2))

    # extend the upper opening 2 m past the last stair so the 3F slab edge
    # is not visible above the descent (it reads as a low ceiling)
    holeA = (XA[0], XA[1], A_Y0, ya2b + 2.0)
    # extend only past the LOWER end of the flight (the top must stay solid so
    # the character can step onto the first tread from the 2F floor)
    holeB = (XB[0], XB[1], yb2b - 2.0, B_Y0)
    # continuous well instead of having a floor right under the second flight
    holeA_2F = (XA[0], XA[1], A_Y0, ya2b)

    if LAND > 0:
        sheets = [
            ("floor_3F", slab_map(xc, yc, z3, (holeA,))),
            ("A1", a1), ("A_land", la), ("A2", a2),
            ("floor_2F", slab_map(xc, yc, z2, (holeB, holeA_2F))),
            ("B1", b1), ("B_land", lb), ("B2", b2),
            ("floor_1F", slab_map(xc, yc, z1)),
        ]
    else:
        sheets = [
            ("floor_3F", slab_map(xc, yc, z3, (holeA,))),
            ("A", a1),
            ("floor_2F", slab_map(xc, yc, z2, (holeB, holeA_2F))),
            ("B", b1),
            ("floor_1F", slab_map(xc, yc, z1)),
        ]
    names = [n for n, _ in sheets]
    maps = [m for _, m in sheets]
    print("sheets:", names)

    # ---- route: A* on each sheet, stitched at the sheet boundaries --------
    xa_c = 0.5 * (XA[0] + XA[1])
    xb_c = 0.5 * (XB[0] + XB[1])
    if LAND > 0:
        portals = [
            (xa_c, A_Y0), (xa_c, ya1), (xa_c, ya1 + LAND), (xa_c, ya2b),
            (xb_c, B_Y0), (xb_c, yb1), (xb_c, yb1 - LAND), (xb_c, yb2b),
        ]
    else:
        portals = [
            (xa_c, A_Y0), (xa_c, ya1),
            (xb_c, B_Y0), (xb_c, yb1),
        ]
    pts_chain = [START] + portals + [GOAL]
    path = []
    for i, nm in enumerate(names):
        p0, p1 = pts_chain[i], pts_chain[i + 1]
        pts = astar(maps[i], xc, yc, p0, p1)
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum()
        print("  A* %-9s %-16s -> %-16s %5.2f m %d pts"
              % (nm, "(%.2f,%.2f)" % p0, "(%.2f,%.2f)" % p1, seg, len(pts)))
        path.append(pts[1:] if path else pts)
    path = np.concatenate(path, axis=0)
    total = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
    print("total route %.2f m, %d waypoints" % (total, len(path)))

    # ---- physics mesh ------------------------------------------------------
    m = Mesh()
    for r in [(WEST, z3), (NORTH, z3)]:
        pass
    # slabs with their holes, split into rectangles
    def add_slab(rects, z, hole, hole2=None):
        if hole2 is not None:
            tmp = []
            hx0, hx1, hy0, hy1 = hole2
            for r in rects:
                ix0, ix1 = max(r[0], hx0), min(r[1], hx1)
                iy0, iy1 = max(r[2], hy0), min(r[3], hy1)
                if ix0 < ix1 and iy0 < iy1:
                    if r[2] < iy0: tmp.append((r[0], r[1], r[2], iy0))
                    if iy1 < r[3]: tmp.append((r[0], r[1], iy1, r[3]))
                    if r[0] < ix0: tmp.append((r[0], ix0, iy0, iy1))
                    if ix1 < r[1]: tmp.append((ix1, r[1], iy0, iy1))
                else:
                    tmp.append(r)
            rects = tmp
        for r in rects:
            pieces = [r]
            if hole is not None:
                hx0, hx1, hy0, hy1 = hole
                ix0, ix1 = max(r[0], hx0), min(r[1], hx1)
                iy0, iy1 = max(r[2], hy0), min(r[3], hy1)
                if ix0 < ix1 and iy0 < iy1:
                    pieces = []
                    if r[2] < iy0:
                        pieces.append((r[0], r[1], r[2], iy0))
                    if iy1 < r[3]:
                        pieces.append((r[0], r[1], iy1, r[3]))
                    if r[0] < ix0:
                        pieces.append((r[0], ix0, iy0, iy1))
                    if ix1 < r[1]:
                        pieces.append((ix1, r[1], iy0, iy1))
            for p in pieces:
                m.box(p[0], p[1], p[2], p[3], z)

    add_slab(U_RECTS, z3, holeA)
    add_slab(U_RECTS, z2, holeB, holeA_2F)
    add_slab(U_RECTS, z1, None)

    def add_flight(xr, y_start, n, d, h, z_top, dr):
        y = y_start
        for i in range(n):
            if dr > 0:
                m.box(xr[0], xr[1], y, y + d, z_top - (i + 1) * h, thick=0.16)
            else:
                m.box(xr[0], xr[1], y - d, y, z_top - (i + 1) * h, thick=0.16)
            y += dr * d

    add_flight(XA, A_Y0, NSTEP, TREAD, RISER, z3, +1)
    add_flight(XB, B_Y0, NSTEP, TREAD, RISER, z2, -1)
    if LAND > 0:
        add_flight(XA, ya2, NSTEP, TREAD, RISER, za1, +1)
        add_flight(XB, yb2, NSTEP, TREAD, RISER, zb1, -1)
    if LAND > 0:
        m.box(XA[0], XA[1], ya1, ya1 + LAND, za1, thick=0.16)
        m.box(XB[0], XB[1], yb1 - LAND, yb1, zb1, thick=0.16)

    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts %d tris  z %.2f..%.2f"
          % (len(verts), len(faces), verts[:, 2].min(), verts[:, 2].max()))

    # ---- query check -------------------------------------------------------
    ok = True
    print("surface query check:")
    RISE = 0.35
    probes = [
        ("3F slab", (1.70, 1.20), z3, z3),
        ("A1 mid", (1.70, 3.00), z3 - 3 * RISER, z3 - 3 * RISER),
        ("A landing", (1.70, ya1 + LAND / 2), za1, za1) if LAND > 0 else ("skipA", (1.70, 1.20), z3, z3),
        ("A2 mid", (1.70, ya2 + 3 * TREAD), za1 - 3 * RISER, za1 - 3 * RISER),
        ("2F slab", (1.70, ya2b + 0.6), z2, z2),
        ("1F under 2F", (1.70, ya2b + 0.6), 0.0, 0.0),
        ("B landing", (9.00, yb1 - LAND / 2), zb1, zb1) if LAND > 0 else ("skipB", (9.20, 1.00), 0.0, 0.0),
        ("1F goal", GOAL, 0.0, 0.0),
    ]
    for label, (px, py), foot, exp in probes:
        ix = int((px - xc[0]) / RES)
        iy = int((py - yc[0]) / RES)
        vals = [float(mm[ix, iy]) for mm in maps]
        usable = [v for v in vals if v <= foot + RISE]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-12s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
              % (label, px, py, foot, got, exp, flag))

    # ---- write -------------------------------------------------------------
    os.makedirs("data/mesh", exist_ok=True)
    os.makedirs("data/paths", exist_ok=True)
    stack = np.stack(maps).astype(np.float32)
    walk = (stack > PIT + 0.1).astype(np.int8)
    cam = np.array([OFFSET[0] + EXTENT[0] / 2.0, OFFSET[1] + EXTENT[1] / 2.0],
                   dtype=np.float32)
    payload = dict(
        vertices=verts, faces=faces, heigthmap=stack, walkable_map=walk,
        cam_pos=cam, x_scale=1.0 / RES, y_scale=1.0 / RES,
        start_block=0, start_z=z3, rise_tol=RISE,
        start_yaw=1.5707963267948966,
        blocks=[dict(id=i, name=n, kind=("floor" if n.startswith("floor") else "stair"))
                for i, n in enumerate(names)],
    )
    with open("data/mesh/%s.pkl" % args.name, "wb") as fh:
        pickle.dump(payload, fh)
    with open("data/paths/%s.json" % args.name, "w", encoding="utf-8") as fh:
        json.dump({"name": args.name,
                   "note": "3F -> A(2 flights) -> 2F -> B(2 flights) -> 1F",
                   "path": [[round(float(p[0]), 3), round(float(p[1]), 3),
                             round(float(p[2]), 3)] for p in path]}, fh, indent=2)

    print("written data/mesh/%s.pkl  (%d sheets)" % (args.name, len(names)))
    print("written data/paths/%s.json (%d points)" % (args.name, len(path)))
    print("START_XY=%.2f,%.2f  GOAL_XY=%.2f,%.2f" % (START + GOAL))
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
