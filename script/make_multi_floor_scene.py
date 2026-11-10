"""Three-storey building, layered heightmaps, per-block A* route.

Layout (plan view, metres, corridor width C = 3.4):

        y
        ^
     12 +-----------------------------+
        |        north landing         |   x[0,10.8] y[8.6,12]
   8.6  +--------+----------+---------+
        |  west  |          |  east   |   west x[0,3.4]  east x[7.4,10.8]
        |  corr. |          |  corr.  |
      0 +--------+----------+---------+
        0      3.4 4.0     7.4      10.8  x

    stair A (3F -> 2F): west corridor, W=3.0, descends +y, hole in the 3F slab
    stair B (2F -> 1F): east corridor, W=2.8, descends -y, hole in the 2F slab

Storey height H = 2.4 m, so 1F z=0, 2F z=2.4, 3F z=4.8.  The slabs are real
geometry (stacked, XY overlapping); the layered heightmaps are only the
perception proxy, so the stairwell holes are genuinely open in the mesh and
read PIT in the owning block's map.

Every stair block's map contains three parts:
    [1.5 m upper slab] [N treads] [1.5 m lower slab]
so it overlaps both neighbouring floor blocks by 1.5 m *with identical heights*
-- switching blocks there cannot change the observation.

Run on the server (needs isaacgym for nothing here; pure numpy):
    python script/make_multi_floor_scene.py --name mf3
"""

import argparse
import json
import os
import pickle

import numpy as np

RES = 0.05
PIT = -1.5
SLAB_T = 0.20          # slab thickness for the physics mesh

# ---- storey / stair parameters (stage4 distribution, storey-constrained) -----
H = 1.8                # storey height (headroom for a 1.75 m body)
C = 3.4                # corridor width  (< 4 m, see the design doc)
OVERLAP = 1.5          # stair <-> floor overlap strip

STAIRS = {
    # name: width, steps, tread, direction (+1/-1 along y), top y, x range
    "A": dict(w=3.0, n=12, d=0.36, dir=+1, y_top=3.00, x=(0.20, 3.20)),
    "B": dict(w=2.8, n=12, d=0.36, dir=-1, y_top=8.00, x=(7.60, 10.40)),
}
for _s in STAIRS.values():
    _s["h"] = H / _s["n"]
    _s["L"] = _s["n"] * _s["d"]
    _s["y_bot"] = _s["y_top"] + _s["dir"] * _s["L"]

# U-shaped corridor rectangles per floor: (x0, x1, y0, y1)
WEST = (0.0, 3.4, 0.0, 12.0)
NORTH = (0.0, 10.8, 8.6, 12.0)
EAST = (7.4, 10.8, 0.0, 12.0)
U_RECTS = [WEST, NORTH, EAST]

START = (1.70, 1.20)      # 3F
GOAL = (9.00, 1.20)       # 1F

OFFSET = (-2.0, -2.0)     # grid origin in world coords
EXTENT = (15.0, 16.0)     # grid size in metres


# ---------------------------------------------------------------------------
# physics mesh
# ---------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.v = []
        self.f = []

    def box(self, x0, x1, y0, y1, z_top, thick=SLAB_T):
        z0, z1 = z_top - thick, z_top
        base = len(self.v)
        self.v += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        quads = [(4, 5, 6, 7), (0, 1, 2, 3), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for a, b, c, d in quads:
            self.f += [(base + a, base + b, base + c), (base + a, base + c, base + d)]


def subtract_rect(rect, hole):
    """Split an axis-aligned rectangle by a hole rectangle; return the pieces."""
    x0, x1, y0, y1 = rect
    hx0, hx1, hy0, hy1 = hole
    ix0, ix1 = max(x0, hx0), min(x1, hx1)
    iy0, iy1 = max(y0, hy0), min(y1, hy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return [rect]
    out = []
    if y0 < iy0:
        out.append((x0, x1, y0, iy0))
    if iy1 < y1:
        out.append((x0, x1, iy1, y1))
    if x0 < ix0:
        out.append((x0, ix0, iy0, iy1))
    if ix1 < x1:
        out.append((ix1, x1, iy0, iy1))
    return out


def floor_rects(hole=None):
    rects = []
    for r in U_RECTS:
        rects += subtract_rect(r, hole) if hole else [r]
    return rects


# ---------------------------------------------------------------------------
# per-block heightmaps
# ---------------------------------------------------------------------------
def grid_axes():
    nx = int(round(EXTENT[0] / RES))
    ny = int(round(EXTENT[1] / RES))
    xc = OFFSET[0] + (np.arange(nx) + 0.5) * RES
    yc = OFFSET[1] + (np.arange(ny) + 0.5) * RES
    return nx, ny, xc, yc


def mask_rect(xc, yc, rect):
    x0, x1, y0, y1 = rect
    return ((xc[:, None] >= x0) & (xc[:, None] <= x1)
            & (yc[None, :] >= y0) & (yc[None, :] <= y1))


def block_map(xc, yc, z_floor, hole_rects=()):
    """Flat floor block at z_floor, minus open holes (PIT)."""
    m = np.full((len(xc), len(yc)), PIT, dtype=np.float32)
    for r in U_RECTS:
        m[mask_rect(xc, yc, r)] = z_floor
    for h in hole_rects:
        m[mask_rect(xc, yc, h)] = PIT
    return m


def stair_map(xc, yc, s, z_top, z_bot):
    """Stair block: 1.5 m upper slab + N treads + 1.5 m lower slab."""
    h, n, d, dr, xr = s["h"], s["n"], s["d"], s["dir"], s["x"]
    y_top, y_bot = s["y_top"], s["y_bot"]
    m = np.full((len(xc), len(yc)), PIT, dtype=np.float32)

    lo, hi = (y_top - dr * n * d, y_top) if dr > 0 else (y_bot, y_top)
    # treads
    for i in range(n):
        if dr > 0:
            y0, y1 = y_top + i * d, y_top + (i + 1) * d
        else:
            y0, y1 = y_top - (i + 1) * d, y_top - i * d
        z = z_top - (i + 1) * h
        m[mask_rect(xc, yc, (xr[0], xr[1], y0, y1))] = z

    # No outward extension: each sheet is a pure physical surface and the
    # height query resolves per cell, so nothing needs to be duplicated onto
    # the neighbouring floor.
    return m


# ---------------------------------------------------------------------------
# A* on one block
# ---------------------------------------------------------------------------
def astar(hmap, xc, yc, start_xy, goal_xy, max_climb=0.28, clearance=0.35):
    """2D A* on one block, kept `clearance` metres away from any drop.

    Without the clearance band the route hugs the corridor edge and cuts the
    corner at the landing: the controller follows it, drifts within ~0.2 m of
    the pit and falls.  Eroding the walkable mask is the cheap equivalent of a
    clearance cost and keeps the whole route inside a safe corridor.
    """
    import heapq
    nx, ny = hmap.shape
    walk = hmap > PIT + 0.1
    if clearance > 0:
        try:
            from scipy.ndimage import distance_transform_edt
            dist_m = distance_transform_edt(walk) * RES
            eroded = walk & (dist_m > clearance)
            # keep at least a thin band everywhere so a narrow flight still works
            if eroded.sum() > 0.5 * walk.sum():
                walk = eroded
            else:
                print("  ! clearance %.2f m too wide, using the raw mask" % clearance)
        except Exception as e:
            print("  ! no distance transform (%s), raw mask" % e)

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
        raise RuntimeError("A* failed on block")
    cells, cur = [], g
    while cur != s:
        cells.append(cur)
        cur = came[cur]
    cells.append(s)
    cells.reverse()
    pts = np.array([[xc[a], yc[b], float(hmap[a, b])] for a, b in cells])
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    keep = [0]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts):
            p, q = pts[i], pts[j + 1]
            seg_ = q - p
            ln = np.linalg.norm(seg_[:2])
            if ln < 1e-9:
                j += 1
                continue
            rel = pts[i + 1:j + 1] - p
            cr = np.abs(seg_[0] * rel[:, 1] - seg_[1] * rel[:, 0]) / ln
            if cr.max() > 0.10:
                break
            j += 1
        keep.append(j)
        i = j
    return pts[keep], float(seg.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mf3")
    args = ap.parse_args()

    nx, ny, xc, yc = grid_axes()
    print("grid %dx%d  (%.1f x %.1f m)  RES=%.3f" % (nx, ny, EXTENT[0], EXTENT[1], RES))

    A, B = STAIRS["A"], STAIRS["B"]
    print("storey H=%.2f m   corridor C=%.1f m" % (H, C))
    print("stair A: W=%.1f N=%d h=%.3f d=%.2f L=%.2f  y %.2f -> %.2f  dir=%+d"
          % (A["w"], A["n"], A["h"], A["d"], A["L"], A["y_top"], A["y_bot"], A["dir"]))
    print("stair B: W=%.1f N=%d h=%.3f d=%.2f L=%.2f  y %.2f -> %.2f  dir=%+d"
          % (B["w"], B["n"], B["h"], B["d"], B["L"], B["y_top"], B["y_bot"], B["dir"]))
    print("flight rise: A %.2f m   B %.2f m" % (A["n"] * A["h"], B["n"] * B["h"]))

    holeA = (A["x"][0], A["x"][1], A["y_top"], A["y_bot"]) if A["dir"] > 0 else \
            (A["x"][0], A["x"][1], A["y_bot"], A["y_top"])
    holeB = (B["x"][0], B["x"][1], min(B["y_top"], B["y_bot"]), max(B["y_top"], B["y_bot"]))

    stairA_map = stair_map(xc, yc, A, 2 * H, H)
    stairB_map = stair_map(xc, yc, B, H, 0.0)

    def overlay(m, tread, s):
        """Put a staircase's treads into a floor block's map.

        A level's own map must show the flights attached to it, otherwise the
        approach to a stairwell reads as a pit and the controller refuses to
        step over the edge: it would have to stand on the stairs before it can
        see them.  With the treads in the level map the descent is visible from
        the landing, which is also how a real stairwell looks.
        """
        hole = (s["x"][0], s["x"][1],
                min(s["y_top"], s["y_bot"]), max(s["y_top"], s["y_bot"]))
        m[mask_rect(xc, yc, hole)] = tread[mask_rect(xc, yc, hole)]
        return m

    # Pure surfaces: the slabs (with their holes) and the two flights.  The
    # query "highest sheet not above foot + rise_tol" picks the right one per
    # cell, so no sheet needs to describe another sheet's geometry.
    maps = {
        "floor_1F": block_map(xc, yc, 0.0),
        "floor_2F": block_map(xc, yc, H, hole_rects=[holeB]),
        "floor_3F": block_map(xc, yc, 2 * H, hole_rects=[holeA]),
        "stair_A": stairA_map,
        "stair_B": stairB_map,
    }
    order = ["floor_3F", "stair_A", "floor_2F", "stair_B", "floor_1F"]
    for nm in order:
        w = (maps[nm] > PIT + 0.1)
        print("  %-9s walkable %5.1f%%  z %.2f .. %.2f" % (
            nm, 100.0 * w.mean(), maps[nm][w].min() if w.any() else 0,
            maps[nm][w].max() if w.any() else 0))

    # ---- route: one A* per block, stitched at portals ----------------------
    portals = [
        (1.70, A["y_top"]),                     # 3F  -> A
        (1.70, A["y_bot"]),                     # A   -> 2F
        (9.00, B["y_top"]),                     # 2F  -> B
        (9.00, B["y_bot"]),                     # B   -> 1F
    ]
    legs = [(START, portals[0]), (portals[0], portals[1]),
            (portals[1], portals[2]), (portals[2], portals[3]),
            (portals[3], GOAL)]
    seg_pts = []
    for nm, (p0, p1) in zip(order, legs):
        pts, length = astar(maps[nm], xc, yc, p0, p1)
        print("  A* %-9s %-16s -> %-16s  %5.2f m  %d pts"
              % (nm, "(%.2f,%.2f)" % p0, "(%.2f,%.2f)" % p1, length, len(pts)))
        seg_pts.append(pts)

    path = []
    for i, pts in enumerate(seg_pts):
        path.append(pts[1:] if i else pts)
    path = np.concatenate(path, axis=0)

    # Per-block sub-paths: the controller re-anchors the trajectory to the
    # current block's sub-path when the block changes, instead of following one
    # long concatenated polyline.  Every block shares the same world frame, so
    # switching needs no coordinate transform.  Each entry carries a ~2 m tail
    # of the next block so the lookahead does not collapse at the portal.
    block_paths = []
    for i, pts in enumerate(seg_pts):
        if i + 1 < len(seg_pts):
            tail = seg_pts[i + 1][1:3]
            block_paths.append(np.vstack([pts, tail]) if len(tail) else pts)
        else:
            block_paths.append(pts)
    seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    print("total route %.2f m, %d waypoints" % (seg.sum(), len(path)))

    # ---- physics mesh: slabs with real holes + treads ----------------------
    m = Mesh()
    for z, hole in ((0.0, None), (H, holeB), (2 * H, holeA)):
        for r in floor_rects(hole):
            m.box(r[0], r[1], r[2], r[3], z)
    for s, z_top in ((A, 2 * H), (B, H)):
        for i in range(s["n"]):
            if s["dir"] > 0:
                y0, y1 = s["y_top"] + i * s["d"], s["y_top"] + (i + 1) * s["d"]
            else:
                y0, y1 = s["y_top"] - (i + 1) * s["d"], s["y_top"] - i * s["d"]
            m.box(s["x"][0], s["x"][1], y0, y1, z_top - (i + 1) * s["h"], thick=0.16)
    verts = np.asarray(m.v, dtype=np.float32)
    faces = np.asarray(m.f, dtype=np.int32)
    print("mesh: %d verts, %d tris   z %.2f .. %.2f"
          % (len(verts), len(faces), verts[:, 2].min(), verts[:, 2].max()))

    # ---- verification probes ----------------------------------------------
    print("probes (world -> block heights):")
    probes = [("3F start", START, ["floor_3F", "floor_2F", "floor_1F"]),
              ("A mid", (1.7, (A["y_top"] + A["y_bot"]) / 2), ["stair_A"]),
              ("2F landing", (1.7, A["y_bot"] + 0.3), ["stair_A", "floor_2F"]),
              ("B mid", (9.0, (B["y_top"] + B["y_bot"]) / 2), ["stair_B"]),
              ("goal", GOAL, ["floor_1F"])]
    ok = True
    for label, (px, py), blocks in probes:
        ix = int((px - xc[0]) / RES)
        iy = int((py - yc[0]) / RES)
        vals = " ".join("%s=%.2f" % (b, maps[b][ix, iy]) for b in blocks)
        print("   %-12s (%5.2f,%5.2f)  %s" % (label, px, py, vals))

    # query check: highest sheet not above foot + rise_tol, per probe point
    print("surface query check (foot -> expected):")
    RISE = 0.35
    _hA = A["h"]                       # riser of flight A
    probes = [
        ("3F slab",       (1.70, 1.20), 2 * H,                 2 * H),
        ("A 1st tread",   (1.70, 3.20), 2 * H - _hA,           2 * H - _hA),
        ("A mid",         (1.70, 5.04), stairA_map[int((5.04 - yc[0]) / RES),
                                                   int((1.70 - xc[0]) / RES)] if False else 2 * H - 6 * _hA, 2 * H - 6 * _hA),
        ("2F slab",       (1.70, 7.38), H,                     H),
        ("1F under 2F",   (1.70, 7.38), 0.0,                   0.0),
        ("1F goal",       (9.00, 1.20), 0.0,                   0.0),
        ("3F sees stairs",(1.70, 5.04), 2 * H,                 2 * H - 6 * _hA),
    ]
    for label, (px, py), foot, exp in probes:
        ix = int((px - xc[0]) / RES)
        iy = int((py - yc[0]) / RES)
        vals = [float(maps[n][ix, iy]) for n in order]
        usable = [v for v in vals if v <= foot + RISE]
        got = max(usable) if usable else PIT
        flag = "OK" if abs(got - exp) < 0.01 else "!!"
        if flag == "!!":
            ok = False
        print("   %-13s (%5.2f,%5.2f) foot %5.2f -> %5.2f  expect %5.2f  %s"
              % (label, px, py, foot, got, exp, flag))
    # ---- write -------------------------------------------------------------
    os.makedirs("data/mesh", exist_ok=True)
    os.makedirs("data/paths", exist_ok=True)
    stack = np.stack([maps[n] for n in order]).astype(np.float32)
    walk = (stack > PIT + 0.1).astype(np.int8)
    cam = np.array([xc[0] - 0.5 * RES + nx * RES / 2.0,
                    yc[0] - 0.5 * RES + ny * RES / 2.0], dtype=np.float32)
    payload = dict(
        vertices=verts, faces=faces,
        heigthmap=stack, walkable_map=walk,
        cam_pos=cam, x_scale=1.0 / RES, y_scale=1.0 / RES,
        start_block=order.index("floor_3F"),
        start_yaw=1.5707963267948966,   # route starts along +y
        start_z=2 * H,                  # spawn reference surface (3F)
        rise_tol=0.35,                  # a surface above foot + 0.35 is not walkable
        block_paths=[[[float(p[0]), float(p[1]), float(p[2])] for p in bp]
                     for bp in block_paths],
        blocks=[dict(id=i, name=n, kind=("stair" if n.startswith("stair") else "floor"),
                     z=float(np.median(maps[n][maps[n] > PIT + 0.1])))
                for i, n in enumerate(order)],
        portals=[dict(a=0, b=1, p=[1.70, A["y_top"]], normal=[0.0, 1.0]),
                 dict(a=1, b=2, p=[1.70, A["y_bot"]], normal=[0.0, 1.0]),
                 dict(a=2, b=3, p=[9.00, B["y_top"]], normal=[0.0, -1.0]),
                 dict(a=3, b=4, p=[9.00, B["y_bot"]], normal=[0.0, -1.0])],
    )
    with open("data/mesh/%s.pkl" % args.name, "wb") as fh:
        pickle.dump(payload, fh)
    pj = {"name": args.name,
          "note": "3F -> stairA -> 2F -> stairB -> 1F (layered heightmaps)",
          "path": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                   for p in path]}
    with open("data/paths/%s.json" % args.name, "w", encoding="utf-8") as fh:
        json.dump(pj, fh, indent=2)

    print("written data/mesh/%s.pkl  (%d blocks)" % (args.name, len(order)))
    print("written data/paths/%s.json (%d points)" % (args.name, len(path)))
    print("START_XY=%.2f,%.2f  GOAL_XY=%.2f,%.2f" % (START + GOAL))
    print("verification: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
