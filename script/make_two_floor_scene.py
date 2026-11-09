"""Two-floor split-level scene with an A* planned route.

Layout (local frame, metres, Z up):

    2F platform  x[-8,-2]  y[-2,10]   z = H2
    stairwell    x[-2, 1]  y[ 2,4.4]  10 steps down, stage-4 difficulty
    1F hall      x[ 1,14]  y[-6, 8]   z = 0
    everything else = pit (-1.5 m)

The heightfield cannot hold two floors at the same (x, y), so this is a
split-level: 2F plateau -> narrow stairwell -> 1F hall.  The route is planned by
A* on the walkable heightfield with:
  * a maximum climb per cell (so it cannot walk off any edge),
  * a corridor-centring term inside the stairwell,
  * a clearance term that pushes the route away from pit borders.

Run on the server (needs isaacgym.terrain_utils):
    python script/make_two_floor_scene.py --name two_floor
"""

import argparse
import heapq
import json
import os
import pickle

import numpy as np
from isaacgym import terrain_utils

RES = 0.05
VSCALE = 0.01
PIT = -1.5

# ---- stage-4 stair difficulty ------------------------------------------------
RISER = 0.16
TREAD = 0.30
NSTEPS = 10
WIDTH = 2.4
H2 = RISER * NSTEPS

# ---- layout in the local frame ----------------------------------------------
# narrow corridor layout: the whole route is one 2.4 m wide stairwell
CY = 3.2                       # corridor centre line (y)
F2 = dict(x=(-9.0, -2.0), y=(CY - WIDTH / 2.0, CY + WIDTH / 2.0))
ST = dict(x=(-2.0, -2.0 + NSTEPS * TREAD), y=(CY - WIDTH / 2.0, CY + WIDTH / 2.0))
F1 = dict(x=(ST["x"][1], 16.0), y=(CY - WIDTH / 2.0, CY + WIDTH / 2.0))
START = (-7.0, CY)
GOAL = (14.5, CY)

MARGIN = 2.0
MAX_CLIMB = 0.20          # metres per RES cell that A* may traverse
W_STAIR_CENTER = 4.0      # corridor centring weight inside the stairwell
W_CLEARANCE = 1.2         # keep away from pit borders


def build_heightfield(shift_x, shift_y):
    x_lo = min(F2["x"][0], F1["x"][0]) - MARGIN + shift_x
    x_hi = max(F2["x"][1], F1["x"][1]) + MARGIN + shift_x
    y_lo = min(F2["y"][0], F1["y"][0]) - MARGIN + shift_y
    y_hi = max(F2["y"][1], F1["y"][1]) + MARGIN + shift_y

    nx = int(round((x_hi - x_lo) / RES))
    ny = int(round((y_hi - y_lo) / RES))
    xc = x_lo + (np.arange(nx) + 0.5) * RES
    yc = y_lo + (np.arange(ny) + 0.5) * RES

    hf = np.full((nx, ny), PIT, dtype=np.float32)

    sx0, sx1 = ST["x"][0] + shift_x, ST["x"][1] + shift_x
    sy0, sy1 = ST["y"][0] + shift_y, ST["y"][1] + shift_y

    in2 = ((xc[:, None] >= F2["x"][0] + shift_x) & (xc[:, None] <= F2["x"][1] + shift_x)
           & (yc[None, :] >= F2["y"][0] + shift_y) & (yc[None, :] <= F2["y"][1] + shift_y))
    in1 = ((xc[:, None] >= F1["x"][0] + shift_x) & (xc[:, None] <= F1["x"][1] + shift_x)
           & (yc[None, :] >= F1["y"][0] + shift_y) & (yc[None, :] <= F1["y"][1] + shift_y))
    ins = ((xc[:, None] >= sx0) & (xc[:, None] < sx1)
           & (yc[None, :] >= sy0) & (yc[None, :] < sy1))

    hf[in2] = H2
    hf[in1] = 0.0

    # stair treads: step i occupies x in [sx0 + i*TREAD, sx0 + (i+1)*TREAD)
    idx = np.floor((xc - sx0) / TREAD).astype(np.int32)
    idx = np.clip(idx, 0, NSTEPS - 1)
    stair_h = (H2 - (idx + 1) * RISER).astype(np.float32)
    stair_map = np.broadcast_to(stair_h[:, None], (nx, ny))
    hf[ins] = stair_map[ins]

    return hf, xc, yc, x_lo, y_lo, nx, ny


def astar(hf, xc, yc, start_xy, goal_xy, stair_bounds):
    nx, ny = hf.shape
    walk = hf > PIT + 0.1

    def to_idx(xy):
        ix = int(np.clip(round((xy[0] - xc[0]) / RES), 0, nx - 1))
        iy = int(np.clip(round((xy[1] - yc[0]) / RES), 0, ny - 1))
        return ix, iy

    # nearest walkable cell
    def snap(ix, iy):
        if walk[ix, iy]:
            return ix, iy
        best, bd = None, 1e9
        for dx in range(-40, 41):
            for dy in range(-40, 41):
                a, b = ix + dx, iy + dy
                if 0 <= a < nx and 0 <= b < ny and walk[a, b]:
                    d = dx * dx + dy * dy
                    if d < bd:
                        bd, best = d, (a, b)
        return best

    s = snap(*to_idx(start_xy))
    g = snap(*to_idx(goal_xy))
    print("  astar: start cell %s -> goal cell %s" % (s, g))

    # clearance from the pit borders
    try:
        from scipy.ndimage import distance_transform_edt
        clear = distance_transform_edt(walk).astype(np.float32)
    except Exception:
        clear = np.ones_like(hf, dtype=np.float32) * 10.0

    sx0, sx1, sy0, sy1 = stair_bounds
    stair_cy = 0.5 * (sy0 + sy1)

    def cell_cost(a, b):
        cost = 1.0
        x, y = xc[a], yc[b]
        if sx0 - 0.5 <= x <= sx1 + 0.5 and sy0 - 0.5 <= y <= sy1 + 0.5:
            cost += W_STAIR_CENTER * (y - stair_cy) ** 2
        cost += W_CLEARANCE / (1.0 + clear[a, b])
        return cost

    def h(a, b):
        dx = (a - g[0]) * RES
        dy = (b - g[1]) * RES
        return float(np.hypot(dx, dy))

    openq = [(h(*s), 0.0, s)]
    gscore = {s: 0.0}
    came = {}
    seen = set()
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
        for da, db, step in nbrs:
            na, nb = a + da, b + db
            if not (0 <= na < nx and 0 <= nb < ny) or not walk[na, nb]:
                continue
            if abs(float(hf[na, nb]) - float(hf[a, b])) > MAX_CLIMB:
                continue
            ng = gc + step * cell_cost(na, nb)
            if ng < gscore.get((na, nb), 1e18):
                gscore[(na, nb)] = ng
                came[(na, nb)] = cur
                heapq.heappush(openq, (ng + h(na, nb), ng, (na, nb)))
    if g not in came and g != s:
        raise RuntimeError("A* found no route")

    cells, cur = [], g
    while cur != s:
        cells.append(cur)
        cur = came[cur]
    cells.append(s)
    cells.reverse()
    return cells


def simplify(cells, xc, yc, hf, tol=0.12):
    pts = np.array([[xc[a], yc[b], float(hf[a, b])] for a, b in cells], dtype=np.float64)
    # keep first/last, drop points that are within tol of the segment
    keep = [0]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts):
            seg = pts[j + 1] - pts[i]
            ln = np.linalg.norm(seg)
            if ln < 1e-9:
                j += 1
                continue
            d = np.linalg.norm(np.cross(seg, pts[i + 1:j + 1] - pts[i])) / ln
            if d.max() > tol:
                break
            j += 1
        keep.append(j)
        i = j
    return pts[keep]


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="two_floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=float, default=RES)
    args = ap.parse_args()
    RES = args.res

    shift_x = MARGIN + 10.0
    shift_y = MARGIN + 8.0
    hf, xc, yc, x_lo, y_lo, nx, ny = build_heightfield(shift_x, shift_y)

    start = (START[0] + shift_x, START[1] + shift_y)
    goal = (GOAL[0] + shift_x, GOAL[1] + shift_y)
    stair_bounds = (ST["x"][0] + shift_x, ST["x"][1] + shift_x,
                    ST["y"][0] + shift_y, ST["y"][1] + shift_y)

    walk = hf > PIT + 0.1
    print("heightfield %dx%d  walkable %.0f%%" % (nx, ny, 100.0 * walk.mean()))
    print("  x %.2f..%.2f   y %.2f..%.2f" % (x_lo, x_lo + nx * RES, y_lo, y_lo + ny * RES))
    print("  2F top z=%.2f   1F z=0.00   stair %.0fx%.1f m  riser=%.2f tread=%.2f"
          % (H2, NSTEPS * TREAD, WIDTH, RISER, TREAD))
    print("  start (%.2f, %.2f)   goal (%.2f, %.2f)" % (start[0], start[1], goal[0], goal[1]))

    cells = astar(hf, xc, yc, start, goal, stair_bounds)
    raw = np.array([[xc[a], yc[b], float(hf[a, b])] for a, b in cells])
    seg = np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1)
    print("  A* cells=%d  raw length=%.2f m" % (len(cells), seg.sum()))

    pts = simplify(cells, xc, yc, hf)
    seg2 = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    print("  simplified waypoints=%d  length=%.2f m" % (len(pts), seg2.sum()))

    # snap endpoints exactly
    pts[0, :2] = start
    pts[-1, :2] = goal

    hf_int = np.rint(hf / VSCALE).astype(np.int16)
    verts, tris = terrain_utils.convert_heightfield_to_trimesh(
        hf_int, RES, VSCALE, slope_threshold=0.75)
    verts = np.asarray(verts, dtype=np.float32).copy()
    tris = np.asarray(tris, dtype=np.int32)
    # convert_heightfield_to_trimesh returns grid metres; shift into world frame
    verts[:, 0] += x_lo
    verts[:, 1] += y_lo

    cam = np.array([x_lo + nx * RES / 2.0, y_lo + ny * RES / 2.0], dtype=np.float32)

    mesh = dict(vertices=verts, faces=tris, heigthmap=hf,
                walkable_map=walk.astype(np.int8), cam_pos=cam,
                x_scale=1.0 / RES, y_scale=1.0 / RES)

    os.makedirs("data/mesh", exist_ok=True)
    os.makedirs("data/paths", exist_ok=True)
    with open("data/mesh/%s.pkl" % args.name, "wb") as fh:
        pickle.dump(mesh, fh)
    payload = {
        "name": args.name,
        "note": "2F platform -> 10-step stage4 stairwell -> 1F hall -> goal",
        "path": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                 for p in pts],
    }
    with open("data/paths/%s.json" % args.name, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("written data/mesh/%s.pkl (%d verts, %d tris)" % (args.name, len(verts), len(tris)))
    print("written data/paths/%s.json (%d points)" % (args.name, len(pts)))
    print("START_XY=%.2f,%.2f" % (start[0], start[1]))
    print("GOAL_XY=%.2f,%.2f" % (goal[0], goal[1]))
    print("waypoints:")
    for p in pts:
        print("   %7.2f %7.2f %6.2f" % (p[0], p[1], p[2]))


if __name__ == "__main__":
    main()
