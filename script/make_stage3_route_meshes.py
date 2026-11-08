"""Stage-3 route meshes: up 5 -> flat -> 90 deg turn -> up 10 -> flat ->
down 8 -> flat -> up 10, with a wider corridor than the training mean.

Riser / tread are sampled from the Stage 3 config (0.10-0.18 / 0.28-0.38); the
width uses the upper end (3.2/3.0/2.8 m) because this test is meant to look
like a real building stairwell rather than the narrowest training tile.

Each mesh is written together with a matching path json (absolute coordinates,
the loader re-aligns the first waypoint to the spawn).

    python script/make_stage3_route_meshes.py --count 4
"""
import argparse
import json
import os
import pickle

import numpy as np
import yaml
from isaacgym import terrain_utils

RES = 0.05
VSCALE = 0.01
APPROACH = 2.0
FLAT1 = 1.5          # flat walk before the turn
FLAT2 = 1.5          # flat after the first flight
FLAT3 = 1.5          # flat after the descent
FLAT4 = 1.5          # flat after the second flight
END_FLAT = 1.5
PIT = -1.5
MARGIN = 1.0
WIDTH_BINS = [3.2, 3.0, 2.8]
WIDTH_PROBS = [0.35, 0.35, 0.30]

N_UP1, N_UP2, N_DOWN, N_UP3 = 5, 10, 8, 10


def ramp(xc, start, n, d, h, base):
    """Height of an n-step flight starting at `start` along a 1-D coordinate."""
    out = np.full(xc.shape, base, dtype=np.float32)
    x = start
    for i in range(n):
        out[(xc >= x) & (xc < x + d)] = base + (i + 1) * h
        x += d
    out[xc >= x] = base + n * h
    return out, x


def build(h, d, width):
    half = width / 2.0
    x_turn = APPROACH + N_UP1 * d + FLAT1
    y_end = half + N_UP2 * d + FLAT2 + N_DOWN * d + FLAT3 + N_UP3 * d + END_FLAT
    x_len = x_turn + half + MARGIN
    y_len = y_end + MARGIN

    nx = int(round(x_len / RES))
    ny = int(round((y_len + half + MARGIN) / RES))
    y_off = half + MARGIN

    xc = (np.arange(nx) + 0.5) * RES
    yc = (np.arange(ny) + 0.5) * RES - y_off

    # piece 1: along +x
    p1, _ = ramp(xc, APPROACH, N_UP1, d, h, 0.0)
    h1 = p1
    # piece 2: along +y, starting at height N_UP1*h after the turn
    top = N_UP1 * h
    s = half + N_UP2 * d + FLAT2
    p2, s = ramp(yc, half, N_UP2, d, h, top)
    p2[yc >= s] = top + N_UP2 * h
    # descend
    level = top + N_UP2 * h
    y = s
    for i in range(N_DOWN):
        p2[(yc >= y) & (yc < y + d)] = level - (i + 1) * h
        y += d
    level -= N_DOWN * h
    p2[yc >= y] = level
    # second ascent
    y0 = y + FLAT3
    p2[(yc >= y) & (yc < y0)] = level
    p3, _ = ramp(yc, y0, N_UP3, d, h, level)
    p2[yc >= y0] = p3[yc >= y0]

    in1 = (xc[:, None] <= x_turn + half) & (np.abs(yc[None, :]) <= half)
    in2 = (np.abs(xc[:, None] - x_turn) <= half) & (yc[None, :] >= -half)

    hf = np.full((nx, ny), PIT, dtype=np.float32)
    hf[in1] = np.broadcast_to(h1[:, None], (nx, ny))[in1]
    hf[in2] = np.broadcast_to(p2[None, :], (nx, ny))[in2]

    hf_int = np.rint(hf / VSCALE).astype(np.int16)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        hf_int, RES, VSCALE, slope_threshold=0.75)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    vertices[:, 1] -= y_off

    walkable = (hf > PIT + 0.1).astype(np.int8)
    cam_pos = np.array([nx * RES / 2.0, ny * RES / 2.0 - y_off], dtype=np.float32)

    # path along the route centre line
    z_top = top
    z_up2 = top + N_UP2 * h
    z_down = z_up2 - N_DOWN * h
    path = [
        [0.5, 0.0, 0.0],
        [APPROACH, 0.0, 0.0],
        [APPROACH + N_UP1 * d, 0.0, z_top],
        [x_turn, 0.0, z_top],
        [x_turn, half, z_top],
        [x_turn, half + N_UP2 * d, z_up2],
        [x_turn, half + N_UP2 * d + FLAT2, z_up2],
        [x_turn, half + N_UP2 * d + FLAT2 + N_DOWN * d, z_down],
        [x_turn, half + N_UP2 * d + FLAT2 + N_DOWN * d + FLAT3, z_down],
        [x_turn, y_end, z_down + N_UP3 * h],
    ]
    return dict(vertices=vertices, faces=faces, heigthmap=hf,
                walkable_map=walkable, cam_pos=cam_pos,
                x_scale=1.0 / RES, y_scale=1.0 / RES), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = yaml.safe_load(
        open("pacer/data/cfg/pacer_stage3_stairs.yaml", encoding="utf-8"))["env"]["terrain"]
    rng = np.random.default_rng(args.seed)
    os.makedirs("data/paths", exist_ok=True)

    for i in range(args.count):
        h = float(rng.uniform(cfg["stair_riser_min"], cfg["stair_riser_max"]))
        d = float(rng.uniform(cfg["stair_tread_min"], cfg["stair_tread_max"]))
        w = float(rng.choice(WIDTH_BINS, p=WIDTH_PROBS))
        mesh, path = build(h, d, w)
        name = f"s3route_r{int(round(h * 100)):02d}_t{int(round(d * 100)):02d}" \
               f"_w{int(round(w * 10)):02d}_{i}"
        with open(f"data/mesh/{name}.pkl", "wb") as fh:
            pickle.dump(mesh, fh)
        payload = {
            "name": name,
            "note": "approach -> up5 -> flat -> 90 turn -> up10 -> flat -> "
                    "down8 -> flat -> up10",
            "path": path,
        }
        with open(f"data/paths/{name}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"{name}  riser={h:.3f} tread={d:.3f} W={w:.2f} "
              f"top={N_UP1*h:.2f} peak={(N_UP1+N_UP2)*h:.2f} "
              f"end={(N_UP1+N_UP2-N_DOWN+N_UP3)*h:.2f} verts={len(mesh['vertices'])}")


if __name__ == "__main__":
    main()
