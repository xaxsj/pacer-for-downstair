"""Building-floor scene: a flat storey, or a storey reached by one flight.

Geometry (metres):
    floor slab at z = 0 over the whole scene (the building floor)
    corridor width 3.0 m centred on y = 0
    --flat           : flat floor only (a single storey, no stairs)
    (default)        : approach 2.0 m, 12 steps of 0.15 riser / 0.30 tread
                       (one storey = 1.8 m), then a 2.5 m landing

Material is not stored in the mesh: the loader takes it from the env config
(plane.staticFriction / dynamicFriction / restitution), i.e. the same
1.0 / 1.0 / 0.0 as the training terrain and every other eval mesh.

    python script/make_floor_mesh.py --flat --name floor_flat
"""
import argparse
import json
import pickle

import numpy as np
import yaml
from isaacgym import terrain_utils

RES = 0.05
VSCALE = 0.01
APPROACH = 2.0
N_STEPS = 12
RISER = 0.15
TREAD = 0.30
LANDING = 2.5
WIDTH = 3.0
FLAT_END = 8.0     # used when --flat
END_FLAT = 1.0


def build(flat=False):
    if flat:
        x_len = FLAT_END
    else:
        x_len = APPROACH + N_STEPS * TREAD + LANDING + END_FLAT
    y_len = WIDTH + 2.0

    nx = int(round(x_len / RES))
    ny = int(round(y_len / RES))
    xc = (np.arange(nx) + 0.5) * RES
    yc = (np.arange(ny) + 0.5) * RES

    profile = np.zeros(nx, dtype=np.float32)
    if not flat:
        x = APPROACH
        for i in range(N_STEPS):
            profile[(xc >= x) & (xc < x + TREAD)] = (i + 1) * RISER
            x += TREAD
        profile[xc >= APPROACH + N_STEPS * TREAD] = N_STEPS * RISER

    hf = np.repeat(profile[:, None], ny, axis=1).astype(np.float32)

    hf_int = np.rint(hf / VSCALE).astype(np.int16)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        hf_int, RES, VSCALE, slope_threshold=0.75)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    vertices[:, 1] -= y_len / 2.0          # corridor centred on y = 0

    walkable = np.ones((nx, ny), dtype=np.int8)
    # cam_pos is the grid centre in world XY: x centre, y centre after the shift
    cam_pos = np.array([nx * RES / 2.0, 0.0], dtype=np.float32)
    mesh = dict(vertices=vertices, faces=faces, heigthmap=hf,
                walkable_map=walkable, cam_pos=cam_pos,
                x_scale=1.0 / RES, y_scale=1.0 / RES)

    if flat:
        path = [[0.5, 0.0, 0.0], [x_len - 2.0, 0.0, 0.0]]
        info = dict(top=0.0, mode="flat", x_end=x_len)
    else:
        x_stairs = APPROACH + N_STEPS * TREAD
        top = N_STEPS * RISER
        path = [[0.5, 0.0, 0.0],
                [APPROACH, 0.0, 0.0],
                [x_stairs, 0.0, top],
                [x_stairs + LANDING, 0.0, top]]
        info = dict(top=top, mode="stairs", x_end=x_stairs + LANDING)
    return mesh, path, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="floor_single")
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()

    mesh, path, info = build(flat=args.flat)
    with open(f"data/mesh/{args.name}.pkl", "wb") as fh:
        pickle.dump(mesh, fh)
    note = ("single storey, flat floor (2nd floor corridor)" if args.flat
            else "single storey: floor -> 12 steps (1.8 m) -> landing")
    with open(f"data/paths/{args.name}.json", "w", encoding="utf-8") as fh:
        json.dump({"name": args.name, "note": note, "path": path}, fh, indent=2)

    cfg = yaml.safe_load(
        open("pacer/data/cfg/pacer_stage6_stairs.yaml", encoding="utf-8"))["env"]
    print(f"mesh data/mesh/{args.name}.pkl   verts={len(mesh['vertices'])}  "
          f"mode={info['mode']}  end={info['x_end']:.2f} m")
    print(f"path data/paths/{args.name}.json")
    print(f"material (from cfg): static={cfg['plane']['staticFriction']} "
          f"dynamic={cfg['plane']['dynamicFriction']} "
          f"restitution={cfg['plane']['restitution']}")


if __name__ == "__main__":
    main()
