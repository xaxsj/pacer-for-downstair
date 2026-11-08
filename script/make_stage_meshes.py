"""Generate stair meshes sampled from a curriculum stage's own distribution.

The existing fixed meshes (wide/narrow/...) are all 0.15-0.18 risers with a
0.30 tread, while Stage 1 trains 0.08-0.12 risers on a 0.40 tread, so its
acceptance was measuring an out-of-distribution scene. This builds meshes from
the same ranges the stage actually samples, with the same route the policy is
trained on: approach -> up flight -> platform -> down flight -> run-out, with a
1.5 m pit outside the corridor (as in the procedural terrain).

Run inside the pacer env (needs isaacgym for the heightfield -> trimesh step):

    python script/make_stage_meshes.py --stage 1 --count 4
"""
import argparse
import pickle

import numpy as np
import yaml
from isaacgym import terrain_utils

RES = 0.05           # horizontal resolution (m)
VSCALE = 0.01        # vertical resolution (m)
APPROACH = 2.0       # flat run-up before the first riser (m)
RUNOUT = 2.5         # flat after the down flight (m)
PLATFORM = 1.5       # platform between the flights (m)
PIT = -1.5           # ground level outside the corridor (m)
MARGIN_Y = 1.0       # extra ground on each side of the corridor (m)


def build(h, d, width, n_up, n_down, seed):
    x_len = APPROACH + n_up * d + PLATFORM + n_down * d + RUNOUT
    half = width / 2.0

    nx = int(round(x_len / RES))
    ny = int(round((width + 2.0 * MARGIN_Y) / RES))

    # Grid coordinates: row j covers y in [j*RES, (j+1)*RES]; the corridor sits
    # at [MARGIN_Y, MARGIN_Y + width]. Heights are evaluated at cell centres so
    # the sampler never lands exactly on a step edge.
    xc = (np.arange(nx) + 0.5) * RES
    yc = (np.arange(ny) + 0.5) * RES

    profile = np.zeros(nx, dtype=np.float32)
    x = APPROACH
    for i in range(n_up):
        profile[(xc >= x) & (xc < x + d)] = (i + 1) * h
        x += d
    top = n_up * h
    profile[(xc >= x) & (xc < x + PLATFORM)] = top
    x += PLATFORM
    for i in range(n_down):
        profile[(xc >= x) & (xc < x + d)] = top - (i + 1) * h
        x += d

    inside = (yc >= MARGIN_Y) & (yc <= MARGIN_Y + width)
    hf = np.full((nx, ny), PIT, dtype=np.float32)
    hf[:, inside] = profile[:, None]

    hf_int = np.rint(hf / VSCALE).astype(np.int16)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        hf_int, RES, VSCALE, slope_threshold=0.75)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    # put the corridor centre at y = 0 (the converter starts the field at 0)
    y_centre = MARGIN_Y + width / 2.0
    vertices[:, 1] -= y_centre

    walkable = (hf > PIT + 0.1).astype(np.int8)
    cam_pos = np.array([nx * RES / 2.0, ny * RES / 2.0 - y_centre],
                       dtype=np.float32)
    return dict(vertices=vertices, faces=faces, heigthmap=hf,
                walkable_map=walkable, cam_pos=cam_pos,
                x_scale=1.0 / RES, y_scale=1.0 / RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(f"pacer/data/cfg/pacer_stage{args.stage}_stairs.yaml",
                              encoding="utf-8"))["env"]["terrain"]
    rng = np.random.default_rng(args.seed + 1000 * args.stage)

    bins = [float(v) for v in cfg["stair_width_bins"]]
    probs = [float(v) for v in cfg["stair_width_probs"]]
    probs = np.array(probs) / np.sum(probs)

    for i in range(args.count):
        h = float(rng.uniform(cfg["stair_riser_min"], cfg["stair_riser_max"]))
        d = float(rng.uniform(cfg["stair_tread_min"], cfg["stair_tread_max"]))
        w = float(rng.choice(bins, p=probs))
        n_up = int(rng.integers(cfg["stair_up_steps_min"],
                               cfg["stair_up_steps_max"] + 1))
        n_down = int(rng.integers(cfg["stair_down_steps_min"],
                                 cfg["stair_down_steps_max"] + 1))
        data = build(h, d, w, n_up, n_down, args.seed + i)
        name = (f"s{args.stage}_r{int(round(h * 100)):02d}"
                f"_t{int(round(d * 100)):02d}_w{int(round(w * 10)):02d}_{i}")
        path = f"data/mesh/{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        print(f"{path:40s} riser={h:.3f} tread={d:.3f} W={w:.2f} "
              f"up={n_up} down={n_down} verts={len(data['vertices'])}")


if __name__ == "__main__":
    main()
