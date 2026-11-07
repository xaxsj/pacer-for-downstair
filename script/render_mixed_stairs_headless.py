#!/usr/bin/env python3
"""Headless Isaac Gym render of the mixed stair mesh (no viewer, no policy).

Requires Python 3.8 + Isaac Gym bindings. Run under Xvfb on a headless server:
  xvfb-run -a -s "-screen 0 1280x720x24" python script/render_mixed_stairs_headless.py
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from isaacgym import gymapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/mesh/mixed_stairs.pkl")
    ap.add_argument("--out", default="experiment_result/mixed_stairs/render.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--graphics_id", type=int, default=0)
    ap.add_argument("--cam_pos", default="13.0,-15.0,10.0")
    ap.add_argument("--cam_tgt", default="4.7,4.1,0.8")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        d = pickle.load(fh)
    verts = np.asarray(d["vertices"], dtype=np.float32)
    faces = np.asarray(d["faces"], dtype=np.uint32)

    gym = gymapi.acquire_gym()

    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0

    sim = gym.create_sim(0, args.graphics_id, gymapi.SIM_PHYSX, sp)

    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = verts.shape[0]
    tm.nb_triangles = faces.shape[0]
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    tm.restitution = 0.0
    gym.add_triangle_mesh(sim, verts.flatten(order="C"), faces.flatten(order="C"), tm)

    lower = gymapi.Vec3(-20.0, -20.0, 0.0)
    upper = gymapi.Vec3(20.0, 20.0, 10.0)
    env = gym.create_env(sim, lower, upper, 1)

    props = gymapi.CameraProperties()
    props.width = args.width
    props.height = args.height
    cam = gym.create_camera_sensor(env, props)
    if cam < 0:
        raise RuntimeError("failed to create camera sensor (need a graphics device; run under xvfb-run)")

    cp = [float(v) for v in args.cam_pos.split(",")]
    ct = [float(v) for v in args.cam_tgt.split(",")]
    gym.set_camera_location(cam, env, gymapi.Vec3(*cp), gymapi.Vec3(*ct))

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    img = np.asarray(gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR))
    if img.ndim == 2 and img.shape[1] == args.width * 4:
        img = img.reshape(args.height, args.width, 4)
    elif img.ndim == 1 and img.size == args.height * args.width * 4:
        img = img.reshape(args.height, args.width, 4)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype(np.uint8)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    from PIL import Image
    Image.fromarray(img).save(args.out)

    print("[render_mixed_stairs] mesh:", args.mesh)
    print("  vertices", verts.shape, "faces", faces.shape)
    print("  image", img.shape, "range", int(img.min()), int(img.max()))
    print("  saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
