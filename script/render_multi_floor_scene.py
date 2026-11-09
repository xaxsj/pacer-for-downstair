"""Orbit render of the layered multi-floor building (no character, no policy).

Loads the stacked trimesh, puts a camera on a spiral around it and writes an
mp4 with cv2.  The building has no walls, so an exterior orbit shows the
interior directly.

    xvfb-run -a -s "-screen 0 1600x900x24" \
        python script/render_multi_floor_scene.py --mesh data/mesh/mf3.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from isaacgym import gymapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/mesh/mf3.pkl")
    ap.add_argument("--out", default="experiment_result/multi_floor/scene_orbit.mp4")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--radius0", type=float, default=19.0)
    ap.add_argument("--radius1", type=float, default=11.5)
    ap.add_argument("--z0", type=float, default=11.0)
    ap.add_argument("--z1", type=float, default=5.5)
    ap.add_argument("--graphics_id", type=int, default=0)
    ap.add_argument("--tgt", default="auto")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        d = pickle.load(fh)
    verts = np.asarray(d["vertices"], dtype=np.float32)
    faces = np.asarray(d["faces"], dtype=np.uint32)

    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    if args.tgt == "auto":
        tgt = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2] * 0.48])
    else:
        tgt = np.array([float(v) for v in args.tgt.split(",")])
    print("[scene] bounds %s .. %s   target %s" % (
        np.round(lo, 2), np.round(hi, 2), np.round(tgt, 2)))

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
    if sim is None:
        raise RuntimeError("create_sim failed (need a graphics device; run under xvfb)")

    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = verts.shape[0]
    tm.nb_triangles = faces.shape[0]
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    tm.restitution = 0.0
    tm.transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.add_triangle_mesh(sim, verts.flatten(order="C"), faces.flatten(order="C"), tm)

    lower = gymapi.Vec3(float(lo[0]) - 5.0, float(lo[1]) - 5.0, float(lo[2]) - 5.0)
    upper = gymapi.Vec3(float(hi[0]) + 5.0, float(hi[1]) + 5.0, float(hi[2]) + 8.0)
    env = gym.create_env(sim, lower, upper, 1)

    props = gymapi.CameraProperties()
    props.width = args.width
    props.height = args.height
    props.horizontal_fov = 55.0
    cam = gym.create_camera_sensor(env, props)
    if cam < 0:
        raise RuntimeError("failed to create camera sensor")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    import cv2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))

    for f in range(args.frames):
        t = (f / max(args.frames - 1, 1)) ** 1.0
        ang = 2.0 * np.pi * args.turns * (f / max(args.frames, 1)) + np.pi * 0.25
        r = args.radius0 + (args.radius1 - args.radius0) * t
        z = args.z0 + (args.z1 - args.z0) * t
        cp = gymapi.Vec3(tgt[0] + r * np.cos(ang), tgt[1] + r * np.sin(ang), z)
        gym.set_camera_location(cam, env, cp, gymapi.Vec3(*tgt))

        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = np.asarray(gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR))
        if img.ndim == 1:
            img = img.reshape(args.height, args.width, 4)
        elif img.ndim == 2 and img.shape[1] == args.width * 4:
            img = img.reshape(args.height, args.width, 4)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img.astype(np.uint8))
        writer.write(img)
        if f % 50 == 0:
            print("  frame %3d  cam (%.1f, %.1f, %.1f)  mean %.1f"
                  % (f, cp.x, cp.y, cp.z, img.mean()))

    writer.release()
    print("[scene] vertices %s  faces %s" % (verts.shape, faces.shape))
    print("[scene] saved", args.out, "  %d frames @ %d fps" % (args.frames, args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
