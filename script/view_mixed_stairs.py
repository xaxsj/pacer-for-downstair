#!/usr/bin/env python3
"""Minimal Isaac Gym viewer for data/mesh/mixed_stairs.pkl.

Run inside WSL where Isaac Gym is available:
  python script/view_mixed_stairs.py
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from isaacgym import gymapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/mesh/mixed_stairs.pkl")
    ap.add_argument("--cam_pos", default="12.0,-14.0,9.0")
    ap.add_argument("--cam_tgt", default="4.7,4.1,0.8")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        d = pickle.load(fh)
    verts = np.asarray(d["vertices"], dtype=np.float32)
    faces = np.asarray(d["faces"], dtype=np.int32)

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = verts.shape[0]
    tm.nb_triangles = faces.shape[0]
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    tm.restitution = 0.0
    gym.add_triangle_mesh(sim, verts.flatten(order="C"), faces.flatten(order="C"), tm)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("failed to create Isaac Gym viewer")

    cp = [float(v) for v in args.cam_pos.split(",")]
    ct = [float(v) for v in args.cam_tgt.split(",")]
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*cp), gymapi.Vec3(*ct))

    print("[view_mixed_stairs] mesh:", args.mesh)
    print("  vertices", verts.shape, "faces", faces.shape)
    print("  close the window to exit")

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
