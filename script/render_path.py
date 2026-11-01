"""Render a PACER state dump with Isaac Gym and overlay the trajectory.

Overlays:
  * magenta polyline = traj_verts (world-frame path)
  * yellow dots      = the 10 waypoints the policy sees at the current frame
  * white circle     = character root

Usage:
    python script/render_path.py --states X.pkl --out Y.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/isaacgym_pkg/isaacgym/python"))
from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s.dot(eye)
    m[1, 3] = -u.dot(eye)
    m[2, 3] = f.dot(eye)
    return m


def perspective(fov_deg, aspect, near=0.05, far=1000.0):
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def project(points, view, proj, width, height):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    homo = np.concatenate([pts, np.ones((n, 1))], axis=1)
    clip = homo @ (proj @ view).T
    w = clip[:, 3]
    ok = w > 1e-6
    ww = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / ww[:, None]
    u = (ndc[:, 0] * 0.5 + 0.5) * width
    v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return u, v, ok


def load_states(path, env=0):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])
    dof = np.asarray(d["dof_pos"])
    if root.ndim == 3:
        root = root[:, env]
    if dof.ndim == 3:
        dof = dof[:, env]
    tv = d.get("traj_verts")
    tv = np.asarray(tv) if tv is not None else None
    ts = d.get("traj_samples")
    if ts is not None:
        ts = np.asarray(ts)
        if ts.ndim == 4:
            ts = ts[:, env]
    return root, dof, tv, ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--asset", default="assets/smpl_humanoid.xml")
    ap.add_argument("--mesh", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    root, dof, traj_verts, traj_samples = load_states(args.states)
    T = root.shape[0] if args.frames <= 0 else min(args.frames, root.shape[0])
    print("[render] states=%s frames=%d traj=%s" % (
        args.states, T, "yes" if traj_verts is not None else "no"))

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = True
    sp.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("create_sim failed")

    if args.mesh:
        with open(args.mesh, "rb") as fh:
            md = pickle.load(fh)
        v = np.asarray(md["vertices"], dtype=np.float32)
        f = np.asarray(md["faces"], dtype=np.uint32)
        tm = gymapi.TriangleMeshParams()
        tm.nb_vertices = v.shape[0]
        tm.nb_triangles = f.shape[0]
        tm.transform.p = gymapi.Vec3(0, 0, 0)
        tm.static_friction = 1.0
        tm.dynamic_friction = 1.0
        gym.add_triangle_mesh(sim, v.flatten(order="C"), f.flatten(order="C"), tm)

    asset_root = os.path.dirname(os.path.abspath(args.asset))
    asset_file = os.path.basename(args.asset)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, asset_root, asset_file, opts)

    s = 4.0
    env = gym.create_env(sim, gymapi.Vec3(-s, -s, 0.0), gymapi.Vec3(s, s, s * 2), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(root[0, 0]), float(root[0, 1]), float(root[0, 2]))
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "humanoid", 0, 0)

    cp = gymapi.CameraProperties()
    cp.width = args.width
    cp.height = args.height
    cam = gym.create_camera_sensor(env, cp)

    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    dof_tensor = gym.acquire_dof_state_tensor(sim)
    root_t = gymtorch.wrap_tensor(root_tensor)
    dof_t = gymtorch.wrap_tensor(dof_tensor)
    idx = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)

    cam_off = np.array([1.5, -7.5, 3.0])
    cam_look = np.array([1.5, 0.5, -0.2])
    fov = 60.0
    aspect = args.width / float(args.height)

    for i in range(T):
        rp = root[i]
        root_t[idx, 0:3] = torch.tensor(rp[0:3], device=root_t.device, dtype=root_t.dtype)
        root_t[idx, 3:7] = torch.tensor(rp[3:7], device=root_t.device, dtype=root_t.dtype)
        root_t[idx, 7:13] = 0.0
        n = min(int(dof.shape[-1]), dof_t.shape[0])
        dof_t[:, 0] = 0.0
        dof_t[:n, 0] = torch.tensor(dof[i, :n], device=dof_t.device, dtype=dof_t.dtype)
        dof_t[:, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, root_tensor)
        gym.set_dof_state_tensor(sim, dof_tensor)

        base = rp[0:3]
        eye = base + cam_off
        tgt = base + cam_look
        gym.set_camera_location(cam, env, gymapi.Vec3(*eye), gymapi.Vec3(*tgt))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        color = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
        img = color.reshape(color.shape[0], -1, 4)[..., :3].copy()

        view = look_at(eye, tgt)
        proj = perspective(fov, aspect)

        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)

        if traj_verts is not None:
            u, v, ok = project(traj_verts, view, proj, args.width, args.height)
            if i == 0:
                print("  [debug] first5 u=", np.round(u[:5], 1).tolist(),
                      " v=", np.round(v[:5], 1).tolist(), " ok=", ok[:5].tolist())
            pts = [(float(u[k]), float(v[k])) for k in range(len(u))
                   if ok[k] and -3000 < u[k] < 5000 and -3000 < v[k] < 5000]
            if len(pts) >= 2:
                dr.line(pts, fill=(255, 0, 200), width=4)
            for k in range(0, len(u), 10):
                if ok[k]:
                    dr.ellipse([u[k] - 5, v[k] - 5, u[k] + 5, v[k] + 5],
                               outline=(255, 0, 200), width=2)

        if traj_samples is not None:
            ss = traj_samples[min(i, traj_samples.shape[0] - 1)]
            u, v, ok = project(ss, view, proj, args.width, args.height)
            for k in range(len(u)):
                if ok[k]:
                    dr.ellipse([u[k] - 7, v[k] - 7, u[k] + 7, v[k] + 7],
                               fill=(255, 220, 0))

        u, v, ok = project(rp[0:3][None, :], view, proj, args.width, args.height)
        if ok[0]:
            dr.ellipse([u[0] - 10, v[0] - 10, u[0] + 10, v[0] + 10],
                       outline=(255, 255, 255), width=3)

        dr.text((20, 20), "frame %d  root=(%.2f,%.2f,%.2f)" % (i, rp[0], rp[1], rp[2]),
                fill=(255, 255, 255))

        writer.append_data(np.asarray(pil))
        if i % 30 == 0:
            print("  frame %d/%d" % (i, T))

    writer.close()
    print("[render] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
