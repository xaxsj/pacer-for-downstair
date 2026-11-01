"""用 Isaac Gym 离线渲染 PACER 导出的状态序列（不依赖 pacer 环境）。

只需要 isaacgym + torch + numpy + imageio。流程：

1. 建 sim（headless，但申请 graphics device 以便相机成像）
2. 把 ``data/mesh/<name>.pkl`` 的三角网格加进 sim（可视 + 碰撞）
3. 载入 SMPL humanoid XML（PACER 生成的那个），fix_base_link=False
4. 每帧把 ``root_states`` / ``dof_pos`` 写进 actor，渲染相机存帧
5. 用 imageio 写 mp4

用法::

    python scripts/render_isaac_states.py \
        --states states.pkl --mesh data/mesh/stairs20.pkl \
        --asset assets/smpl_humanoid.xml --out out.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

# isaacgym 必须先于 torch 导入
sys.path.insert(0, os.path.expanduser("~/isaacgym_pkg"))
from isaacgym import gymapi, gymutil, gymtorch  # noqa: E402
import torch  # noqa: E402


def to_torch(x, device="cuda:0"):
    return torch.from_numpy(np.asarray(x)).to(torch.float32).to(device)


def build_sim(gym, sim_params, width, height):
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 1
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("create_sim failed")
    return sim


def add_ground(gym, sim, mesh_data):
    verts = np.asarray(mesh_data["vertices"], dtype=np.float32)
    faces = np.asarray(mesh_data["faces"], dtype=np.uint32)
    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = verts.shape[0]
    tm.nb_triangles = faces.shape[0]
    tm.transform.p = gymapi.Vec3(0, 0, 0)
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    tm.restitution = 0.0
    gym.add_triangle_mesh(sim, verts.flatten(order="C"),
                          faces.flatten(order="C"), tm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", default="")
    ap.add_argument("--asset", required=True, help="SMPL humanoid XML")
    ap.add_argument("--out", default="")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import imageio.v2 as imageio

    with open(args.states, "rb") as fh:
        data = pickle.load(fh)
    root = np.asarray(data["root_states"])
    dof = np.asarray(data["dof_pos"])
    # 兼容两种 dump 形状：(T,E,13)/(T,E,D) 与 (T,13)/(T,D)
    if root.ndim == 3:
        root = root[:, args.env]
    if dof.ndim == 3:
        dof = dof[:, args.env]
    T = root.shape[0] if args.frames <= 0 else min(args.frames, root.shape[0])
    print(f"[render_isaac] states={args.states} frames={T} dof={dof.shape[-1]}")

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim = build_sim(gym, sim_params, args.width, args.height)

    if args.mesh:
        with open(args.mesh, "rb") as fh:
            add_ground(gym, sim, pickle.load(fh))
        print(f"[render_isaac] mesh={args.mesh}")

    # 载入 SMPL humanoid
    asset_root = os.path.dirname(os.path.abspath(args.asset))
    asset_file = os.path.basename(args.asset)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"load_asset failed: {args.asset}")
    num_dofs = gym.get_asset_dof_count(asset)
    print(f"[render_isaac] asset dofs={num_dofs} (states dof={dof.shape[-1]})")

    spacing = 4.0
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0.0),
                         gymapi.Vec3(spacing, spacing, spacing * 2), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(root[0, 0]), float(root[0, 1]), float(root[0, 2]))
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "humanoid", 0, 0)

    # 相机
    cam_props = gymapi.CameraProperties()
    cam_props.width = args.width
    cam_props.height = args.height
    cam_handle = gym.create_camera_sensor(env, cam_props)

    # 张量
    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    dof_tensor = gym.acquire_dof_state_tensor(sim)
    root_t = gymtorch.wrap_tensor(root_tensor)
    dof_t = gymtorch.wrap_tensor(dof_tensor)

    actor_idx = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    out = args.out or os.path.splitext(args.states)[0] + "_isaac.mp4"
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=None)

    # 相机视角：侧前方，跟随角色
    cam_offset = np.array([1.5, -7.5, 3.0])
    cam_look = np.array([1.5, 0.5, -0.2])

    for i in range(T):
        rp = root[i]
        root_t[actor_idx, 0:3] = to_torch(rp[0:3], root_t.device)
        root_t[actor_idx, 3:7] = to_torch(rp[3:7], root_t.device)  # w,x,y,z
        root_t[actor_idx, 7:13] = 0.0

        n = min(int(dof.shape[-1]), dof_t.shape[0])  # dof_t is (num_dofs, 2)
        pos = to_torch(dof[i, :n], dof_t.device)
        dof_t[:, 0] = 0.0
        dof_t[:n, 0] = pos
        dof_t[:, 1] = 0.0

        gym.set_actor_root_state_tensor(sim, root_tensor)
        gym.set_dof_state_tensor(sim, dof_tensor)

        base = rp[0:3]
        gym.set_camera_location(
            cam_handle, env,
            gymapi.Vec3(float(base[0] + cam_offset[0]),
                        float(base[1] + cam_offset[1]),
                        float(base[2] + cam_offset[2])),
            gymapi.Vec3(float(base[0] + cam_look[0]),
                        float(base[1] + cam_look[1]),
                        float(base[2] + cam_look[2])),
        )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        color = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
        if color is None:
            raise RuntimeError("get_camera_image returned None")
        frame = color.reshape(color.shape[0], -1, 4)[..., :3]
        writer.append_data(frame)
        if i % 30 == 0:
            print(f"  frame {i}/{T}")

    writer.close()
    print(f"[render_isaac] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
