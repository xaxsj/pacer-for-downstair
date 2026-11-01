"""人偶静置跌落测试：只放地形 + 人偶，不给任何策略/驱动，只受重力。

目的：把「人偶与地形的碰撞」和「策略/驱动」彻底分开。
      若人偶最终稳在平台高度 -> 碰撞正常，塌下是策略造成的
      若人偶穿过平台         -> 碰撞配置有问题，与策略无关

用法：
    python script/drop_test.py --mesh data/mesh/stairs05_neg.pkl \
        --states output/recordings/states-XXX.pkl --frame 0 --steps 90
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/isaacgym_pkg/isaacgym/python")
from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--asset", default="assets/smpl_humanoid.xml")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--floor", type=float, default=0.75)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        sd = pickle.load(fh)
    root = np.asarray(sd["root_states"])[args.frame, 0]
    dof = np.asarray(sd["dof_pos"])[args.frame, 0]

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("create_sim failed")

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    v = np.asarray(md["vertices"], dtype=np.float32)
    f = np.asarray(md["faces"], dtype=np.uint32)
    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = v.shape[0]
    tm.nb_triangles = f.shape[0]
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    gym.add_triangle_mesh(sim, v.flatten(order="C"), f.flatten(order="C"), tm)
    print("地形已加载: %d 顶点 %d 面, z 范围 %.2f..%.2f" % (
        v.shape[0], f.shape[0], v[:, 2].min(), v[:, 2].max()))

    root_dir = os.path.dirname(os.path.abspath(args.asset))
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, root_dir, os.path.basename(args.asset), opts)
    print("人偶资产: %d 刚体, %d dof" % (gym.get_asset_rigid_body_count(asset),
                                        gym.get_asset_dof_count(asset)))

    env = gym.create_env(sim, gymapi.Vec3(-6, -3, -1), gymapi.Vec3(2, 3, 6), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(root[0]), float(root[1]), float(root[2]))
    pose.r = gymapi.Quat(float(root[3]), float(root[4]), float(root[5]), float(root[6]))
    actor = gym.create_actor(env, asset, pose, "humanoid", 0, 0)

    gym.prepare_sim(sim)
    root_t = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_t = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    rb_t = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    idx = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    n = min(int(dof.shape[-1]), dof_t.shape[0])
    dof_t[:n, 0] = torch.tensor(dof[:n], dtype=dof_t.dtype)
    dof_t[:n, 1] = 0.0
    gym.set_dof_state_tensor(sim, gym.acquire_dof_state_tensor(sim))

    print("\n初始: root z=%.3f  平台=%.3f" % (root[2], args.floor))
    print(" step    root_z    最低刚体z   相对平台")
    for s in range(args.steps + 1):
        if s % 10 == 0 or s == args.steps:
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            rz = float(root_t[idx, 2])
            rb = rb_t.cpu().numpy()
            lo = float(rb[:, 2].min())
            print("  %4d  %8.3f   %8.3f   %+8.3f" % (s, rz, lo, lo - args.floor))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
