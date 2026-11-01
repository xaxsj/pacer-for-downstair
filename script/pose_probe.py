"""用 Isaac Gym 按 (root 状态 + dof) 反算刚体的真实世界坐标。

用途：dump 里的 rigid_body_pos 可能是被 ZL hack 覆盖过的、坐标系不同的数据，
     不能直接当世界坐标用。本脚本只把 root_rot / root_pos / dof 喂给仿真，
     让 PhysX 用正运动学算出真实刚体位置，再打印。

用法：
    python script/pose_probe.py --states X.pkl --frames 0,3,6,9,12 --floor 0.75

判据：
    脚底 z 明显低于 --floor  -> 穿模（角色陷进平台）
    脚底 z ≈ --floor 且躯干下沉 -> 关节折叠/蹲下
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
    ap.add_argument("--states", required=True)
    ap.add_argument("--asset", default="assets/smpl_humanoid.xml")
    ap.add_argument("--frames", default="0,3,6,9,12")
    ap.add_argument("--floor", type=float, default=0.75, help="该处地面高度")
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])[:, args.env]
    dof = np.asarray(d["dof_pos"])[:, args.env]
    frames = [int(s) for s in args.frames.split(",") if s.strip()]

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)      # 纯运动学：不要让重力干扰
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = True
    sp.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("create_sim failed")

    asset_root = os.path.dirname(os.path.abspath(args.asset))
    asset_file = os.path.basename(args.asset)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, asset_root, asset_file, opts)

    env = gym.create_env(sim, gymapi.Vec3(-4, -4, 0), gymapi.Vec3(4, 4, 4), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "humanoid", 0, 0)

    gym.prepare_sim(sim)
    root_t = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_t = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    rb_t = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    idx = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    print("dof 数 = %d, 刚体数 = %d" % (dof.shape[-1], rb_t.shape[0]))
    print("地面高度 floor = %.3f" % args.floor)
    print()

    for fr in frames:
        rp = root[fr]
        root_t[idx, 0:3] = torch.tensor(rp[0:3], dtype=root_t.dtype)
        root_t[idx, 3:7] = torch.tensor(rp[3:7], dtype=root_t.dtype)
        root_t[idx, 7:13] = 0.0
        n = min(int(dof.shape[-1]), dof_t.shape[0])
        dof_t[:n, 0] = torch.tensor(dof[fr, :n], dtype=dof_t.dtype)
        dof_t[:n, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, gym.acquire_actor_root_state_tensor(sim))
        gym.set_dof_state_tensor(sim, gym.acquire_dof_state_tensor(sim))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)

        rb = rb_t.cpu().numpy()
        p = rb[:, 0:3]
        order = np.argsort(p[:, 2])
        lo2 = order[:2]
        print("frame %2d  root=(%.3f, %.3f, %.3f)" % (fr, rp[0], rp[1], rp[2]))
        print("   最低两个刚体: " + ", ".join(
            "idx%d z=%.3f (相对地面 %+.3f)" % (i, p[i, 2], p[i, 2] - args.floor) for i in lo2))
        print("   最高刚体:     idx%d z=%.3f" % (order[-1], p[order[-1], 2]))
        print("   竖直跨度:     %.3f m   刚体 z 范围 [%.3f, %.3f]" % (
            p[:, 2].max() - p[:, 2].min(), p[:, 2].min(), p[:, 2].max()))
        below = int((p[:, 2] < args.floor - 0.02).sum())
        print("   低于地面的刚体数: %d / %d" % (below, p.shape[0]))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
