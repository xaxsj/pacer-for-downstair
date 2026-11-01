"""按 docs/初始位姿设置指南.md 的口径测朝向。

指南 第六节约定：
    方法 B（数值）: body +X = quat_apply(root_rot, [1,0,0]) 的 atan2
    标定表用:      视觉正面 = 脚趾 − 脚踝（且只有双脚落地时才可靠）

两者相差约 90°~180°（SMPL base rotation，见指南陷阱 2），不能混用。

本脚本同时输出这两个量，并打印轨迹方向，用于判断"是否对准轨迹"。

用法：
    python script/check_heading.py --states X.pkl --mesh data/mesh/stairs05_neg.pkl
    python script/check_heading.py --states X.pkl --list-bodies   # 先看刚体名
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


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--asset", default="assets/smpl_humanoid.xml")
    ap.add_argument("--mesh", default="")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--list-bodies", action="store_true")
    ap.add_argument("--floor", type=float, default=0.75, help="该处地面高度，用于判断脚是否落地")
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        sd = pickle.load(fh)
    root = np.asarray(sd["root_states"])[args.frame, 0]
    dof = np.asarray(sd["dof_pos"])[args.frame, 0]

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0)
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, os.path.dirname(os.path.abspath(args.asset)),
                           os.path.basename(args.asset), opts)

    names = list(gym.get_asset_rigid_body_names(asset))
    if args.list_bodies:
        print("刚体名 (%d):" % len(names))
        for i, n in enumerate(names):
            print("  %2d  %s" % (i, n))
        return 0

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
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)

    rb = rb_t.cpu().numpy()[:, 0:3]

    # --- 方法 B：body +X ---
    q = root[3:7]
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.cross([x, y, z], v)
    d = v + w * t + np.cross([x, y, z], t)
    body_x = float(np.degrees(np.arctan2(d[1], d[0])))

    # --- 视觉正面：脚趾 − 脚踝 ---
    def bid(*keys):
        for k in keys:
            for i, nm in enumerate(names):
                if k.lower() in nm.lower():
                    return i
        return None

    lt, rt = bid("L_Toe", "L_Foot", "L_Ankle"), bid("R_Toe", "R_Foot", "R_Ankle")
    la, ra = bid("L_Ankle"), bid("R_Ankle")
    print("刚体索引: L_Toe=%s R_Toe=%s L_Ankle=%s R_Ankle=%s" % (lt, rt, la, ra))
    print()
    print("  方法 B  body +X 航向      = %+7.1f deg" % wrap180(body_x))

    for tag, t_id, a_id in (("左", lt, la), ("右", rt, ra)):
        if t_id is None or a_id is None or t_id == a_id:
            print("  %s脚: 找不到脚趾/脚踝刚体" % tag)
            continue
        vec = rb[t_id] - rb[a_id]
        ang = float(np.degrees(np.arctan2(vec[1], vec[0])))
        print("  视觉正面(%s: %s - %s) = %+7.1f deg   (z分量 %.3f，双脚离地时不可靠)"
              % (tag, names[t_id], names[a_id], wrap180(ang), vec[2]))

    # 脚是否落地（与该处地面高度比较，不是与 0 比）
    zs = rb[:, 2]
    grounded = zs.min() - args.floor < 0.12
    print("\n  最低刚体 z = %.3f，地面 = %.3f，离地 %.3f  (落地: %s)"
          % (zs.min(), args.floor, zs.min() - args.floor, "是" if grounded else "否，悬空"))

    # --- 轨迹方向 ---
    if "traj_verts" in sd:
        tv = np.asarray(sd["traj_verts"], dtype=np.float64)
        dxy = tv[-1, :2] - tv[0, :2]
        print("  轨迹方向 (首->末)        = %+7.1f deg" % wrap180(np.degrees(np.arctan2(dxy[1], dxy[0]))))
    print()
    print("  指南口径：标定表用「视觉正面」，判据是它对准轨迹方向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
