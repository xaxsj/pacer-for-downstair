"""比对「碰撞网格的真实表面高度」与「PACER 高度图给的高度」。

这是 MeshTerrain 一致性检查的通用版：不写死任何地形参数，
直接从 pkl 里读 heigthmap / x_scale / y_scale / cam_pos，用与
pacer/env/tasks/humanoid_pedestrain_terrain.py:world_points_to_map 相同的公式
把世界坐标换成高度图索引，再和小球实测的碰撞面对比。

用法：
    python script/check_collision_vs_heightmap.py --mesh data/mesh/stairs05_neg.pkl
    python script/check_collision_vs_heightmap.py --mesh ... --xs "-1.0,-0.5,0.0" --y 0.0

判据：
    diff = 实测碰撞面 − 高度图  接近 0  -> 一致
    diff ≈ −heightmap 值           -> 碰撞面缺失（小球落到地面/掉下去）
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/isaacgym_pkg/isaacgym/python")
from isaacgym import gymapi  # noqa: E402


def hm_sample(hm, cam_pos, x_scale, y_scale, x, y):
    """复刻 world_points_to_map + 一次最近邻采样。"""
    xs, ys = hm.shape
    x_c = (xs / 2.0) / x_scale
    y_c = (ys / 2.0) / y_scale
    px = (x - (cam_pos[0] - x_c)) * x_scale
    py = (y - (cam_pos[1] - y_c)) * y_scale
    px = int(np.clip(round(px), 0, xs - 1))
    py = int(np.clip(round(py), 0, ys - 1))
    return float(hm[px, py]), px, py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--xs", default="", help="逗号分隔的 x 列表；空=自动扫描")
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        d = pickle.load(fh)
    hm = np.asarray(d["heigthmap"], dtype=np.float64)
    cam_pos = np.asarray(d["cam_pos"], dtype=np.float64)
    x_scale = float(d["x_scale"])
    y_scale = float(d["y_scale"])
    v = np.asarray(d["vertices"], dtype=np.float64)
    print("mesh=%s  heigthmap=%s  x_scale=%.2f  cam_pos=%s" % (
        args.mesh, hm.shape, x_scale, cam_pos.tolist()))
    print("vertices bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]" % (
        v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
        v[:, 2].min(), v[:, 2].max()))

    if args.xs:
        xs = [float(s) for s in args.xs.split(",")]
    else:
        xs = [round(float(t), 2) for t in np.arange(v[:, 0].min() + 0.1, v[:, 0].max(), 0.2)]

    # ---- 仿真 ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = args.gpu
    sp.physx.use_gpu = args.gpu
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("create_sim failed")

    md = gymapi.TriangleMeshParams()
    fv = np.asarray(d["vertices"], dtype=np.float32)
    ff = np.asarray(d["faces"], dtype=np.uint32)
    md.nb_vertices = fv.shape[0]
    md.nb_triangles = ff.shape[0]
    md.transform.p = gymapi.Vec3(0, 0, 0)
    md.static_friction = 1.0
    md.dynamic_friction = 1.0
    gym.add_triangle_mesh(sim, fv.flatten(order="C"), ff.flatten(order="C"), md)

    env = gym.create_env(sim, gymapi.Vec3(v[:, 0].min() - 2, -3, -1),
                         gymapi.Vec3(v[:, 0].max() + 2, 3, 8), 1)
    RADIUS = 0.03
    opts = gymapi.AssetOptions(); opts.density = 1000.0
    sphere = gym.create_sphere(sim, RADIUS, opts)

    actors, refs = [], []
    for x in xs:
        h, _, _ = hm_sample(hm, cam_pos, x_scale, y_scale, x, args.y)
        pose = gymapi.Transform(); pose.r = gymapi.Quat(0, 0, 0, 1)
        pose.p = gymapi.Vec3(float(x), float(args.y), float(h + 0.30))
        actors.append(gym.create_actor(env, sphere, pose, "p_%+.2f" % x, 0, 0))
        refs.append(h)

    gym.prepare_sim(sim)
    for _ in range(args.steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    print("\n y=%.2f   x      高度图    碰撞面实测    diff    索引(x,y)" % args.y)
    print("  " + "-" * 66)
    bad = 0
    for x, a, h in zip(xs, actors, refs):
        st = gym.get_actor_rigid_body_states(env, a, gymapi.STATE_POS)
        z = float(st["pose"]["p"]["z"][0]) - RADIUS
        _, px, py = hm_sample(hm, cam_pos, x_scale, y_scale, x, args.y)
        diff = z - h
        flag = ""
        if abs(diff) > 0.05:
            flag = "  <== 不一致"
            bad += 1
        print("      %+7.3f  %8.3f   %8.3f   %+7.3f   (%d,%d)%s" % (x, h, z, diff, px, py, flag))

    print("\n不一致点数: %d / %d" % (bad, len(xs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
