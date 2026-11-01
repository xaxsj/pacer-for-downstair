"""比对 dump 里的观测侧 root 与仿真侧 root，并检查刚体分布。

用法：
    python script/compare_sim_obs.py --states X.pkl
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def q_to_deg_up(q):
    """四元数 (x,y,z,w) 下 body +Z 的世界方向与竖直方向的夹角。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    v = np.array([0.0, 0.0, 1.0])
    t = 2.0 * np.cross(xyz, v)
    d = v + w * t + np.cross(xyz, t)
    return float(np.degrees(np.arccos(np.clip(d[2], -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    print("字段:", ", ".join(sorted(d.keys())))

    obs = np.asarray(d["root_states"])[:, args.env]
    if "sim_root_states" not in d:
        print("\n没有 sim_root_states 字段（补丁没生效或用了旧 dump）")
        return 1
    sim = np.asarray(d["sim_root_states"])[:, args.env]
    T = obs.shape[0]
    print("T=%d  obs shape=%s  sim shape=%s" % (T, obs.shape, sim.shape))

    print("\n 帧    obs(x,y,z)                  sim(x,y,z)                  位置差")
    for i in list(range(0, min(8, T))) + list(range(8, T, max(1, T // 8))):
        dp = np.linalg.norm(obs[i, 0:3] - sim[i, 0:3])
        print(" %3d  %+7.3f %+7.3f %+7.3f    %+7.3f %+7.3f %+7.3f    %.4f" % (
            i, obs[i, 0], obs[i, 1], obs[i, 2], sim[i, 0], sim[i, 1], sim[i, 2], dp))

    print("\n=== body +Z 与竖直方向夹角（0=直立，90=横躺）===")
    print(" 帧   obs_root   sim_root")
    for i in list(range(0, min(8, T))) + list(range(8, T, max(1, T // 8))):
        print(" %3d   %6.1f     %6.1f" % (i, q_to_deg_up(obs[i, 3:7]), q_to_deg_up(sim[i, 3:7])))

    a_obs = np.mean([q_to_deg_up(obs[i, 3:7]) for i in range(T)])
    a_sim = np.mean([q_to_deg_up(sim[i, 3:7]) for i in range(T)])
    print("\n 全段均值: obs_root %.1f deg   sim_root %.1f deg" % (a_obs, a_sim))
    dmax = np.abs(obs[:, 0:7] - sim[:, 0:7]).max()
    print(" 位置/四元数最大偏差: %.4f" % dmax)
    print(" 结论: %s" % ("一致" if dmax < 1e-3 else "★ 不一致 —— 观测与仿真已解耦"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
