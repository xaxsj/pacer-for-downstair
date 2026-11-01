"""从 dump 反解真实参数：参考点速度、真实采样间隔、角色速度。

事实：_fetch_traj_samples 的第 0 个球就是 calc_pos(t) —— 也就是 tar_fail 用的参考点。
      第 k 个球是 calc_pos(t + k * traj_sample_timestep)。

所以：
    v_ref   = d(arc(sample0)) / dt            # 参考点真实速度
    dt_eff  = Δarc(sample0 -> sample1) / v_ref  # 真实采样间隔

用法：
    python script/check_traj_speed.py --states X.pkl
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def arc(v):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1))])


def proj(v, s, xy):
    best_s, best_d = 0.0, 1e9
    for i in range(len(v) - 1):
        a, b = v[i, :2], v[i + 1, :2]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = float(np.clip((xy - a) @ ab / L2, 0.0, 1.0))
        p = a + t * ab
        d = float(np.linalg.norm(xy - p))
        if d < best_d:
            best_d, best_s = d, s[i] + t * (s[i + 1] - s[i])
    return best_s, best_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    tv = np.asarray(d["traj_verts"], dtype=np.float64)
    ts = np.asarray(d["traj_samples"])[:, args.env]
    root = np.asarray(d["root_states"])[:, args.env]
    dt = float(np.asarray(d.get("dt", 0.0333)).ravel()[0])
    T = ts.shape[0]
    s = arc(tv)
    L = float(s[-1])
    print("path XY length = %.3f m, dt = %.5f s, frames = %d (%.1f s)" % (L, dt, T, T * dt))

    # 每个球的弧长轨迹
    A = np.array([[proj(tv, s, ts[f, k, 0:2])[0] for k in range(ts.shape[1])] for f in range(T)])

    # 参考速度：sample0 的弧长随时间增长（跳过被夹住的尾段）
    a0 = A[:, 0]
    t = np.arange(T) * dt
    moving = (a0 < L - 0.15) & (np.arange(T) > 0)
    if moving.sum() > 10:
        slope = np.polyfit(t[moving], a0[moving], 1)[0]
    else:
        slope = float("nan")
    print("参考点速度 v_ref = %.3f m/s  (由 sample0 弧长对时间线性拟合, 用 %d 帧)" % (slope, int(moving.sum())))

    # 真实采样间隔：相邻球弧长差 / v_ref
    gaps = []
    for f in range(0, T, max(1, T // 20)):
        row = A[f]
        if row[-1] >= L - 0.2:
            continue  # 尾部被夹住的行不能用
        dk = np.diff(row)
        gaps.append(dk.mean())
    if gaps:
        gap = float(np.mean(gaps))
        print("相邻两球弧长差 = %.3f m" % gap)
        print("真实采样间隔 dt_eff = %.3f s   (间隔 / v_ref)" % (gap / slope))
    print()

    # 角色速度
    sp = np.array([proj(tv, s, root[f, 0:2])[0] for f in range(T)])
    print("角色沿路径速度 = %.3f m/s" % (sp[-1] / ((T - 1) * dt)))
    print()

    print("  帧   角色弧长  sample0(tar_pos)   差(超前+)")
    for f in range(0, T, max(1, T // 15)):
        print("  %4d  %8.3f   %8.3f      %+8.3f" % (f, sp[f], A[f, 0], sp[f] - A[f, 0]))
    diff = sp - A[:, 0]
    print("\n超前/落后: 最大超前 %+.3f m, 最大落后 %+.3f m" % (diff.max(), diff.min()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
