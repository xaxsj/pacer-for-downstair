"""量出「角色 vs 时间参考点」的沿路径进度差（超前为正，落后为负）。

原理：
    tar_pos = traj_gen.calc_pos(t)          # 按时间线性推进顶点索引
    traj_verts 是等距重采样后的 101 个顶点，总时长 = episodeLength * dt = 10 s
    => 参考点弧长进度 s_ref(t) = (t / 10 s) * L
    角色弧长进度 s_char(t) = 把角色 XY 投影到路径上

用法：
    python script/check_tracking.py --states X.pkl [--episode 10.0]
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def arc_lengths(v):
    d = np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def project_to_path(v, s, xy):
    """把 xy 投影到折线，返回弧长坐标。"""
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
    ap.add_argument("--episode", type=float, default=10.0, help="traj_verts 覆盖的总时长 (s)")
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    tv = np.asarray(d["traj_verts"], dtype=np.float64)
    root = np.asarray(d["root_states"])[:, args.env]
    dt = float(np.asarray(d.get("dt", 0.0333)).ravel()[0])
    T = root.shape[0]

    s = arc_lengths(tv)
    L = float(s[-1])
    print("traj_verts: %d 顶点, 总弧长 %.3f m, 覆盖 %.2f s -> 参考速度 %.3f m/s"
          % (len(tv), L, args.episode, L / args.episode))
    print("dt = %.5f s, frames = %d (%.1f s)" % (dt, T, T * dt))

    print("\n  帧   时间s   角色进度  参考进度    差(超前+)   横向偏移")
    rows = []
    for i in range(T):
        t = i * dt
        s_ref = min(L, L * t / args.episode)
        s_ch, lat = project_to_path(tv, s, root[i, 0:2])
        rows.append((i, t, s_ch, s_ref, s_ch - s_ref, lat))
        if i % max(1, T // 20) == 0 or i == T - 1:
            print("  %4d  %6.2f  %8.3f  %8.3f  %+10.3f  %8.3f"
                  % (i, t, s_ch, s_ref, s_ch - s_ref, lat))

    diffs = np.array([r[4] for r in rows])
    lats = np.array([r[5] for r in rows])
    print("\n=== 汇总 ===")
    print("  进度差: 均值 %+.3f m, 最大超前 %+.3f m, 最大落后 %+.3f m"
          % (diffs.mean(), diffs.max(), diffs.min()))
    print("  横向偏移: 均值 %.3f m, 最大 %.3f m" % (lats.mean(), lats.max()))
    print("  超过 +4 m 的帧: %d / %d  (tar_fail 阈值 4 m)"
          % (int((diffs > 4.0).sum()), T))
    # 角色平均速度 = 净弧长进度 / 时长
    print("  角色平均速度(沿路径): %.3f m/s   参考速度: %.3f m/s"
          % (rows[-1][2] / (T * dt), L / args.episode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
