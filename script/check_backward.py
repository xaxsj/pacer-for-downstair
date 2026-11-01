"""检查角色是否出现「往回走」以及它和重置帧的关系。

对每帧算：
    v_along(t) = d(沿路径弧长)/dt        负值 = 沿路径倒退（往回走）
    heading(t) = body +X 的朝向角
并打印重置帧附近的窗口。

用法：
    python script/check_backward.py --states X.pkl [--window 25]
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


def quat_forward(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.cross(xyz, v)
    d = v + w * t + np.cross(xyz, t)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    tv = np.asarray(d["traj_verts"], dtype=np.float64)
    root = np.asarray(d["root_states"])[:, args.env]
    dt = float(np.asarray(d.get("dt", 0.0333)).ravel()[0])
    T = root.shape[0]
    s = arc(tv)

    pos = np.array([proj(tv, s, root[f, 0:2])[0] for f in range(T)])
    v = np.zeros(T)
    v[1:] = np.diff(pos) / dt
    head = np.array([quat_forward(root[f, 3:7]) for f in range(T)])
    z = root[:, 2]
    resets = list(np.where(np.diff(z) > 0.4)[0] + 1)

    print("=== %s ===" % args.states)
    print("重置帧:", resets)
    print("沿路径速度: 均值 %+.3f m/s | 最小 %+.3f | 最大 %+.3f"
          % (v[1:].mean(), v[1:].min(), v[1:].max()))

    neg = v < -0.05
    print("出现倒退(<-0.05 m/s)的帧: %d / %d (%.0f%%)"
          % (int(neg.sum()), T - 1, 100.0 * neg.sum() / (T - 1)))
    if neg.any():
        segs = []
        run = None
        for f in range(1, T):
            if neg[f]:
                run = [f, f] if run is None else [run[0], f]
            else:
                if run:
                    segs.append(tuple(run)); run = None
        if run:
            segs.append(tuple(run))
        print("倒退片段 (起, 止) 共 %d 段:" % len(segs))
        for a, b in segs[:14]:
            print("    帧 %3d-%3d (%.2f-%.2f s)  最快倒退 %+.2f m/s"
                  % (a, b, a * dt, b * dt, v[a:b + 1].min()))
        if resets:
            near = sum(1 for a, b in segs if any(abs(r - b) <= 5 or abs(r - a) <= 5 for r in resets))
            print("其中紧邻重置帧(±5 帧)的倒退段: %d / %d" % (near, len(segs)))

    print("\n=== 航向与倒退（每 %d 帧采样）===" % max(1, T // 18))
    print("  帧   时间s   航向deg   沿路径速度")
    for f in range(0, T, max(1, T // 18)):
        flag = "  <== 倒退" if v[f] < -0.05 else ""
        print("  %4d %6.2f  %+8.1f   %+7.3f%s" % (f, f * dt, head[f], v[f], flag))

    if resets:
        print("\n=== 重置前 %d 帧窗口 ===" % args.window)
        for r in resets[:6]:
            a = max(0, r - args.window)
            print("  重置@%d:" % r)
            for f in range(a, min(T, r + 2)):
                print("     f%3d  z=%.2f  航向%+7.1f  v=%+6.3f" % (f, z[f], head[f], v[f]))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
