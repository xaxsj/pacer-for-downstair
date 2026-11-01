"""验证「超前过多会停下来等」的负反馈机制。

对每帧计算：
    gap(t)     = 角色弧长 − 参考点弧长        （超前为正）
    v_char(t)  = 10 帧滑窗速度
并检查：
    - gap 是否有界（不单调增大）
    - v_char 与 gap 的相关性（负相关 = 有负反馈）
    - 是否存在"停等"段（速度接近 0 而 gap 较大）

用法：
    python script/check_waiting.py --states X.pkl
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

    char = np.array([proj(tv, s, root[f, 0:2])[0] for f in range(T)])
    ref = np.array([proj(tv, s, ts[f, 0, 0:2])[0] for f in range(T)])
    gap = char - ref

    # 10 帧滑窗速度
    W = 10
    v = np.full(T, np.nan)
    for f in range(W, T):
        v[f] = (char[f] - char[f - W]) / (W * dt)

    print("=== %s ===" % args.states)
    print("gap: 均值 %+.3f m | 最小 %+.3f | 最大 %+.3f | 首末 %+.3f -> %+.3f"
          % (gap.mean(), gap.min(), gap.max(), gap[0], gap[-1]))
    print("gap 在窗口 300-599 的均值 %+.3f, 标准差 %.3f"
          % (gap[T // 2:].mean(), gap[T // 2:].std()))
    sl = np.polyfit(np.arange(T) * dt, gap, 1)[0]
    print("gap 线性趋势斜率 = %+.4f m/s （≈0 表示已稳定）" % sl)

    ok = ~np.isnan(v)
    corr = np.corrcoef(v[ok], gap[ok])[0, 1]
    print("速度 vs gap 相关系数 = %+.3f  （负值 = 超前后会减速）" % corr)

    slow = ok & (v < 0.10)
    print("速度 <0.10 m/s 的帧: %d / %d (%.0f%%)" % (slow.sum(), ok.sum(),
                                                     100.0 * slow.sum() / ok.sum()))
    if slow.any():
        print("  这些帧的平均 gap = %+.3f m （对比全局均值 %+.3f）" % (gap[slow].mean(), gap.mean()))
    fast = ok & (v > 0.60)
    if fast.any():
        print("速度 >0.60 m/s 的帧: %d, 平均 gap = %+.3f m" % (fast.sum(), gap[fast].mean()))

    print("\n  帧  时间s   gap(m)   速度(m/s)   备注")
    for f in range(0, T, max(1, T // 22)):
        note = ""
        if not np.isnan(v[f]):
            if v[f] < 0.10:
                note = "停等"
            elif v[f] > 0.60:
                note = "追赶/快走"
        print("  %4d %6.2f  %+7.3f   %8.3f   %s" % (f, f * dt, gap[f],
                                                    0.0 if np.isnan(v[f]) else v[f], note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
