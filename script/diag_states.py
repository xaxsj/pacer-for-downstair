"""PACER 状态 dump 诊断工具（可复用）。

用途：不渲染、纯数值地判断一次 run 发生了什么 ——
      走了多远、掉了多高、重置几次、出生那一刻身体是站着还是躺着。

用法：
    python script/diag_states.py output/recordings/states-XXX.pkl
    python script/diag_states.py states.pkl --env 0

输出四块：
    1. 轨迹总览（起止点、位移、横向漂移、重置次数）
    2. 逐帧采样表
    3. 出生帧刚体分布（判断是否站姿）
    4. body +X 航向（方法 B，见 docs/初始位姿设置指南.md）
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 (x,y,z,w) 旋转向量 v，纯 numpy 实现（与 Isaac Gym quat_apply 同约定）。"""
    q = np.asarray(q_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def load(path: str, env: int):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])
    rb = np.asarray(d["rigid_body_pos"]) if "rigid_body_pos" in d else None
    if root.ndim == 3:
        root = root[:, env]
    if rb is not None and rb.ndim == 4:
        rb = rb[:, env]
    return d, root, rb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--jump", type=float, default=0.4, help="判定重置的 z 跳变阈值 (m)")
    args = ap.parse_args()

    d, root, rb = load(args.states, args.env)
    pos = root[:, 0:3]
    T = pos.shape[0]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    print("=" * 64)
    print("文件: %s" % args.states)
    print("字段: %s" % ", ".join(sorted(d.keys())))
    print("dt=%.5f  frames=%d  envs=%s" % (
        float(np.asarray(d.get("dt", 0)).ravel()[0]) if "dt" in d else -1, T,
        np.asarray(d["root_states"]).shape[1] if np.asarray(d["root_states"]).ndim == 3 else 1))

    # ---------- 1. 总览 ----------
    dz = np.diff(z)
    resets = np.where(dz > args.jump)[0] + 1
    print("\n[1] 轨迹总览")
    print("  起点 (x,y,z) = %+.3f %+.3f %+.3f" % (x[0], y[0], z[0]))
    print("  终点 (x,y,z) = %+.3f %+.3f %+.3f" % (x[-1], y[-1], z[-1]))
    print("  位移         dx=%+.3f  dy=%+.3f  dz=%+.3f" % (x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]))
    print("  横向漂移     max|dy| = %.3f m" % np.abs(y - y[0]).max())
    print("  高度范围     z: %.3f .. %.3f" % (z.min(), z.max()))
    if len(resets) > 1:
        per = np.diff(resets)
        print("  重置         %d 次，帧 %s" % (len(resets), resets[:12].tolist()))
        print("  重置周期     中位 %d 帧 (%.2f s)" % (int(np.median(per)), np.median(per) * 0.0333))
    else:
        print("  重置         %d 次" % len(resets))

    # ---------- 2. 逐帧 ----------
    print("\n[2] 逐帧采样")
    print("   frame      x       y       z    heading(deg)")
    ref = np.array([1.0, 0.0, 0.0])
    idxs = list(range(0, min(16, T))) + list(range(16, T, max(1, T // 10)))
    for i in idxs:
        hv = quat_rotate(root[i, 3:7], ref)
        print("  %5d  %+7.3f %+7.3f %+7.3f     %+8.1f" % (
            i, x[i], y[i], z[i], np.degrees(np.arctan2(hv[1], hv[0]))))

    # ---------- 3. 出生帧刚体分布 ----------
    if rb is not None:
        print("\n[3] 出生帧刚体分布（判断是否站姿）")
        b0 = rb[0]
        zs = b0[:, 2]
        spread = zs.max() - zs.min()
        horiz = np.linalg.norm(b0[:, :2] - b0[0, :2], axis=1).max()
        print("  刚体数=%d  z范围 %.3f..%.3f  竖直跨度=%.3f m  水平跨度=%.3f m" % (
            b0.shape[0], zs.min(), zs.max(), spread, horiz))
        if spread < 0.4 and horiz > 0.8:
            print("  >>> 判定：身体横躺（竖直跨度远小于水平跨度）")
        elif spread > 0.9:
            print("  >>> 判定：身体直立（竖直跨度 > 0.9 m）")
        else:
            print("  >>> 判定：不明确，建议用 script/render_path.py 看第 0 帧")
        order = np.argsort(zs)[::-1]
        print("  最高 3 个刚体: %s" % [(int(i), round(float(zs[i]), 3)) for i in order[:3]])
        print("  最低 3 个刚体: %s" % [(int(i), round(float(zs[i]), 3)) for i in order[-3:]])

    # ---------- 4. 航向 ----------
    print("\n[4] body +X 航向（度，方法 B）")
    hs = [np.degrees(np.arctan2(*(quat_rotate(root[i, 3:7], ref)[1::-1]))) for i in range(T)]
    print("  首帧 %+.1f   末帧 %+.1f   范围 [%+.1f, %+.1f]" % (hs[0], hs[-1], min(hs), max(hs)))
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
