"""检查角色是否真的踩在地形上，以及是否走出了楼梯横向范围。

判据：
    clearance = 最低刚体 z − 该处地形高度
        ≈ 0        -> 踩实了
        持续 > 3cm -> 悬空（在滑/飞，不是走）
        持续 < -2cm-> 穿模
    |y| > 楼梯半宽 -> 已经走出楼梯侧面

用法：
    python script/check_contact.py --states X.pkl --mesh data/mesh/stairs05_neg.pkl
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def hm_height(hm, cam_pos, xs, ys, x, y):
    x_c = (hm.shape[0] / 2.0) / xs
    y_c = (hm.shape[1] / 2.0) / ys
    px = int(np.clip(round((x - (cam_pos[0] - x_c)) * xs), 0, hm.shape[0] - 1))
    py = int(np.clip(round((y - (cam_pos[1] - y_c)) * ys), 0, hm.shape[1] - 1))
    return float(hm[px, py])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--halfwidth", type=float, default=0.6, help="楼梯半宽 (m)")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    hm = np.asarray(md["heigthmap"], dtype=np.float64)
    cam = np.asarray(md["cam_pos"], dtype=np.float64)
    xsc, ysc = float(md["x_scale"]), float(md["y_scale"])

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    root = np.asarray(d["root_states"])[:, args.env]
    rb = np.asarray(d["rigid_body_pos"])[:, args.env]
    T = root.shape[0]

    print(" 帧    root(x,y,z)              地形h   最低刚体   clearance   横向|y|")
    rows = []
    for i in range(T):
        x, y, z = root[i, 0], root[i, 1], root[i, 2]
        h = hm_height(hm, cam, xsc, ysc, x, y)
        low = float(rb[i, :, 2].min())
        cl = low - h
        rows.append((i, x, y, z, h, low, cl))
        if i % 10 == 0 or i == T - 1:
            flag = ""
            if cl > 0.03:
                flag += " 悬空"
            if cl < -0.02:
                flag += " 穿模"
            if abs(y) > args.halfwidth:
                flag += " 出楼梯"
            print(" %3d  %+7.3f %+7.3f %+7.3f   %6.3f  %7.3f   %+8.3f   %6.3f%s"
                  % (i, x, y, z, h, low, cl, abs(y), flag))

    cl = np.array([r[6] for r in rows])
    yy = np.abs(np.array([r[2] for r in rows]))
    airborne = int((cl > 0.03).sum())
    sunk = int((cl < -0.02).sum())
    outside = int((yy > args.halfwidth).sum())
    print("\n=== 汇总 (%d 帧) ===" % T)
    print("  clearance: 均值 %+.3f  最小 %+.3f  最大 %+.3f" % (cl.mean(), cl.min(), cl.max()))
    print("  悬空帧 (>3cm): %d (%.0f%%)" % (airborne, 100.0 * airborne / T))
    print("  穿模帧 (<-2cm): %d (%.0f%%)" % (sunk, 100.0 * sunk / T))
    print("  走出楼梯 (|y|>%.2f): %d 帧 (%.0f%%)，最大 |y| = %.3f m"
          % (args.halfwidth, outside, 100.0 * outside / T, yy.max()))
    if outside:
        first = int(np.argmax(yy > args.halfwidth))
        print("  首次出界于第 %d 帧" % first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
