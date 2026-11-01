"""分析 dof 序列：关节到底怎么动的。

回答两个问题：
  1. 腿在动吗？（各关节角度变化范围）
  2. 是怎么动的？（一个周期内哪些关节在大幅摆动、是否趋于极限）

doF 布局（69 = 23 关节 × 3 轴）：dof i -> 关节 i//3+1, 轴 i%3 (0=x,1=y,2=z)
SMPL 关节索引：0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee,
               6 spine2, 7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot,
               12 neck, 15 head

用法：
    python script/diag_dof.py --states X.pkl --cycle 13
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

SMPL = {0: "pelvis", 1: "L_hip", 2: "R_hip", 3: "spine1", 4: "L_knee", 5: "R_knee",
        6: "spine2", 7: "L_ankle", 8: "R_ankle", 9: "spine3", 10: "L_foot",
        11: "R_foot", 12: "neck", 13: "L_collar", 14: "R_collar", 15: "head",
        16: "L_shoulder", 17: "R_shoulder", 18: "L_elbow", 19: "R_elbow",
        20: "L_wrist", 21: "R_wrist", 22: "L_hand", 23: "R_hand"}


def name_of(di: int) -> str:
    j = di // 3 + 1
    ax = "xyz"[di % 3]
    return "%-10s.%s" % (SMPL.get(j, "j%d" % j), ax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--cycle", type=int, default=13)
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    dof = np.asarray(d["dof_pos"])[:, args.env]
    T, D = dof.shape
    print("dof_pos: %s  (%d 关节 × 3 轴)" % (dof.shape, D // 3))

    c = min(args.cycle, T)
    seg = dof[:c]
    rng = seg.max(axis=0) - seg.min(axis=0)
    print("\n=== 一个重置周期内(前 %d 帧)角度变化幅度 前 15 名 ===" % c)
    order = np.argsort(rng)[::-1]
    print("   dof  关节.轴        最小     最大     幅度(rad)  幅度(deg)")
    for di in order[:15]:
        print("  %3d  %-12s %8.3f %8.3f  %8.4f  %8.1f" % (
            di, name_of(di), seg[:, di].min(), seg[:, di].max(), rng[di], np.degrees(rng[di])))

    moved = int((rng > np.radians(5)).sum())
    print("\n变化 >5° 的 dof 数: %d / %d" % (moved, D))

    # 腿部关节
    legs = [1, 2, 4, 5, 7, 8]
    print("\n=== 腿部关节（髋/膝/踝）逐帧 ===")
    idxs = [3 * (j - 1) + a for j in legs for a in range(3)]
    keep = [i for i in idxs if i < D and rng[i] > np.radians(2)]
    if not keep:
        print("  (腿部关节几乎没有变化)")
    else:
        hdr = "  frame " + "".join("%9s" % name_of(i)[:8] for i in keep)
        print(hdr)
        for f in range(c):
            print("  %5d " % f + "".join("%9.3f" % dof[f, i] for i in keep))

    # 是否贴极限
    print("\n=== 关节是否贴近限位（|角度| > 1.2 rad 视为接近极限）===")
    hit = [(i, float(seg[:, i].min()), float(seg[:, i].max()))
           for i in range(D) if np.abs(seg[:, i]).max() > 1.2]
    if not hit:
        print("  无")
    for i, lo, hi in sorted(hit, key=lambda t: -max(abs(t[1]), abs(t[2])))[:10]:
        print("  %-12s [%+.3f, %+.3f]" % (name_of(i), lo, hi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
