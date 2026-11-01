"""把 traj_samples（画面上的球）和角色位置逐帧对齐，判断有几个球在身后。

用法：
    python script/check_markers.py --states X.pkl --frames 0,60,120,300
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
    ap.add_argument("--frames", default="0,30,60,120,180,240,300")
    ap.add_argument("--env", type=int, default=0)
    args = ap.parse_args()

    with open(args.states, "rb") as fh:
        d = pickle.load(fh)
    tv = np.asarray(d["traj_verts"], dtype=np.float64)
    ts = np.asarray(d["traj_samples"])[:, args.env]      # (T, 10, 3)
    root = np.asarray(d["root_states"])[:, args.env]
    s = arc(tv)

    print("traj_samples shape = %s  (T, 10, 3)" % (ts.shape,))
    for f in [int(x) for x in args.frames.split(",")]:
        if f >= root.shape[0]:
            continue
        sp, _ = proj(tv, s, root[f, 0:2])
        smp = [proj(tv, s, ts[f, k, 0:2])[0] for k in range(ts.shape[1])]
        ahead = [x - sp for x in smp]
        n_behind = sum(1 for a in ahead if a < 0)
        print("\nframe %3d  root=(%.2f, %.2f, %.2f)  沿路径 %.3f m"
              % (f, root[f, 0], root[f, 1], root[f, 2], sp))
        print("   球相对角色的进度差(m): " + " ".join("%+.2f" % a for a in ahead))
        print("   身后的球: %d / %d   最近的在 %+.2f m"
              % (n_behind, len(ahead), min(ahead, key=abs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
