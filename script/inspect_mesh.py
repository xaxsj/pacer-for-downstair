"""量一下地形网格：平台边缘在哪、台阶多高多宽、出生点是否贴着边。

用法：
    python script/inspect_mesh.py data/mesh/stairs05_neg.pkl
    python script/inspect_mesh.py data/mesh/stairs05_neg.pkl --y 0.0
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--y", type=float, default=0.0, help="沿该 y 取剖面")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        d = pickle.load(fh)
    print("字段: %s" % ", ".join(sorted(d.keys())))

    v = np.asarray(d["vertices"], dtype=np.float64)
    f = np.asarray(d["faces"])
    print("vertices=%s faces=%s" % (v.shape, f.shape))
    print("bbox  x[%.2f, %.2f]  y[%.2f, %.2f]  z[%.2f, %.2f]" % (
        v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
        v[:, 2].min(), v[:, 2].max()))

    zs = np.unique(np.round(v[:, 2], 4))
    print("高度层 (%d): %s" % (len(zs), zs[:12].tolist()))

    # 高度图（注意：本仓库字段名拼作 heigthmap）
    key = "heightmap" if "heightmap" in d else ("heigthmap" if "heigthmap" in d else None)
    if key is not None:
        hm = np.asarray(d[key], dtype=np.float64)
        print("\n%s: %s  z范围 %.3f..%.3f" % (key, hm.shape, hm.min(), hm.max()))
        xs = float(d.get("x_scale", 0.05)); ys = float(d.get("y_scale", 0.05))
        print("  x_scale=%.4f y_scale=%.4f -> 覆盖 %.2f m x %.2f m" % (
            xs, ys, hm.shape[1] * xs, hm.shape[0] * ys))
        print("  walkable_map: %s" % ("有" if "walkable_map" in d else "无"))
        # 出生点 (0,0) 在高度图里对应的格子
        cx, cy = hm.shape[1] // 2, hm.shape[0] // 2
        print("\n  出生点附近高度图采样 (中心 3x3):")
        for r in range(cy - 1, cy + 2):
            print("    " + "  ".join("%.3f" % hm[r, c] for c in range(cx - 1, cx + 2)))
        print("  cam_pos = %s" % (d.get("cam_pos")))

    # 沿 x 的阶跃剖面（在给定 y 处取网格顶点，按 x 排序）
    print("\n=== 沿 x 的台阶剖面 (y≈%.2f) ===" % args.y)
    band = np.abs(v[:, 1] - args.y) < 0.35
    sub = v[band]
    if len(sub) == 0:
        print("  (该 y 处无顶点)")
    else:
        xs = np.unique(np.round(sub[:, 0], 3))
        print("  x 取值 %d 个: %s ... %s" % (len(xs), xs[:6].tolist(), xs[-3:].tolist()))
        prev = None
        for x in xs:
            zz = sub[np.isclose(sub[:, 0], x, atol=1e-3)][:, 2]
            hi = zz.max()
            if prev is None or abs(hi - prev) > 1e-3:
                print("    x=%+7.3f  z_top=%.4f" % (x, hi))
                prev = hi

    # 平台边缘：最高 z 层覆盖的 x 范围
    zmax = v[:, 2].max()
    top = v[np.isclose(v[:, 2], zmax, atol=1e-4)]
    print("\n=== 顶部平台 (z=%.3f) ===" % zmax)
    print("  x 范围 [%.3f, %.3f]   y 范围 [%.3f, %.3f]" % (
        top[:, 0].min(), top[:, 0].max(), top[:, 1].min(), top[:, 1].max()))
    print("  平台朝 -x 的边缘在 x=%.3f" % top[:, 0].min())
    print("  >>> 出生点 x=0 距该边缘 %.3f m" % abs(0.0 - top[:, 0].min()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
