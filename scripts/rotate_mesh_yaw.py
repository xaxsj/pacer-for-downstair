"""绕 z 轴旋转地形（网格 + 高度图 + cam_pos）。

目的：让楼梯的下降方向与角色“自然朝向”一致，从而初始位姿与轨迹天然对齐，
不必依赖 --start_yaw。

用法::

    python scripts/rotate_mesh_yaw.py src.pkl dst.pkl --yaw 90
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def bilinear(h, fx, fy):
    """在 h[px, py] 上双线性采样（像素坐标）。"""
    W, H = h.shape
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    return (h[x0, y0] * (1 - tx) * (1 - ty) + h[x1, y0] * tx * (1 - ty)
            + h[x0, y1] * (1 - tx) * ty + h[x1, y1] * tx * ty)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--yaw", type=float, required=True, help="degrees, CCW")
    args = ap.parse_args()

    with open(args.src, "rb") as fh:
        d = pickle.load(fh)

    th = np.deg2rad(args.yaw)
    c, s = np.cos(th), np.sin(th)

    verts = np.asarray(d["vertices"], dtype=np.float32).copy()
    faces = np.asarray(d["faces"], dtype=np.int32).copy()
    heights = np.asarray(d["heigthmap"], dtype=np.float32)
    walkable = np.asarray(d["walkable_map"], dtype=np.int8)
    cam = np.asarray(d["cam_pos"], dtype=np.float32).copy()
    xs = float(d["x_scale"])
    ys = float(d["y_scale"])

    # 旋转顶点（绕原点）
    vx, vy = verts[:, 0].copy(), verts[:, 1].copy()
    verts[:, 0] = c * vx - s * vy
    verts[:, 1] = s * vx + c * vy
    # 旋转保持手性，faces 不变

    # 旋转 cam_pos
    cx, cy = cam[0], cam[1]
    cam[0] = c * cx - s * cy
    cam[1] = s * cx + c * cy

    # 重建高度图：对每个新格中心，反旋转回原坐标，双线性采样
    W, H = heights.shape
    new_h = np.zeros((W, H), dtype=np.float32)
    new_w = np.zeros((W, H), dtype=np.int8)

    # 保证输出网格能覆盖旋转后的地形：用旋转后顶点的包围盒重新取格
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    n_x = int(round((x_max - x_min) * xs)) + 1
    n_y = int(round((y_max - y_min) * ys)) + 1
    new_h = np.zeros((n_x, n_y), dtype=np.float32)
    new_w = np.zeros((n_x, n_y), dtype=np.int8)
    new_cam = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)

    px = np.arange(n_x)
    py = np.arange(n_y)
    gx = new_cam[0] + (px - n_x / 2.0) / xs
    gy = new_cam[1] + (py - n_y / 2.0) / ys

    # 原图上的采样坐标（反旋转）
    for j in range(n_y):
        for i in range(n_x):
            wx, wy = gx[i], gy[j]
            ox = c * wx + s * wy          # R(-theta)
            oy = -s * wx + c * wy
            fx = (ox - cam[0]) * xs + W / 2.0
            fy = (oy - cam[1]) * ys + H / 2.0
            if 0 <= fx <= W - 1 and 0 <= fy <= H - 1:
                new_h[i, j] = bilinear(heights, np.array([fx]), np.array([fy]))[0]
                new_w[i, j] = int(bilinear(walkable.astype(np.float32),
                                           np.array([fx]), np.array([fy]))[0] > 0.5)

    out = dict(d)
    out.update(vertices=verts, faces=faces, heigthmap=new_h,
               walkable_map=new_w, cam_pos=new_cam)

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "wb") as fh:
        pickle.dump(out, fh, protocol=4)

    print(f"[rotate] {args.src} --yaw {args.yaw} --> {args.dst}")
    print(f"  vertices {verts.shape}  x {verts[:,0].min():.2f}..{verts[:,0].max():.2f}  "
          f"y {verts[:,1].min():.2f}..{verts[:,1].max():.2f}")
    print(f"  heightmap {new_h.shape}  z {new_h.min():.2f}..{new_h.max():.2f}")
    print(f"  cam_pos {new_cam.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
