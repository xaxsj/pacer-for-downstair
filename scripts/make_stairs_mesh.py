"""把一组楼梯参数生成 PACER 需要的 ``data/mesh/<name>.pkl``。

几何模型 = 一块**地面** + 一组**实心台阶**（每级从地面 z=0 长到该级踏面高度）。
实心台阶自带侧面竖墙，因此不会出现「高度图说有台阶、碰撞网格却没有」的悬空区。

产物字段（与 ``humanoid_pedestrain_terrain.MeshTerrain`` 对齐）::

    vertices      (V, 3) float32   PhysX 三角网格顶点（世界坐标，米）
    faces         (F, 3) int32     三角面索引
    heigthmap     (W, H) float32   俯视高度图（代码里就是错拼 "heigth"）
    walkable_map  (W, H) int8      可站立掩码（1 = 可采样出生点）
    cam_pos       (2,) float32     高度图中心的世界坐标 (x, y)
    x_scale       float            每米多少像素（PACER 用 20 → 5 cm/格）
    y_scale       float

坐标约定（从 ``MeshTerrain.world_points_to_map`` 反推）::

    px = (x - cam_pos[0]) * x_scale + W / 2
    py = (y - cam_pos[1]) * y_scale + H / 2

用法::

    python scripts/make_stairs_mesh.py                       # 默认 20 级 0.15/0.32
    python scripts/make_stairs_mesh.py --spec stairs30.json
    python scripts/make_stairs_mesh.py --steps 10 --riser 0.2 --tread 0.3 --name stairs10
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

DEFAULTS = {
    "n_steps": 20,
    "riser": 0.15,
    "tread": 0.32,
    "width": 1.2,
    "x_start": 0.8,
    "top_platform": 1.2,
    "bottom_buffer": 2.0,
    "px_per_m": 20.0,
    "y_margin": 1.0,
    "name": "stairs20",
}


def z_profile(x_start, n_steps, riser, tread, top_z, x):
    """给定 x，返回该处世界表面的高度（平台 / 台阶 / 地面）。"""
    if x < x_start:
        return top_z
    idx = int((x - x_start) // tread)
    if idx >= n_steps:
        return 0.0
    return top_z - (idx + 1) * riser


def add_quad(vertices, faces, p0, p1, p2, p3):
    base = len(vertices)
    vertices.extend([p0, p1, p2, p3])
    faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])


def add_box_on_ground(vertices, faces, x0, x1, y0, y1, z_top):
    """实心盒子（底面在地面 z=0，不生成底面）：顶面 + 前面 + 两个侧面。"""
    if z_top <= 1e-9:
        return
    # 顶面（+z）
    add_quad(vertices, faces,
             (x0, y0, z_top), (x1, y0, z_top), (x1, y1, z_top), (x0, y1, z_top))
    # 前面（朝 -x，踢面）
    add_quad(vertices, faces,
             (x0, y0, 0.0), (x0, y0, z_top), (x0, y1, z_top), (x0, y1, 0.0))
    # 侧面（y0 朝 -y）
    add_quad(vertices, faces,
             (x0, y0, 0.0), (x1, y0, 0.0), (x1, y0, z_top), (x0, y0, z_top))
    # 侧面（y1 朝 +y）
    add_quad(vertices, faces,
             (x0, y1, 0.0), (x0, y1, z_top), (x1, y1, z_top), (x1, y1, 0.0))


def build_mesh(x_min, x_max, y_min, y_max, y_lo, y_hi,
               x_start, n_steps, riser, tread, top_z):
    """地面 + 实心台阶。y_lo/y_hi 是台阶宽度范围，y_min/y_max 是地面范围。"""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    # 地面（z = 0），覆盖整个地图
    add_quad(vertices, faces,
             (x_min, y_min, 0.0), (x_max, y_min, 0.0),
             (x_max, y_max, 0.0), (x_min, y_max, 0.0))

    # 顶部平台（实心，从地面长到 top_z）
    add_box_on_ground(vertices, faces, x_min, x_start, y_lo, y_hi, top_z)

    # 每一级台阶：实心盒子
    for i in range(n_steps):
        x0 = x_start + i * tread
        x1 = x0 + tread
        z_top = top_z - (i + 1) * riser
        add_box_on_ground(vertices, faces, x0, x1, y_lo, y_hi, z_top)

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def build_heightmap(x_min, x_max, y_min, y_max, y_lo, y_hi, half_width,
                    x_start, n_steps, riser, tread, top_z, px_per_m, edge_margin=0.05):
    """按 5 cm 网格采样高度；台阶只在 y_lo..y_hi 范围内，其余是地面 0。"""
    n_x = int(round((x_max - x_min) * px_per_m))
    n_y = int(round((y_max - y_min) * px_per_m))
    cam_pos = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)

    xs = cam_pos[0] + (np.arange(n_x) - n_x / 2.0) / px_per_m
    ys = cam_pos[1] + (np.arange(n_y) - n_y / 2.0) / px_per_m

    heights = np.zeros((n_x, n_y), dtype=np.float32)
    walkable = np.zeros((n_x, n_y), dtype=np.int8)

    on_stairs = (ys > y_lo + edge_margin) & (ys < y_hi - edge_margin)
    for i, x in enumerate(xs):
        z = z_profile(x_start, n_steps, riser, tread, top_z, x)
        inside_x = (x > x_min + edge_margin) and (x < x_max - edge_margin)
        for j, y in enumerate(ys):
            if not on_stairs[j]:
                heights[i, j] = 0.0          # 台阶之外就是地面
                walkable[i, j] = 1 if inside_x else 0
            else:
                heights[i, j] = z
                # 台阶面上可站立（离踢面边缘留一格，避免出生在棱上）
                x_local = (x - x_start) % tread if x >= x_start else 0.0
                near_edge = (x >= x_start) and (x_local < edge_margin or x_local > tread - edge_margin)
                walkable[i, j] = 1 if (not near_edge) else 0

    return heights, walkable, cam_pos


def generate(params: dict, out_path: Path, save_preview: bool = True) -> dict:
    n_steps = int(params["n_steps"])
    riser = float(params["riser"])
    tread = float(params["tread"])
    width = float(params["width"])
    x_start = float(params["x_start"])
    top_platform = float(params["top_platform"])
    bottom_buffer = float(params["bottom_buffer"])
    px_per_m = float(params["px_per_m"])
    y_margin = float(params["y_margin"])

    top_z = n_steps * riser
    x_min = x_start - top_platform
    x_max = x_start + n_steps * tread + bottom_buffer
    half_w = width / 2.0
    y_lo, y_hi = -half_w, half_w
    y_min, y_max = -half_w - y_margin, half_w + y_margin

    vertices, faces = build_mesh(x_min, x_max, y_min, y_max, y_lo, y_hi,
                                 x_start, n_steps, riser, tread, top_z)
    heights, walkable, cam_pos = build_heightmap(
        x_min, x_max, y_min, y_max, y_lo, y_hi, half_w,
        x_start, n_steps, riser, tread, top_z, px_per_m)

    payload = {
        "vertices": vertices,
        "faces": faces,
        "heigthmap": heights,
        "walkable_map": walkable,
        "cam_pos": cam_pos,
        "x_scale": px_per_m,
        "y_scale": px_per_m,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)

    if save_preview:
        _save_preview(heights, out_path.with_suffix(".png"))

    return payload


def _save_preview(heights: np.ndarray, path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    norm = heights - heights.min()
    if norm.max() > 0:
        norm = norm / norm.max()
    img = (norm.T[::-1] * 255).astype(np.uint8)
    imageio.imwrite(path, np.stack([img] * 3, axis=-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=str, default="", help="JSON 参数文件")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--riser", type=float, default=None)
    parser.add_argument("--tread", type=float, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--px_per_m", type=float, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="data/mesh")
    args = parser.parse_args()

    params = dict(DEFAULTS)
    if args.spec:
        params.update(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    for key in ("steps", "riser", "tread", "width", "px_per_m", "name"):
        value = getattr(args, key)
        if value is not None:
            params["n_steps" if key == "steps" else key] = value

    out_path = Path(args.out_dir) / f"{params['name']}.pkl"
    payload = generate(params, out_path)

    heights = payload["heigthmap"]
    print(f"[make_stairs_mesh] {out_path}")
    print(f"  vertices   {payload['vertices'].shape}")
    print(f"  faces      {payload['faces'].shape}")
    print(f"  heigthmap  {heights.shape}  @ {payload['x_scale']:.1f} px/m")
    print(f"  cam_pos    {payload['cam_pos'].tolist()}")
    print(f"  size       {heights.shape[0]/payload['x_scale']:.1f} m x "
          f"{heights.shape[1]/payload['y_scale']:.1f} m")
    print(f"  z range    {heights.min():.2f} .. {heights.max():.2f} m")
    print(f"  walkable   {int(payload['walkable_map'].sum())} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
