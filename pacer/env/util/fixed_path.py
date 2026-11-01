"""固定路径支持：用一条给定的 2D 折线替代 TrajGenerator 的程序化随机轨迹。

背景
----
``TrajGenerator.calc_pos`` 按**线性时间**推进顶点索引：

    phase = t / traj_dur;  seg = phase * num_segs

也就是说**每段分到相同的时长**。因此只有当折线的段长相等时，隐含速度才恒定
（``v = 段长 / _dt``）。所以外部给定的路径必须**按弧长等距重采样**，否则角色会在
长段上冲刺、在短段上挪步。

用法
----
命令行：``--path_file data/paths/stairs20.json``
JSON 格式（两种都接受）：

    [[x, y, z], [x, y, z], ...]
    {"name": "stairs20", "path": [[x, y, z], ...]}

z 分量不参与控制（策略只取 XY），保留它是为了可视化和调试。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_path(path_file: str | Path) -> np.ndarray:
    """读取路径 JSON，返回 ``(N, 3)`` 的 float 数组。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 结构不合法、点数 < 2、含非有限值。
    """
    source = Path(path_file).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"path file not found: {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("path", "points", "waypoints"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"{source}: JSON object must contain 'path' / 'points' / 'waypoints'")

    points = np.asarray(payload, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError(f"{source}: path must be a list of >= 2 [x, y, z] points, got shape {points.shape}")
    if points.shape[1] == 2:
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    if points.shape[1] != 3:
        raise ValueError(f"{source}: each point must have 2 or 3 components, got {points.shape[1]}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source}: path contains non-finite values")
    return points


def resample_uniform(path: np.ndarray, n: int) -> np.ndarray:
    """按弧长等距重采样成 ``n`` 个点（保证隐含速度恒定）。"""
    if n < 2:
        raise ValueError("n must be >= 2")
    points = np.asarray(path, dtype=np.float64)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        raise ValueError("path has zero length")

    targets = np.linspace(0.0, total, n)
    return np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in range(3)], axis=-1)


def apply_to_verts(
    verts: torch.Tensor,
    env_ids: torch.Tensor,
    init_pos: torch.Tensor,
    path_file: str | Path,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """把固定路径写入 ``verts[env_ids]``，并把首个顶点对齐到角色当前根位置（仅 XY）。

    Args:
        verts: ``(num_envs, num_verts, 3)``，会被原地修改。
        env_ids: 选中的环境 id。
        init_pos: ``(len(env_ids), 3)`` 当前根位置。
        path_file: 路径 JSON。

    Returns:
        实际写入的等距折线（numpy，``(num_verts, 3)``），便于调试打印。
    """
    num_verts = verts.shape[1]
    resampled = resample_uniform(load_path(path_file), num_verts)

    # 可选：绕起点在 XY 平面内预旋转（用于补偿坐标约定差，例如 90 度）
    if abs(yaw_deg) > 1e-9:
        theta = np.deg2rad(yaw_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rel = resampled - resampled[0]
        rot_x = rel[:, 0] * cos_t - rel[:, 1] * sin_t
        rot_y = rel[:, 0] * sin_t + rel[:, 1] * cos_t
        resampled = np.stack([rot_x, rot_y, rel[:, 2]], axis=-1) + resampled[0]

    base = init_pos if torch.is_tensor(init_pos) else torch.as_tensor(init_pos)
    base = base[..., :3].to(device=verts.device, dtype=verts.dtype)
    points = torch.as_tensor(resampled, dtype=verts.dtype, device=verts.device)

    # 只对齐 XY：z 交给物理与高度图，路径里的 z 仅供可视化
    shift_xy = base[:, :2].unsqueeze(1) - points[:1, :2].unsqueeze(0)  # (n,1,2)
    aligned = points.unsqueeze(0).repeat(len(env_ids), 1, 1)
    aligned[..., :2] = aligned[..., :2] + shift_xy

    verts[env_ids] = aligned
    return resampled


__all__ = ["apply_to_verts", "load_path", "resample_uniform"]
