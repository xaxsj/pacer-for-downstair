"""把 PACER 导出的状态 pkl 渲染成 mp4（本地、纯 CPU、无需 Isaac Gym）。

输入（由 ``--auto_record`` 在服务器上导出）::

    root_states      (T, E, 13)  位置 + 四元数(w,x,y,z)
    rigid_body_pos   (T, E, 24, 3)
    dof_pos / progress / path_file ...

用法::

    python scripts/render_states.py --states states.pkl --mesh data/mesh/stairs20.pkl \
        --out out.mp4 --env 0

骨架顺序沿用 PACER 代码里标注的 24 个 mujoco 关节名。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# PACER 注释里给出的 24 关节顺序
JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11, 12,
    11, 14, 15, 16, 17,
    11, 19, 20, 21, 22,
]


def load_states(path: Path, env: int):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    root = np.asarray(data["root_states"])[:, env]
    bodies = np.asarray(data["rigid_body_pos"])[:, env]
    return root, bodies, data


def load_mesh(path: Path):
    with open(path, "rb") as fh:
        mesh = pickle.load(fh)
    return np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])


def render(states_path: Path, mesh_path: Path | None, out_path: Path,
           env: int = 0, fps: int = 30, dpi: int = 90, stride: int = 1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    root, bodies, data = load_states(states_path, env)
    T = bodies.shape[0]

    vertices = faces = None
    if mesh_path is not None and Path(mesh_path).is_file():
        vertices, faces = load_mesh(mesh_path)

    frames = []
    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # 世界范围：以轨迹包围盒为准
    pts = bodies.reshape(-1, 3)
    lo = pts.min(axis=0) - 1.0
    hi = pts.max(axis=0) + 1.0
    if vertices is not None:
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))

    def draw(i):
        ax.cla()
        if vertices is not None:
            tris = vertices[faces]
            # 只画朝上的面，视觉上像台阶
            coll = Poly3DCollection(tris, facecolor="#b8b8b8",
                                    edgecolor="#606060", linewidths=0.25, alpha=0.95)
            ax.add_collection3d(coll)

        p = bodies[i]
        for j, par in enumerate(PARENTS):
            if par < 0:
                continue
            ax.plot([p[par, 0], p[j, 0]], [p[par, 1], p[j, 1]], [p[par, 2], p[j, 2]],
                    color="#1f77b4", linewidth=2.2, solid_capstyle="round")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#d62728", s=8, depthshade=False)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(max(lo[2], -1.0), hi[2])
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], max(hi[2] - lo[2], 0.5)))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=18, azim=-62)
        ax.set_title(f"frame {i+1}/{T}   root=({root[i,0]:.2f}, {root[i,1]:.2f}, {root[i,2]:.2f})"
                     f"   progress={int(np.asarray(data['progress'])[i, env])}",
                     fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    for i in range(0, T, stride):
        frames.append(draw(i))

    plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=None)
    print(f"[render_states] wrote {out_path}  frames={len(frames)}  fps={fps}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--mesh", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    states = Path(args.states)
    out = Path(args.out) if args.out else states.with_suffix(".mp4")
    render(states, Path(args.mesh) if args.mesh else None, out,
           env=args.env, fps=args.fps, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
