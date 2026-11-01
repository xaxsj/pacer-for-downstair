"""把 stairs20.pkl 沿 x 镜像成"向下走 −x"的版本。

用途：PACER 的 SMPL 人体默认朝向约 180°（面朝 −x），而默认朝向下的路径必须朝 −x
才能让它正常行走（朝 +x 时会原地崩溃）。镜像世界比强行改朝向可靠。

用法::

    python scripts/mirror_mesh_x.py data/mesh/stairs20.pkl data/mesh/stairs20_neg.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def mirror(payload: dict) -> dict:
    verts = np.asarray(payload["vertices"], dtype=np.float32).copy()
    faces = np.asarray(payload["faces"], dtype=np.int32).copy()
    heights = np.asarray(payload["heigthmap"], dtype=np.float32).copy()
    walkable = np.asarray(payload["walkable_map"], dtype=np.int8).copy()
    cam = np.asarray(payload["cam_pos"], dtype=np.float32).copy()

    verts[:, 0] = -verts[:, 0]
    # 镜像后三角形绕序翻转，交换后两列恢复外法线
    faces = faces[:, [0, 2, 1]]
    # 高度图沿 x 轴翻转
    heights = heights[::-1, :].copy()
    walkable = walkable[::-1, :].copy()
    cam[0] = -cam[0]

    out = dict(payload)
    out.update(
        vertices=verts,
        faces=faces,
        heigthmap=heights,
        walkable_map=walkable,
        cam_pos=cam,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    with open(args.src, "rb") as fh:
        payload = pickle.load(fh)
    out = mirror(payload)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "wb") as fh:
        pickle.dump(out, fh, protocol=4)

    h = out["heigthmap"]
    print(f"[mirror] {args.src} -> {args.dst}")
    print(f"  vertices {out['vertices'].shape}  faces {out['faces'].shape}")
    print(f"  x range  {out['vertices'][:,0].min():.2f} .. {out['vertices'][:,0].max():.2f}")
    print(f"  cam_pos  {out['cam_pos'].tolist()}")
    print(f"  z range  {h.min():.2f} .. {h.max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
