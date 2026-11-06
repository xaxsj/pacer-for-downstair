#!/usr/bin/env python3
"""Extract a 4-frame contact sheet from each evaluation mp4."""
from __future__ import annotations

import glob
import math
import os

import imageio.v2 as imageio
import numpy as np

ROOT = "experiment_result/eval_stage1"
OUT = os.path.join(ROOT, "frames")
os.makedirs(OUT, exist_ok=True)

for path in sorted(glob.glob(os.path.join(ROOT, "*.mp4"))):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        try:
            nf = float(meta.get("nframes", 0))
            n = int(nf) if math.isfinite(nf) and nf > 0 else 0
        except Exception:
            n = 0
        if n <= 0:
            try:
                n = int(float(meta.get("duration", 30.0)) * float(meta.get("fps", 30.0)))
            except Exception:
                n = 300
        idxs = [0, n // 3, (2 * n) // 3, n - 1]
        frames = []
        for i in idxs:
            try:
                frame = reader.get_data(min(i, n - 1))
                frames.append(np.asarray(frame)[..., :3])
            except Exception:
                pass
        reader.close()
        if not frames:
            print(name, "NO_FRAMES")
            continue
        h = min(f.shape[0] for f in frames)
        w = min(f.shape[1] for f in frames)
        frames = [f[:h, :w] for f in frames]
        sheet = np.concatenate(frames, axis=1)
        out = os.path.join(OUT, name + "_sheet.png")
        imageio.imwrite(out, sheet)
        print(name, "frames", n, "fps", meta.get("fps"), "sheet", out, sheet.shape)
    except Exception as exc:
        print(name, "ERROR", type(exc).__name__, exc)
