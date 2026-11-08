"""Detect episode resets (teleports) in a rendered video by frame differencing.

Usage: python script/detect_video_resets.py <video.mp4> [threshold]
"""
import sys

import imageio.v2 as imageio
import numpy as np

src = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

rdr = imageio.get_reader(src)
prev = None
diffs = []
for frame in rdr:
    g = np.asarray(frame)[::8, ::8].mean(axis=-1)  # cheap downsample
    if prev is not None:
        diffs.append(float(np.abs(g - prev).mean()))
    prev = g
rdr.close()

d = np.array(diffs)
print(f"frames={len(d) + 1}  mean diff={d.mean():.2f}  max={d.max():.2f}")
peaks = [(i + 1, float(v)) for i, v in enumerate(d) if v > thr]
print(f"frames above threshold {thr}: {len(peaks)}")
for i, v in peaks[:12]:
    print(f"  frame {i:4d} ({i / 30.0:5.2f}s)  diff={v:6.2f}")
if peaks:
    print("largest:", max(peaks, key=lambda t: t[1]))
