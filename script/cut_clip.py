"""Cut a frame range out of a rollout video (optionally slowed down).

Usage: python script/cut_clip.py in.mp4 out.mp4 START END [FPS]
"""
import sys

import imageio

src, dst = sys.argv[1], sys.argv[2]
start, end = int(sys.argv[3]), int(sys.argv[4])
fps = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0

rdr = imageio.get_reader(src)
writer = imageio.get_writer(dst, fps=fps, macro_block_size=None)
written = 0
for i, frame in enumerate(rdr):
    if i < start:
        continue
    if i >= end:
        break
    writer.append_data(frame)
    written += 1
writer.close()
rdr.close()
print(f"wrote {dst}: frames {start}..{end - 1} ({written} frames at {fps} fps)")
