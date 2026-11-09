"""Write a rollout video that freezes on the last frame before a reset.

Usage: python script/video_hold.py in.mp4 out.mp4 RESET_FRAME [HOLD_FRAMES]
The clip plays normally, holds the frame just before RESET_FRAME for
HOLD_FRAMES frames at 30 fps, then continues with the rest of the rollout.
"""
import sys

import imageio

src, dst = sys.argv[1], sys.argv[2]
reset = int(sys.argv[3])
hold = int(sys.argv[4]) if len(sys.argv) > 4 else 30

rdr = imageio.get_reader(src)
frames = [f for f in rdr]
rdr.close()
print(f"read {len(frames)} frames; reset at {reset}, hold {hold}")

writer = imageio.get_writer(dst, fps=30, macro_block_size=None)
for i, frame in enumerate(frames):
    if i == reset:
        # freeze the final frame of the episode, then resume
        for _ in range(hold):
            writer.append_data(frames[reset - 1])
    writer.append_data(frame)
writer.close()
print(f"wrote {dst}")
