"""Render one episode, freeze on its last frame, then stop (no next episode).

Usage: python script/video_episode_end.py in.mp4 out.mp4 RESET_FRAME [HOLD] [LABEL]
Plays frames [0, RESET_FRAME), holds the final frame for HOLD frames, and stops.
The held frames carry a short label so the termination is unmistakable.
"""
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

src, dst = sys.argv[1], sys.argv[2]
reset = int(sys.argv[3])
hold = int(sys.argv[4]) if len(sys.argv) > 4 else 60
label = sys.argv[5] if len(sys.argv) > 5 else "GOAL REACHED -> +5 -> RESET"

rdr = imageio.get_reader(src)
writer = imageio.get_writer(dst, fps=30, macro_block_size=None)

last = None
for i, frame in enumerate(rdr):
    if i >= reset:
        break
    writer.append_data(frame)
    last = frame

if last is not None:
    img = Image.fromarray(np.asarray(last)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([0, h - 90, w, h], fill=(0, 0, 0))
    draw.text((40, h - 66), label, fill=(255, 80, 80))
    draw.text((40, h - 40), f"episode ends at frame {reset}  "
                            f"(goal radius reached, reward +5)", fill=(230, 230, 230))
    still = np.asarray(img)
    for _ in range(hold):
        writer.append_data(still)

writer.close()
rdr.close()
print(f"wrote {dst}: {reset} frames + {hold} held frames ({label})")
