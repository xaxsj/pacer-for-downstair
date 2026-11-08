"""Extract a few frames from a rendered mp4 so they can be inspected."""
import sys

import imageio.v2 as imageio
import numpy as np

src = sys.argv[1]
out_prefix = sys.argv[2]
idxs = [int(x) for x in sys.argv[3].split(",")]

rdr = imageio.get_reader(src)
frames = []
for i, frame in enumerate(rdr):
    if i in idxs:
        imageio.imwrite(f"{out_prefix}_{i:04d}.png", np.asarray(frame))
        frames.append(i)
    if i > max(idxs):
        break
rdr.close()
print("wrote", frames)
