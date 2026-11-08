"""Print the height profile along +x at y=0 for a mesh pkl."""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
hm = np.asarray(d["heigthmap"])
cam = np.asarray(d["cam_pos"])
xs, ys = float(d["x_scale"]), float(d["y_scale"])
nx, ny = hm.shape
x_min = cam[0] - (nx / 2.0) / xs
y_min = cam[1] - (ny / 2.0) / ys
print("shape", hm.shape, "x_min", round(x_min, 2), "y_min", round(y_min, 2),
      "h range", round(float(hm.min()), 2), round(float(hm.max()), 2))
for x in np.arange(0.0, 8.01, 0.5):
    px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
    py = int(np.clip(np.floor((0.0 - y_min) * ys), 0, ny - 1))
    print(f"  x={x:4.1f} px={px:4d} py={py:4d} h={hm[px, py]:6.3f}")
