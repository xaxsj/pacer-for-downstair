"""Print the terrain height along y at a fixed x for a stair mesh."""
import pickle
import sys

import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else "wide_stairs"
x_probe = float(sys.argv[2]) if len(sys.argv) > 2 else 2.6

md = pickle.load(open(f"data/mesh/{name}.pkl", "rb"))
hm = np.asarray(md["heigthmap"], dtype=float)
cam = np.asarray(md["cam_pos"], dtype=float)
xs, ys = float(md["x_scale"]), float(md["y_scale"])
nx, ny = hm.shape
x_min = cam[0] - (nx / 2.0) / xs
y_min = cam[1] - (ny / 2.0) / ys


def h(x, y):
    px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
    py = int(np.clip(np.floor((y - y_min) * ys), 0, ny - 1))
    return float(hm[px, py])


print(f"{name}: y-range of mesh = [{y_min:.2f}, {y_min + ny / ys:.2f}]")
print(f"height along y at x={x_probe}:")
row = []
for y in np.arange(-1.0, 9.0, 0.1):
    row.append(f"{y:+.1f}:{h(x_probe, y):.2f}")
for i in range(0, len(row), 8):
    print("  " + "  ".join(row[i:i + 8]))
