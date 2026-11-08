"""Print the structure of a stair mesh pkl so a generator can match it."""
import pickle
import sys

import numpy as np

path = sys.argv[1]
d = pickle.load(open(path, "rb"))
print("keys:", sorted(d.keys()))
for k, v in d.items():
    if isinstance(v, np.ndarray):
        print(f"  {k:14s} array {v.shape} {v.dtype}  "
              f"min={np.nanmin(v):.4f} max={np.nanmax(v):.4f}")
    else:
        print(f"  {k:14s} {type(v).__name__} = {v}")
hm = np.asarray(d["heigthmap"])
print("heightmap shape", hm.shape, "unique heights (first 12):",
      np.unique(hm)[:12])
print("cam_pos", d.get("cam_pos"), "x_scale", d.get("x_scale"),
      "y_scale", d.get("y_scale"))
