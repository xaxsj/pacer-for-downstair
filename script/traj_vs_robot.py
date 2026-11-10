"""Compare the target polyline extent with the path json and the robot's travel."""
import json
import pickle
import sys

import numpy as np

st = pickle.load(open(sys.argv[1], "rb"))
verts = np.asarray(st["traj_verts"]); print("traj_verts raw", verts.shape, verts.dtype); verts = verts.reshape(-1, 3)[: , :]
samples = np.asarray(st["traj_samples"])[:, 0]  # (frames, 10, 3)
root = np.asarray(st["root_states"])[:, 0, :13]

print(f"traj_verts  {verts.shape}  x {verts[:,0].min():.2f} .. {verts[:,0].max():.2f}")
print(f"samples     {samples.shape}  first x {samples[0,0,0]:.2f}  "
      f"last-frame first x {samples[-1,0,0]:.2f}")
print(f"robot       x {root[0,0]:.2f} -> {root[-1,0]:.2f}")
advance = np.linalg.norm(np.diff(samples[:, 0, :2], axis=0), axis=1) * 30.0
print(f"target advance speed: mean {advance.mean():.2f}  "
      f"frames moving >0.5 m/s: {(advance > 0.5).sum()} / {len(advance)}")
print(f"target last-frame[:3] {np.round(samples[-1,0],2)}  "
      f"robot last {np.round(root[-1,:3],2)}")
gap = np.linalg.norm(samples[-1, 0, :2] - root[-1, :2])
print(f"gap between frozen target and robot at end: {gap:.2f} m")
