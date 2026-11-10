"""Inspect the recorded target trajectory speed for a rollout."""
import pickle
import sys

import numpy as np

st = pickle.load(open(sys.argv[1], "rb"))
print("keys:", sorted(st.keys()))
if "traj_samples" in st:
    t = np.asarray(st["traj_samples"])
    print("traj_samples", t.shape)
    # last dim usually (x, y, z, yaw?) - print the first few rows of env 0
    seq = t[:, 0]
    print("first 5 samples:", np.round(seq[:5], 2).tolist())
    print("last 5 samples:", np.round(seq[-5:], 2).tolist())
    d = np.linalg.norm(np.diff(seq[:, :2], axis=0), axis=1) * 30.0
    print(f"target speed: mean {d.mean():.2f}  min {d.min():.2f}  "
          f"max {d.max():.2f} m/s")
