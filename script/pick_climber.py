"""Pick the best-climbing env in a multi-env rollout and print its spawn pose.

Usage: python script/pick_climber.py <tag> [seed]
"""
import pickle
import sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "train_scene_plat1"
seed = sys.argv[2] if len(sys.argv) > 2 else "0"
path = f"experiment_result/validation/{tag}/seed_{seed}/states.pkl"

d = pickle.load(open(path, "rb"))
rs = np.asarray(d["root_states"], dtype=np.float64)[:, :, :13]
prog = np.asarray(d["progress"])
dt = float(d["dt"])

climb = np.zeros(rs.shape[1])
for e in range(rs.shape[1]):
    dec = np.where(np.diff(prog[:, e]) < 0)[0]
    end = int(dec[0]) if len(dec) else rs.shape[0]
    climb[e] = rs[:end, e, 2].max() - rs[0, e, 2]

best = np.argsort(-climb)[:8]
print("top climbers (env, rise [m], risers, dist [m], speed [m/s], spawn xy, yaw)")
for e in best:
    dec = np.where(np.diff(prog[:, e]) < 0)[0]
    end = int(dec[0]) if len(dec) else rs.shape[0]
    seg = rs[:end, e]
    dist = float(np.linalg.norm(seg[-1, :2] - seg[0, :2]))
    q = seg[0, 3:7]
    yaw = np.degrees(2 * np.arctan2(q[2], q[3]))
    print(f"  env {e:3d}  {climb[e]:6.2f}  {climb[e] / 0.15:5.1f}  "
          f"{dist:5.2f}  {dist / max((end - 1) * dt, 1e-6):4.2f}  "
          f"({seg[0, 0]:7.2f},{seg[0, 1]:7.2f})  yaw={yaw:+.4f} rad")
