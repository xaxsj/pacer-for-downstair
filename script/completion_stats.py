"""Per-env stair-climbing statistics from a multi-env states.pkl.

Classifies every env by what its root height did during the rollout:
  climbed     max rise > 0.25 m
  completed   climbed, then came back down to within 0.15 m of the start height
              (i.e. it went up AND descended the flight)
  stalled     climbed but the last third of the rollout barely moved in x

Usage: python script/completion_stats.py <states.pkl>
"""
import pickle
import sys

import numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"])          # [frames, envs, 13]
z = rs[:, :, 2]
x = rs[:, :, 0]
n_frames, n_env = z.shape

z0 = z[0]
rise = z.max(axis=0) - z0
peak_idx = z.argmax(axis=0)

climbed = rise > 0.25
completed = np.zeros(n_env, dtype=bool)
for i in range(n_env):
    if not climbed[i]:
        continue
    after = z[peak_idx[i]:, i]
    if after.min() < z0[i] + 0.15:
        completed[i] = True

tail = max(n_frames // 3, 1)
dx = np.abs(x[-1] - x[-tail]).mean() if n_env else 0.0
tail_motion = np.abs(x[-1] - x[-tail])

print(f"frames={n_frames} envs={n_env}")
print(f"climbed   (>0.25 m)          {climbed.sum():4d} / {n_env}  "
      f"({100.0 * climbed.sum() / n_env:.0f}%)")
print(f"completed (up + down)        {completed.sum():4d} / {n_env}  "
      f"({100.0 * completed.sum() / n_env:.0f}%)")
print(f"stalled   (tail dx < 0.3 m)  {(tail_motion < 0.3).sum():4d} / {n_env}  "
      f"({100.0 * (tail_motion < 0.3).sum() / n_env:.0f}%)")
print(f"mean rise {rise.mean():.2f} m   median {np.median(rise):.2f} m   "
      f"max {rise.max():.2f} m")
print(f"mean tail dx {tail_motion.mean():.2f} m")
