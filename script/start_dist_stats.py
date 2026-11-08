"""Distribution of the start distance (spawn -> first step-up) for a rollout.

Usage: python script/start_dist_stats.py <tag> [<tag> ...]

Complements runup_metrics.py with percentiles and a histogram, so the value can
be compared against a target window such as 0.4-0.8 m.
"""
import os
import pickle
import sys

import numpy as np

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 1e9]


def analyse(tag, target=(0.4, 0.8)):
    path = f"experiment_result/validation/{tag}/seed_0/states.pkl"
    if not os.path.exists(path):
        print(f"{tag}: missing")
        return
    d = pickle.load(open(path, "rb"))
    rs = np.asarray(d["root_states"], dtype=np.float64)[:, :, :13]
    prog = np.asarray(d["progress"])
    dt = float(d["dt"])
    T, N = prog.shape
    valid = np.ones((T - 1, N), dtype=bool)
    for e in range(N):
        dec = np.where(np.diff(prog[:, e]) < 0)[0]
        if len(dec):
            valid[dec[0]:, e] = False
    dz = np.diff(rs[:, :, 2], axis=0)

    dist = []
    for e in range(N):
        idx = np.where((dz[:, e] >= 0.025) & valid[:, e])[0]
        if not len(idx):
            continue
        i = int(idx[0])
        dist.append(float(np.linalg.norm(rs[i, e, :2] - rs[0, e, :2])))
    dist = np.array(dist)

    print(f"=== {tag} === envs={N}  climbed>=1 step: {len(dist)}/{N}")
    if not len(dist):
        return
    qs = [5, 10, 25, 50, 75, 90, 95]
    print("  percentiles [m]: " + "  ".join(
        f"p{q}={np.percentile(dist, q):.2f}" for q in qs))
    inside = np.mean((dist >= target[0]) & (dist <= target[1])) * 100
    print(f"  inside {target[0]}-{target[1]} m: {inside:.1f}%")
    hist, _ = np.histogram(dist, bins=BINS)
    labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
              "0.5-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
    for lab, h in zip(labels, hist):
        if h:
            print(f"    {lab:9s} {'#' * int(h):s} {h}")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["train_scene_stage1"]):
        analyse(t)
