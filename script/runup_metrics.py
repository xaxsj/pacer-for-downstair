"""How much run-up does a robot get before its first step-up?

Usage: python script/runup_metrics.py <tag> [<tag> ...]

Reads experiment_result/validation/<tag>/seed_0/states.pkl (multi-env rollout)
and reports, per env: horizontal distance travelled before the first step-up,
the speed at that moment, and how high it eventually climbs.
"""
import os
import pickle
import sys

import numpy as np


def first_step_up(dz, valid):
    idx = np.where((dz >= 0.025) & valid)[0]
    return int(idx[0]) if len(idx) else -1


def main(tags):
    for tag in tags:
        path = f"experiment_result/validation/{tag}/seed_0/states.pkl"
        if not os.path.exists(path):
            print(f"{tag}: missing {path}")
            continue
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
        v = np.linalg.norm(np.diff(rs[:, :, :2], axis=0), axis=2) / dt
        runup, speed_at, climb, climbed = [], [], [], 0
        for e in range(N):
            i = first_step_up(dz[:, e], valid[:, e])
            if i < 0:
                climb.append(0.0)
                continue
            climbed += 1
            runup.append(float(np.linalg.norm(rs[i, e, :2] - rs[0, e, :2])))
            speed_at.append(float(v[max(i - 1, 0), e]))
            climb.append(float(rs[:, e, 2].max() - rs[0, e, 2]))
        ru = np.array(runup)
        sp = np.array(speed_at)
        print(f"=== {tag} ===  envs={N}  climbed>=1 step: {climbed}/{N}")
        if len(ru):
            print(f"  run-up before first step-up [m]: "
                  f"median {np.median(ru):.2f}  mean {ru.mean():.2f}  "
                  f"p90 {np.percentile(ru, 90):.2f}")
            print(f"  speed at first step-up [m/s]:   "
                  f"median {np.median(sp):.2f}  mean {sp.mean():.2f}")
        print(f"  max climb in episode [m]: {max(climb):.2f} "
              f"({max(climb) / 0.15:.1f} risers);  "
              f"envs climbing >=0.15 m: {sum(1 for c in climb if c >= 0.15)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train_scene", "train_scene_plat1"])
