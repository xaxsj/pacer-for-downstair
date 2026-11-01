"""Sweep waypoint spacing and produce a comparison table.

For each trajSampleTimestep value it:
  1. runs one episode via script/run_stairs.sh
  2. measures from the recorded states.pkl:
       - measured waypoint spacing (arc distance between adjacent traj_samples)
       - steady-state lead: char_arc - sample0_arc   (sample0 == calc_pos(t) == tar_pos)
       - lateral drift, falls, descent completeness, mean speed
  3. prints a table and writes a CSV

Usage (from repo root):
    python script/sweep_spacing.py --dt 1.0,2.0,3.0,4.0,6.0 --frames 300
    python script/sweep_spacing.py --dt 1.0,2.0 --mesh stairs05_neg --path data/paths/stairs05_neg.json

Notes:
  - "falls" excludes the episode-boundary reset (last frame).
  - spacing is measured, not assumed: the nominal speed printed by
    run_stairs.sh uses L/episode and is NOT the real reference rate.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys

import numpy as np


def arc_len(v):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1))])


def project(v, s, xy):
    best_s, best_d = 0.0, 1e9
    for i in range(len(v) - 1):
        a, b = v[i, :2], v[i + 1, :2]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = float(np.clip((xy - a) @ ab / L2, 0.0, 1.0))
        p = a + t * ab
        d = float(np.linalg.norm(xy - p))
        if d < best_d:
            best_d, best_s = d, s[i] + t * (s[i + 1] - s[i])
    return best_s, best_d


def analyse(path, env=0):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    tv = np.asarray(d["traj_verts"], dtype=np.float64)
    ts = np.asarray(d["traj_samples"])[:, env]
    root = np.asarray(d["root_states"])[:, env]
    dt = float(np.asarray(d.get("dt", 0.0333)).ravel()[0])
    T = ts.shape[0]
    s = arc_len(tv)
    L = float(s[-1])

    # arc position of every waypoint sample
    A = np.array([[project(tv, s, ts[f, k, 0:2])[0] for k in range(ts.shape[1])]
                  for f in range(T)])
    # character arc position
    P = np.array([project(tv, s, root[f, 0:2])[0] for f in range(T)])

    # measured spacing: drop clamped (zero-length) steps, then take the median
    gaps = []
    for f in range(0, T, max(1, T // 40)):
        dk = np.diff(A[f])
        dk = dk[dk > 1e-6]                      # clamped samples give 0
        if dk.size >= 3:
            gaps.append(float(np.median(dk[:6])))   # leading steps are unclamped
    spacing = float(np.median(gaps)) if gaps else float("nan")

    lead = P - A[:, 0]

    z = root[:, 2]
    jumps = np.where(np.diff(z) > 0.4)[0] + 1
    falls = int(sum(1 for f in jumps if f != T - 1))
    boundary = bool(len(jumps) and jumps[-1] == T - 1)

    return {
        "frames": T,
        "spacing_m": spacing,
        "lead_mean_m": float(lead.mean()),
        "lead_absmax_m": float(np.abs(lead).max()),
        "speed_mps": float((P[-1] - P[0]) / ((T - 1) * dt)),
        "lateral_max_m": float(np.abs(root[:, 1] - root[0, 1]).max()),
        "falls": falls,
        "episode_boundary": boundary,
        "final_x": float(root[-1, 0]),
        "min_z": float(z.min()),
        "descended_m": float(z[0] - z.min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True, help="comma separated trajSampleTimestep values")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--mesh", default="stairs20w50_neg")
    ap.add_argument("--path", default="data/paths/stairs20_neg.json")
    ap.add_argument("--yaw", default="1.5708")
    ap.add_argument("--start_xy", default="0.0,0.0")
    ap.add_argument("--tag_prefix", default="sweep")
    ap.add_argument("--csv", default="experiment_result/spacing_sweep.csv")
    ap.add_argument("--skip_run", action="store_true", help="only analyse existing runs")
    args = ap.parse_args()

    dts = [x.strip() for x in args.dt.split(",") if x.strip()]
    rows = []
    for dtv in dts:
        tag = "%s_dt%s" % (args.tag_prefix, dtv.replace(".", "p"))
        outdir = os.path.join("experiment_result", tag)
        states = os.path.join(outdir, "states.pkl")

        if not args.skip_run:
            print("=" * 70)
            print("[run] traj_dt=%s  tag=%s" % (dtv, tag))
            print("=" * 70)
            cmd = ["bash", "script/run_stairs.sh",
                   "--mesh", args.mesh, "--path", args.path,
                   "--yaw", args.yaw, "--start_xy", args.start_xy,
                   "--frames", str(args.frames), "--traj_dt", dtv,
                   "--tag", tag]
            rc = subprocess.call(cmd)
            if rc != 0:
                print("[warn] run returned %d" % rc)
        if not os.path.exists(states):
            print("[skip] no states for traj_dt=%s" % dtv)
            continue

        m = analyse(states)
        m["traj_dt"] = dtv
        m["tag"] = tag
        rows.append(m)
        print("[ok] dt=%-6s spacing=%.3f m  lead=%+.3f m  falls=%d"
              % (dtv, m["spacing_m"], m["lead_mean_m"], m["falls"]))

    if not rows:
        print("no results")
        return 1

    print()
    header = ("traj_dt", "spacing", "lead_mean", "lead_max", "speed",
              "lat_max", "falls", "final_x", "min_z", "descended")
    print("%-8s %9s %10s %9s %8s %8s %6s %9s %8s %10s" % header)
    print("-" * 96)
    for m in rows:
        print("%-8s %9.3f %+10.3f %9.3f %8.3f %8.3f %6d %9.3f %8.3f %10.3f" % (
            m["traj_dt"], m["spacing_m"], m["lead_mean_m"], m["lead_absmax_m"],
            m["speed_mps"], m["lateral_max_m"], m["falls"],
            m["final_x"], m["min_z"], m["descended_m"]))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    keys = ["traj_dt", "tag", "spacing_m", "lead_mean_m", "lead_absmax_m",
            "speed_mps", "lateral_max_m", "falls", "episode_boundary",
            "final_x", "min_z", "descended_m", "frames"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in rows:
            w.writerow(m)
    print("\nCSV -> %s" % args.csv)
    print("columns: spacing / lead = steady-state overshoot vs calc_pos(t)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
