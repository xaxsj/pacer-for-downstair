"""2-D sweep over waypoint spacing (traj_dt) x view floor (PACER_VIEW_MIN).

For every (traj_dt, view_min) pair it runs one episode, then measures from the
recorded states.pkl:
    measured spacing, steady-state lead, mean/peak speed,
    heading error vs path direction (deg), lateral drift, falls, descent.

Usage (from repo root):
    python script/sweep_2d.py --dt 1.0,2.0,3.0 --view_min 0,1,2 --frames 300

Outputs a table plus CSV. ASCII only on purpose (scp-safe).
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys

import numpy as np


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


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


def heading_of(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.cross(xyz, v)
    d = v + w * t + np.cross(xyz, t)
    return float(np.degrees(np.arctan2(d[1], d[0])))


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

    A = np.array([[project(tv, s, ts[f, k, 0:2])[0] for k in range(ts.shape[1])]
                  for f in range(T)])
    P = np.array([project(tv, s, root[f, 0:2])[0] for f in range(T)])

    gaps = []
    for f in range(0, T, max(1, T // 40)):
        dk = np.diff(A[f])
        dk = dk[dk > 1e-6]
        if dk.size >= 3:
            gaps.append(float(np.median(dk[:6])))
    spacing = float(np.median(gaps)) if gaps else float("nan")

    vel = np.zeros(T)
    vel[1:] = np.diff(P) / dt
    speed_mean = float((P[-1] - P[0]) / ((T - 1) * dt))
    speed_peak = float(vel.max())

    path_dir = float(np.degrees(np.arctan2(tv[-1, 1] - tv[0, 1], tv[-1, 0] - tv[0, 0])))
    heads = np.array([heading_of(root[f, 3:7]) for f in range(T)])
    err = np.abs(wrap180(heads - path_dir))

    z = root[:, 2]
    jumps = np.where(np.diff(z) > 0.4)[0] + 1
    falls = int(sum(1 for f in jumps if f != T - 1))

    return {
        "frames": T,
        "spacing_m": spacing,
        "lead_mean_m": float((P - A[:, 0]).mean()),
        "speed_mean": speed_mean,
        "speed_peak": speed_peak,
        "heading_err_mean_deg": float(err.mean()),
        "heading_err_end_deg": float(err[-1]),
        "lateral_max_m": float(np.abs(root[:, 1] - root[0, 1]).max()),
        "falls": falls,
        "final_x": float(root[-1, 0]),
        "min_z": float(z.min()),
        "descended_m": float(z[0] - z.min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True)
    ap.add_argument("--view_min", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--mesh", default="stairs20w50_neg")
    ap.add_argument("--path", default="data/paths/stairs20_neg.json")
    ap.add_argument("--yaw", default="1.5708")
    ap.add_argument("--tag_prefix", default="s2d")
    ap.add_argument("--csv", default="experiment_result/sweep_2d.csv")
    ap.add_argument("--skip_run", action="store_true")
    args = ap.parse_args()

    dts = [x.strip() for x in args.dt.split(",") if x.strip()]
    vms = [x.strip() for x in args.view_min.split(",") if x.strip()]
    rows = []

    for dtv in dts:
        for vmv in vms:
            tag = "%s_dt%s_vm%s" % (args.tag_prefix, dtv.replace(".", "p"), vmv.replace(".", "p"))
            outdir = os.path.join("experiment_result", tag)
            states = os.path.join(outdir, "states.pkl")

            if not args.skip_run:
                env = dict(os.environ)
                env["PACER_VIEW_MIN"] = vmv
                cmd = ["bash", "script/run_stairs.sh",
                       "--mesh", args.mesh, "--path", args.path,
                       "--yaw", args.yaw, "--start_xy", "0.0,0.0",
                       "--frames", str(args.frames), "--traj_dt", dtv,
                       "--tag", tag]
                print("[run] dt=%s view_min=%s -> %s" % (dtv, vmv, tag), flush=True)
                rc = subprocess.call(cmd, env=env)
                if rc != 0:
                    print("[warn] rc=%d" % rc, flush=True)
            if not os.path.exists(states):
                print("[skip] no states for %s" % tag, flush=True)
                continue

            m = analyse(states)
            m["traj_dt"] = dtv
            m["view_min"] = vmv
            m["tag"] = tag
            rows.append(m)
            print("[ok] dt=%-4s vm=%-4s spacing=%.3f speed=%.3f head_err=%.1f falls=%d"
                  % (dtv, vmv, m["spacing_m"], m["speed_mean"],
                     m["heading_err_mean_deg"], m["falls"]), flush=True)

    if not rows:
        print("no results")
        return 1

    print()
    print("%-7s %-6s %9s %8s %8s %9s %9s %8s %6s %9s %8s" % (
        "dt", "vm", "spacing", "speed", "peak", "head_err", "head_end",
        "lat_max", "falls", "final_x", "descend"))
    print("-" * 100)
    for m in rows:
        print("%-7s %-6s %9.3f %8.3f %8.3f %9.1f %9.1f %8.3f %6d %9.3f %8.3f" % (
            m["traj_dt"], m["view_min"], m["spacing_m"], m["speed_mean"],
            m["speed_peak"], m["heading_err_mean_deg"], m["heading_err_end_deg"],
            m["lateral_max_m"], m["falls"], m["final_x"], m["descended_m"]))

    # simple ranking: fewest falls, then lowest heading error, then closest to a
    # natural walking speed (~0.8 m/s), then least lateral drift
    def score(m):
        return (m["falls"],
                m["heading_err_mean_deg"] / 30.0,
                abs(m["speed_mean"] - 0.8) / 0.4,
                m["lateral_max_m"] / 0.2)

    best = min(rows, key=score)
    print("\n[best] dt=%s view_min=%s  (head_err=%.1f deg, speed=%.3f, falls=%d)"
          % (best["traj_dt"], best["view_min"], best["heading_err_mean_deg"],
             best["speed_mean"], best["falls"]))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    keys = ["traj_dt", "view_min", "tag", "spacing_m", "lead_mean_m", "speed_mean",
            "speed_peak", "heading_err_mean_deg", "heading_err_end_deg",
            "lateral_max_m", "falls", "final_x", "min_z", "descended_m", "frames"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in rows:
            w.writerow(m)
    print("CSV -> %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
