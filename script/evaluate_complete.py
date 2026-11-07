#!/usr/bin/env python3
"""Complete first-failure evaluation from an eval states.pkl.

Uses recorded rigid-body poses + contact forces:
- OBB foot-box corners vs terrain height (penetration/clearance);
- contact force on ankle/toe and its horizontal component;
- first contact/hook, first penetration, first reset.

This is evaluation only; the policy and reward are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import pickle

import numpy as np

# body index, name, box center, box half-size (from models/smpl_humanoid.xml)
FOOT_BOXES = [
    (3, "L_Ankle", (0.0596, 0.0132, -0.0279), (0.0850, 0.0483, 0.0208)),
    (4, "L_Toe",   (0.0606, -0.0132, 0.0270), (0.0202, 0.0478, 0.0200)),
    (7, "R_Ankle", (0.0617, -0.0127, -0.0241), (0.0865, 0.0483, 0.0208)),
    (8, "R_Toe",   (0.0627, 0.0127, 0.0249), (0.0195, 0.0479, 0.0216)),
]


def height_sampler(md):
    hm = np.asarray(md["heigthmap"], dtype=np.float64)
    cam = np.asarray(md["cam_pos"], dtype=np.float64)
    xs, ys = float(md["x_scale"]), float(md["y_scale"])
    nx, ny = hm.shape
    x_min = float(cam[0]) - (nx / 2.0) / xs
    y_min = float(cam[1]) - (ny / 2.0) / ys

    def sample(x, y):
        px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
        py = int(np.clip(np.floor((y - y_min) * ys), 0, ny - 1))
        return float(hm[px, py])

    return sample


def quat_rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_forward(q):
    d = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return np.array([d[0], d[1]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--halfwidth", type=float, default=1.4)
    ap.add_argument("--window", type=int, default=55)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.mesh, "rb") as fh:
        md = pickle.load(fh)
    sample = height_sampler(md)
    with open(args.states, "rb") as fh:
        d = pickle.load(fh)

    root = np.asarray(d["root_states"])[:, 0]
    rb = np.asarray(d["rigid_body_pos"])[:, 0]
    rrot = np.asarray(d["rigid_body_rot"])[:, 0]
    rvel = np.asarray(d["rigid_body_vel"])[:, 0]
    cf = np.asarray(d["contact_forces"])[:, 0]
    T = root.shape[0]
    print("keys", sorted(d.keys()))
    print("shapes", {k: tuple(np.asarray(d[k]).shape) for k in
                     ("root_states", "rigid_body_pos", "rigid_body_rot",
                      "rigid_body_vel", "contact_forces") if k in d})
    if args.inspect:
        mag = np.linalg.norm(cf, axis=-1)
        print("contact max", float(mag.max()), "body max", np.round(mag.max(axis=0), 2).tolist())
        for i in range(0, min(T, 50)):
            ids = np.where(mag[i] > 1.0)[0].tolist()
            if ids:
                print(i, [(j, round(float(mag[i, j]), 2)) for j in ids])
        return 0

    # OBB corners and clearance per foot per frame.
    corners = []
    for idx, name, c, hs in FOOT_BOXES:
        local = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local.append(np.array(c) + np.array([sx * hs[0], sy * hs[1], sz * hs[2]]))
        local = np.stack(local)
        world = np.empty((T, 8, 3))
        for i in range(T):
            for k in range(8):
                world[i, k] = rb[i, idx] + quat_rotate(rrot[i, idx], local[k])
        h = np.array([[sample(float(p[0]), float(p[1])) for p in frame] for frame in world])
        cl = world[:, :, 2] - h
        corners.append({"idx": idx, "name": name, "cl": cl.min(axis=1), "pen": np.maximum(0.0, -cl).max(axis=1)})

    clear = np.stack([c["cl"] for c in corners], axis=1)
    pen = np.stack([c["pen"] for c in corners], axis=1)
    rxy = root[:, :2]
    rv = np.zeros_like(rxy)
    rv[1:] = np.diff(rxy, axis=0) * 30.0
    rv[0] = rv[1]
    fwd = np.array([quat_forward(q) for q in root[:, 3:7]])

    # Contact force on each foot  (all bodies are stored).
    contact_mag = np.linalg.norm(cf, axis=-1)
    contact_fwd = np.zeros((T, 4))
    for j, (idx, name, _, _) in enumerate([(b[0], b[1], b[2], b[3]) for b in FOOT_BOXES]):
        fxy = cf[:, idx, :2]
        contact_fwd[:, j] = np.sum(fxy * fwd, axis=1)
    c_mag = contact_mag[:, [b[0] for b in FOOT_BOXES]]

    z = root[:, 2]
    progress = np.asarray(d.get("progress", np.array([]))).reshape(T, -1)[:, 0] \
        if "progress" in d and np.asarray(d["progress"]).size else np.zeros(T)
    resets = np.where(np.diff(z) > 0.4)[0] + 1
    if progress.size == T:
        resets = np.unique(np.concatenate([resets, np.where(np.diff(progress) < -5)[0] + 1]))
    first_reset = int(resets[0]) if len(resets) else -1

    search_end = first_reset if first_reset >= 0 else T
    valid = np.arange(T) < search_end
    first_contact = int(np.where((c_mag.max(axis=1) > 1.0) & valid)[0][0]) if np.any((c_mag.max(axis=1) > 1.0) & valid) else -1
    first_pen = int(np.where((pen.max(axis=1) > 0.005) & valid)[0][0]) if np.any((pen.max(axis=1) > 0.005) & valid) else -1
    # Hook proxy: a foot in contact with a rearward horizontal force while root advances.
    hook_score = contact_mag[:, [4, 8]].max(axis=1) * np.maximum(0.0, -contact_fwd[:, [1, 3]].max(axis=1))
    first_hook = int(np.where((hook_score > 1.0) & valid)[0][0]) if np.any((hook_score > 1.0) & valid) else -1

    print("T", T, "first_reset", first_reset, "first_contact", first_contact,
          "first_pen", first_pen, "first_hook_proxy", first_hook)
    ev = min(x for x in (first_reset, first_pen, first_hook) if x >= 0) if any(x >= 0 for x in (first_reset, first_pen, first_hook)) else 0
    lo, hi = max(0, ev - 12), min(args.window, ev + 18)
    for i in range(lo, hi):
        j = int(np.argmin(clear[i]))
        c = corners[j]
        cj = [b[0] for b in FOOT_BOXES].index(c["idx"])
        print("%3d root=(%+.2f,%+.2f,%+.2f) foot=%s cl=%+.3f pen=%.3f cf=%.1f cf_fwd=%+.1f" %
              (i, root[i, 0], root[i, 1], root[i, 2], c["name"], clear[i, j],
               pen[i, j], c_mag[i, cj], contact_fwd[i, cj]))
    if args.out:
        row = {"tag": str(args.states).replace("\\", "/").split("/")[-4],
               "mesh": str(args.states).replace("\\", "/").split("/")[-3],
               "seed": str(args.states).replace("\\", "/").split("/")[-2],
               "first_reset": first_reset, "first_contact": first_contact,
               "first_pen": first_pen, "first_hook": first_hook,
               "min_clearance": round(float(clear[valid].min()), 4) if np.any(valid) else -1.0,
               "max_pen": round(float(pen[valid].max()), 4) if np.any(valid) else -1.0,
               "max_contact": round(float(c_mag[valid].max()), 2) if np.any(valid) else -1.0}
        from pathlib import Path as _P
        out = _P(args.out)
        if out.exists():
            with open(out) as fh:
                old = list(csv.DictReader(fh))
            fields = list(row.keys())
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(old); w.writerow(row)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
