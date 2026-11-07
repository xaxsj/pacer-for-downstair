#!/usr/bin/env python3
"""Validate the paper-style dense foot/riser collision risk offline.

This script does not alter PACER or its reward.  It replays recorded rigid-body
states against the recorded scene height map and evaluates whether

    risk = max(0, v_foot_xy dot d_hat) * max(0, 1 - ||d|| / d_unsafe)

detects a steep rising edge before a measured foot/riser contact.  It also
reports activation rates on locally flat and descending terrain.

Single case:
  python script/validate_collision_risk.py \
    --states experiment_result/validation/release/wide_stairs/seed_0/states.pkl \
    --mesh data/mesh/wide_stairs.pkl

Validation tree:
  python script/validate_collision_risk.py \
    --root experiment_result/validation \
    --out-dir experiment_result/collision_risk_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np


# PACER SMPL rigid-body order and collision-box bottom offsets.
FEET = (
    (3, "L_Ankle", -0.0487),
    (4, "L_Toe", +0.0070),
    (7, "R_Ankle", -0.0487),
    (8, "R_Toe", +0.0070),
)


class HeightMap:
    def __init__(self, mesh_data):
        self.values = np.asarray(mesh_data["heigthmap"], dtype=np.float64)
        cam = np.asarray(mesh_data["cam_pos"], dtype=np.float64)
        self.x_scale = float(mesh_data["x_scale"])
        self.y_scale = float(mesh_data["y_scale"])
        nx, ny = self.values.shape
        self.x_min = float(cam[0]) - (nx / 2.0) / self.x_scale
        self.y_min = float(cam[1]) - (ny / 2.0) / self.y_scale
        self.resolution = max(1.0 / self.x_scale, 1.0 / self.y_scale)

    def sample(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        ix = np.floor((xy[..., 0] - self.x_min) * self.x_scale).astype(np.int64)
        iy = np.floor((xy[..., 1] - self.y_min) * self.y_scale).astype(np.int64)
        ix = np.clip(ix, 0, self.values.shape[0] - 1)
        iy = np.clip(iy, 0, self.values.shape[1] - 1)
        return self.values[ix, iy]


def unit(vector, eps=1e-8):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / norm if norm > eps else None


def rotate_2d(vector, angle_deg):
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    x, y = float(vector[0]), float(vector[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float64)


def finite_difference(values, dt):
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        out[1:] = np.diff(values, axis=0) / dt
        out[0] = out[1]
    return out


def first_reset_frame(data, root):
    candidates = list(np.where(np.diff(root[:, 2]) > 0.4)[0] + 1)
    if "progress" in data:
        progress = np.asarray(data["progress"])
        if progress.size:
            progress = progress.reshape(root.shape[0], -1)[:, 0]
            candidates.extend((np.where(np.diff(progress) < -5)[0] + 1).tolist())
    return min(candidates) if candidates else -1


def nearest_riser(height_map, origin, direction, unsafe_distance, cone_deg,
                  rays, height_threshold, slope_threshold):
    """Return the nearest steep height increase inside the velocity cone."""
    direction = unit(direction)
    if direction is None:
        return None
    base_h = float(height_map.sample(origin))
    step = min(0.01, height_map.resolution / 2.0)
    distances = np.arange(step, unsafe_distance + 0.5 * step, step)
    best = None
    for angle in np.linspace(-cone_deg, cone_deg, rays):
        ray = rotate_2d(direction, float(angle))
        previous_h = base_h
        for distance in distances:
            point = origin + ray * distance
            current_h = float(height_map.sample(point))
            local_slope = (current_h - previous_h) / step
            if (current_h > base_h + height_threshold
                    and local_slope > slope_threshold):
                candidate = {
                    "distance": float(distance),
                    "direction": ray,
                    "top": current_h,
                    "rise": current_h - base_h,
                    "slope": local_slope,
                }
                if best is None or candidate["distance"] < best["distance"]:
                    best = candidate
                break
            previous_h = current_h
    return best


def terrain_mode(height_map, xy, direction, lookahead=0.25, threshold=0.04):
    direction = unit(direction)
    if direction is None:
        return "flat"
    here = float(height_map.sample(xy))
    ahead = float(height_map.sample(xy + direction * lookahead))
    delta = ahead - here
    if delta > threshold:
        return "ascent"
    if delta < -threshold:
        return "descent"
    return "flat"


def local_relief(height_map, xy, radius):
    samples = [np.asarray(xy, dtype=np.float64)]
    for angle in np.linspace(0.0, 315.0, 8):
        samples.append(np.asarray(xy, dtype=np.float64)
                       + rotate_2d((radius, 0.0), float(angle)))
    heights = np.asarray(height_map.sample(np.stack(samples)))
    return float(heights.max() - heights.min())


def classify_frame(height_map, root_xy, root_velocity, foot_xy,
                   flat_radius, height_threshold):
    # "Normal flat" means that neither the root nor any foot has a height
    # discontinuity nearby.  An approach slab immediately before a staircase
    # is deliberately excluded: activation there is the desired anticipation,
    # not a flat-ground false positive.
    probes = [root_xy] + [xy for xy in foot_xy]
    if max(local_relief(height_map, xy, flat_radius) for xy in probes) \
            <= height_threshold:
        return "flat"
    trend = terrain_mode(height_map, root_xy, root_velocity,
                         lookahead=flat_radius,
                         threshold=height_threshold)
    return trend if trend != "flat" else "edge"


def collapse_events(events, merge_frames=3):
    """Merge simultaneous ankle/toe reports into physical collision events."""
    merged = []
    for event in sorted(events, key=lambda item: item["frame"]):
        if merged and event["frame"] - merged[-1]["frame"] <= merge_frames:
            if event["foot"] not in merged[-1]["bodies"]:
                merged[-1]["bodies"].append(event["foot"])
            merged[-1]["horizontal_force"] = max(
                merged[-1]["horizontal_force"], event["horizontal_force"])
        else:
            merged.append({
                "frame": event["frame"],
                "bodies": [event["foot"]],
                "horizontal_force": event["horizontal_force"],
            })
    return merged


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else 0.0


def max_positive_run(values, modes, target_mode, threshold):
    longest = current = 0
    for value, mode in zip(values, modes):
        if mode == target_mode and value >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def analyse_case(states_path, mesh_path, args):
    with open(states_path, "rb") as handle:
        data = pickle.load(handle)
    with open(mesh_path, "rb") as handle:
        height_map = HeightMap(pickle.load(handle))

    env = args.env
    root = np.asarray(data["root_states"], dtype=np.float64)[:, env]
    body_pos = np.asarray(data["rigid_body_pos"], dtype=np.float64)[:, env]
    dt = float(np.asarray(data.get("dt", 1.0 / 30.0)).reshape(-1)[0])
    body_vel = (np.asarray(data["rigid_body_vel"], dtype=np.float64)[:, env]
                if "rigid_body_vel" in data
                else finite_difference(body_pos, dt))
    contact = (np.asarray(data["contact_forces"], dtype=np.float64)[:, env]
               if "contact_forces" in data else None)

    reset_frame = first_reset_frame(data, root)
    valid_frames = reset_frame if reset_frame >= 0 else len(root)
    valid_frames = max(0, min(valid_frames, len(root)))
    root_vel = finite_difference(root[:, :2], dt)

    mode_by_frame = []
    foot_ids = [item[0] for item in FEET]
    for frame in range(valid_frames):
        mode_by_frame.append(classify_frame(
            height_map,
            root[frame, :2],
            root_vel[frame],
            body_pos[frame, foot_ids, :2],
            args.unsafe_distance + 0.05,
            args.height_threshold))

    frame_rows = []
    risk_by_body = {}
    event_candidates = []
    for body_id, foot_name, bottom_offset in FEET:
        risks = np.zeros(len(root), dtype=np.float64)
        event_active = False
        for frame in range(valid_frames):
            xy = body_pos[frame, body_id, :2]
            velocity = body_vel[frame, body_id, :2]
            speed = float(np.linalg.norm(velocity))
            direction = unit(velocity)
            riser = None
            risk = 0.0
            if direction is not None and speed >= args.min_foot_speed:
                riser = nearest_riser(
                    height_map, xy, direction, args.unsafe_distance,
                    args.cone_deg, args.rays, args.height_threshold,
                    args.slope_threshold)
                if riser is not None:
                    approach = max(0.0, float(np.dot(velocity, riser["direction"])))
                    distance_weight = max(
                        0.0, 1.0 - riser["distance"] / args.unsafe_distance)
                    risk = approach * distance_weight
            risks[frame] = risk

            mode = mode_by_frame[frame]
            obstacle_distance = riser["distance"] if riser else -1.0
            obstacle_top = riser["top"] if riser else float(height_map.sample(xy))
            gap = float(body_pos[frame, body_id, 2] + bottom_offset - obstacle_top)
            horizontal_force = (float(np.linalg.norm(contact[frame, body_id, :2]))
                                if contact is not None else 0.0)

            # Sparse ground-truth proxy: horizontal foot force while the foot is
            # immediately beside a rising edge and not safely above its top.
            collision = bool(
                riser is not None
                and obstacle_distance <= args.contact_distance
                and gap <= args.contact_gap
                and horizontal_force >= args.horizontal_force)
            if collision and not event_active:
                event_candidates.append({
                    "frame": frame,
                    "foot": foot_name,
                    "horizontal_force": horizontal_force,
                })
            event_active = collision

            frame_rows.append({
                "frame": frame,
                "foot": foot_name,
                "mode": mode,
                "risk": risk,
                "foot_speed": speed,
                "riser_distance": obstacle_distance,
                "riser_top": obstacle_top,
                "foot_bottom_gap": gap,
                "horizontal_force": horizontal_force,
                "collision_proxy": int(collision),
            })
        risk_by_body[foot_name] = risks

    # The reward sees all feet; use the maximum per frame for false positives.
    max_risk = np.max(np.stack(list(risk_by_body.values()), axis=1), axis=1)
    events = collapse_events(event_candidates)
    detected = 0
    detected_1_5 = 0
    event_details = []
    for event in events:
        frame = int(event["frame"])
        lo, hi = max(0, frame - 5), max(0, frame - 1)  # frames t-5 ... t-2
        pre_risk = float(max_risk[lo:hi].max()) if hi > lo else 0.0
        pre_risk_1_5 = float(max_risk[lo:frame].max()) if frame > lo else 0.0
        is_detected = pre_risk >= args.risk_threshold
        is_detected_1_5 = pre_risk_1_5 >= args.risk_threshold
        detected += int(is_detected)
        detected_1_5 += int(is_detected_1_5)
        event_details.append({
            **event,
            "pre_2_5_max_risk": pre_risk,
            "detected": is_detected,
            "pre_1_5_max_risk": pre_risk_1_5,
            "detected_1_5": is_detected_1_5,
        })

    mode_stats = {}
    mode_array = np.asarray(mode_by_frame)
    for mode in ("flat", "ascent", "descent", "edge"):
        values = max_risk[:valid_frames][mode_array == mode]
        mode_stats[mode] = {
            "frames": int(values.size),
            "positive_rate": float(np.mean(values >= args.risk_threshold)) if values.size else 0.0,
            "p95": percentile(values, 95),
            "max": float(values.max()) if values.size else 0.0,
            "max_positive_run": max_positive_run(
                max_risk[:valid_frames], mode_array, mode,
                args.risk_threshold),
        }

    recall = float(detected / len(events)) if events else None
    summary = {
        "states": str(states_path),
        "mesh": str(mesh_path),
        "frames": int(len(root)),
        "valid_frames": int(valid_frames),
        "reset_frame": int(reset_frame),
        "collision_events": int(len(events)),
        "pre_2_5_detected": int(detected),
        "pre_2_5_recall": recall,
        "pre_1_5_detected": int(detected_1_5),
        "pre_1_5_recall": (float(detected_1_5 / len(events))
                           if events else None),
        "risk_threshold": float(args.risk_threshold),
        "flat_positive_rate": mode_stats["flat"]["positive_rate"],
        "descent_positive_rate": mode_stats["descent"]["positive_rate"],
        "flat_p95": mode_stats["flat"]["p95"],
        "descent_p95": mode_stats["descent"]["p95"],
        "mode_stats": mode_stats,
        "events": event_details,
    }
    return summary, frame_rows


def discover_cases(root):
    for states_path in sorted(Path(root).rglob("states.pkl")):
        mesh_name = states_path.parent.parent.name
        mesh_path = Path("data/mesh") / f"{mesh_name}.pkl"
        if mesh_path.is_file():
            yield states_path, mesh_path


def safe_slug(path):
    return "__".join(Path(path).with_suffix("").parts[-5:])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--states")
    source.add_argument("--root")
    parser.add_argument("--mesh")
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--out-dir", default="experiment_result/collision_risk_validation")
    parser.add_argument("--unsafe-distance", type=float, default=0.20)
    parser.add_argument("--cone-deg", type=float, default=30.0)
    parser.add_argument("--rays", type=int, default=5)
    parser.add_argument("--height-threshold", type=float, default=0.03)
    parser.add_argument("--slope-threshold", type=float, default=0.5)
    parser.add_argument("--min-foot-speed", type=float, default=0.05)
    parser.add_argument("--risk-threshold", type=float, default=0.05)
    parser.add_argument("--horizontal-force", type=float, default=10.0)
    parser.add_argument("--contact-distance", type=float, default=0.06)
    parser.add_argument("--contact-gap", type=float, default=0.04)
    args = parser.parse_args()

    if args.states:
        if not args.mesh:
            parser.error("--mesh is required with --states")
        cases = [(Path(args.states), Path(args.mesh))]
    else:
        cases = list(discover_cases(args.root))
        if not cases:
            parser.error(f"no validation cases found below {args.root}")

    out_dir = Path(args.out_dir)
    summaries = []
    for states_path, mesh_path in cases:
        summary, frames = analyse_case(states_path, mesh_path, args)
        summaries.append(summary)
        slug = safe_slug(states_path)
        write_csv(out_dir / "frames" / f"{slug}.csv", frames)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    flat_rows = []
    for item in summaries:
        flat_rows.append({
            key: value for key, value in item.items()
            if key not in ("mode_stats", "events")
        })
    write_csv(out_dir / "summary.csv", flat_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    total_events = sum(item["collision_events"] for item in summaries)
    total_detected = sum(item["pre_2_5_detected"] for item in summaries)
    total_detected_1_5 = sum(item["pre_1_5_detected"] for item in summaries)
    flat_frames = sum(item["mode_stats"]["flat"]["frames"] for item in summaries)
    flat_positive = sum(
        item["mode_stats"]["flat"]["positive_rate"]
        * item["mode_stats"]["flat"]["frames"] for item in summaries)
    descent_frames = sum(item["mode_stats"]["descent"]["frames"] for item in summaries)
    descent_positive = sum(
        item["mode_stats"]["descent"]["positive_rate"]
        * item["mode_stats"]["descent"]["frames"] for item in summaries)
    aggregate = {
        "cases": len(summaries),
        "collision_events": total_events,
        "pre_2_5_detected": total_detected,
        "pre_2_5_recall": (total_detected / total_events
                            if total_events else None),
        "pre_1_5_detected": total_detected_1_5,
        "pre_1_5_recall": (total_detected_1_5 / total_events
                            if total_events else None),
        "pure_flat_frames": flat_frames,
        "pure_flat_positive_rate": (flat_positive / flat_frames
                                    if flat_frames else None),
        "descent_frames": descent_frames,
        "descent_positive_rate": (descent_positive / descent_frames
                                  if descent_frames else None),
        "descent_max_positive_run": max(
            (item["mode_stats"]["descent"]["max_positive_run"]
             for item in summaries), default=0),
    }
    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    print("aggregate", json.dumps(aggregate, ensure_ascii=False))
    print(f"wrote {len(summaries)} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
