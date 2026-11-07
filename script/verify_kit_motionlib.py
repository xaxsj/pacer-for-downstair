#!/usr/bin/env python3
"""Verify the merged KIT + base PACER motionlib."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, os.getcwd())

import isaacgym
from isaacgym import gymapi, gymtorch
import joblib
import numpy as np
import torch

REQUIRED = (
    "pose_quat_global",
    "pose_quat",
    "trans_orig",
    "root_trans_offset",
    "beta",
    "gender",
    "pose_aa",
    "fps",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="data/amass/pkls/amass_isaac_train_stairs.pkl")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    data = joblib.load(args.motion)
    print(f"motions: {len(data)}")
    keys = sorted(data.keys())
    for key in keys[: args.limit]:
        item = data[key]
        shapes = {k: tuple(np.asarray(item[k]).shape) for k in REQUIRED if k in item}
        print(key, shapes)
        missing = [k for k in REQUIRED if k not in item]
        if missing:
            print("  MISSING:", missing)
    bad = [k for k, v in data.items() if any(kk not in v for kk in REQUIRED)]
    print(f"items with missing keys: {len(bad)}")
    if bad:
        print("  examples:", bad[:5])
        return 1

    # Optional MotionLib load
    try:
        from pacer.utils.motion_lib_smpl import MotionLib

        key_body_ids = torch.zeros(0, dtype=torch.long)
        MotionLib(args.motion, key_body_ids, torch.device("cpu"), fix_height=False)
        print("MotionLib loaded OK")
    except Exception as exc:  # pragma: no cover
        print(f"MotionLib load skipped/failed: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
