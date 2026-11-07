#!/usr/bin/env python3
"""Convert *all* processed KIT stair clips to PACER motionlib.

PACER's official convert_amass_isaac.py filters by the hand-curated
``amass_run_data`` list.  For stair-focused fine-tuning we want every KIT
stair clip, so this wrapper replaces that whitelist with all keys from the
processed motion file and calls the existing converter.  No PACER source file
is modified.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import joblib

import uhc.smpllib.smpl_local_robot as slr

if not hasattr(slr, "SMPL_Robot"):
    slr.SMPL_Robot = slr.Robot

from uhc.data_process import convert_amass_isaac as conv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="sample_data/amass_copycat_take5_train.pkl")
    ap.add_argument("--out_file", default="data/amass/pkls/amass_isaac_kit_stairs_all.pkl")
    args = ap.parse_args()

    data = joblib.load(args.in_file)
    keys = sorted(data.keys())
    conv.amass_run_data = keys
    print(f"all clips: {len(keys)}")
    conv.run(args.in_file, args.out_file)
    print(f"wrote {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
