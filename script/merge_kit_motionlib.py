#!/usr/bin/env python3
"""Merge the KIT stair motionlib into PACER's base motion file and report new clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sample_data/amass_isaac_standing_upright_slim.pkl")
    ap.add_argument("--stairs", default="data/amass/pkls/amass_isaac_kit_stairs.pkl")
    ap.add_argument("--out", default="data/amass/pkls/amass_isaac_train_stairs.pkl")
    ap.add_argument("--report", default="data/amass/pkls/kit_stairs_report.json")
    args = ap.parse_args()

    base = joblib.load(args.base)
    stairs = joblib.load(args.stairs)
    before = set(base.keys())
    new_keys = sorted(set(stairs.keys()) - before)

    print(f"base motions:   {len(base)}")
    print(f"stair motions:  {len(stairs)}")
    print(f"new clips:      {len(new_keys)}")
    for key in new_keys[:30]:
        print("  +", key)
    if len(new_keys) > 30:
        print(f"  ... and {len(new_keys) - 30} more")

    base.update(stairs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, args.out)
    print(f"merged:         {len(base)} -> {args.out}")

    report = {
        "base": len(before),
        "stairs": len(stairs),
        "new": len(new_keys),
        "merged": len(base),
        "out": args.out,
        "new_keys": new_keys,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report:         {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
