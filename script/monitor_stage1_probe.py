#!/usr/bin/env python3
"""Non-invasive progress/reward probe for the running Stage 1 job.

It only tails the training log and stats the checkpoint directory; it never
signals, pauses, or restarts the trainer.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import time
from datetime import datetime, timedelta

EP_RE = re.compile(
    r"Ep:\s*(?P<ep>\d+).*?"
    r"rwd:\s*(?P<rwd>-?\d+(?:\.\d+)?).*?"
    r"fps_step:\s*(?P<fps>-?\d+(?:\.\d+)?).*?"
    r"eps_len:\s*(?P<eps_len>-?\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/data/ljx/tmp/pacer_stage1_train.log")
    ap.add_argument("--start-epoch", type=int, default=41250)
    ap.add_argument("--target-epoch", type=int, default=41750)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--from-start", action="store_true")
    ap.add_argument("--checkpoint-dir", default="output/exp/pacer_stage1_stairs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.log):
        print(f"waiting for log: {args.log}", flush=True)
        while not os.path.exists(args.log):
            time.sleep(2)

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        if not args.from_start:
            f.seek(0, os.SEEK_END)

        rewards = collections.deque(maxlen=args.window)
        first_time = None
        first_ep = None
        last_ep = None
        last_time = None

        while True:
            line = f.readline()
            if not line:
                time.sleep(args.interval)
                continue

            m = EP_RE.search(line)
            if not m:
                continue

            ep = int(m.group("ep"))
            rwd = float(m.group("rwd"))
            fps = float(m.group("fps"))
            eps_len = float(m.group("eps_len"))
            now = time.time()

            if first_time is None:
                first_time, first_ep = now, ep
            last_ep, last_time = ep, now
            rewards.append(rwd)

            done = max(0, ep - args.start_epoch)
            total = max(1, args.target_epoch - args.start_epoch)
            pct = 100.0 * done / total
            mean = sum(rewards) / len(rewards)
            recent = ", ".join(f"{x:.1f}" for x in list(rewards)[-5:])

            eta = ""
            if done > 0 and first_ep is not None and first_time is not None:
                elapsed = max(1e-6, now - first_time)
                rate = done / elapsed
                remain = (args.target_epoch - ep) / rate if rate > 0 else 0
                eta = f"  eta={timedelta(seconds=int(remain))}"

            ckpt = ""
            path = os.path.join(args.checkpoint_dir, "Humanoid.pth")
            if os.path.exists(path):
                st = os.stat(path)
                ckpt = f"  ckpt={st.st_size/1e6:.0f}MB@{datetime.fromtimestamp(st.st_mtime).strftime('%H:%M:%S')}"

            print(
                f"{datetime.now().strftime('%H:%M:%S')} "
                f"Ep {ep}/{args.target_epoch} ({pct:5.1f}%) "
                f"rwd={rwd:7.1f} mean{len(rewards)}={mean:7.1f} "
                f"fps={fps:7.0f} eps_len={eps_len:6.1f}{eta}{ckpt} "
                f"recent=[{recent}]",
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
