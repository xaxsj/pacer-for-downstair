"""Turn a --reward_debug CSV into a readable table and a breakdown figure.

Usage: python script/reward_report.py [csv] [out.png]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/reward_debug.csv"
out_png = sys.argv[2] if len(sys.argv) > 2 else "output/reward_debug.png"

rows = list(csv.DictReader(open(csv_path)))
names = [r["term"] for r in rows]
means = np.array([float(r["mean"]) for r in rows])
stds = np.array([float(r["std"]) for r in rows])
shares = np.array([float(r["abs_share"]) for r in rows])

width = max(len(n) for n in names)
print(f"{'term':{width}s} {'mean':>9s} {'std':>9s} {'|share|':>8s}")
print("-" * (width + 30))
for n, m, s, sh in zip(names, means, stds, shares):
    bar = "#" * int(round(sh * 40))
    print(f"{n:{width}s} {m:+9.4f} {s:9.4f} {sh:8.1%}  {bar}")
print("-" * (width + 30))
print(f"{'task_total':{width}s} {means[-1]:+9.4f} "
      f"(0.6x = {0.6 * means[-1]:+.4f}; final adds 0.4x AMP style)")
print()
print("safety composition (weighted inside the clipped term):")
for n, w in (("swing_raw", 0.20), ("landing_raw", 0.15), ("impact_raw", 0.05)):
    i = names.index(n)
    print(f"  {n:14s} raw mean {means[i]:+.4f}  x{w:.2f} = {means[i] * w:+.4f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
order = np.argsort(means)
colors = ["#d62728" if m < 0 else "#2ca02c" for m in means[order]]
ax.barh([names[i] for i in order], means[order], color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean contribution")
ax.set_title("reward decomposition (per control step, batch mean)")
for i, v in zip(range(len(order)), means[order]):
    ax.text(v, i, f" {v:+.3f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
fig.tight_layout()
os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
fig.savefig(out_png, dpi=130)
print(f"\nsaved {out_png}")
