"""Print the reward / episode-length trend from a PACER training log.

Usage: python script/reward_trend.py <log> [stride]
"""
import re
import sys

path = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 100

pat = re.compile(r"Ep:\s*(\d+)\s+rwd:\s*(-?[\d.]+).*?eps_len:\s*([\d.]+)")
rows = []
for line in open(path, errors="ignore"):
    m = pat.search(line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

if not rows:
    print("no epoch lines found")
    raise SystemExit

print(f"{'epoch':>8} {'rwd':>8} {'eps_len':>8}")
start = rows[0][0]
for e, r, l in rows:
    if (e - start) % stride == 0 or e == rows[-1][0]:
        print(f"{e:8d} {r:8.1f} {l:8.1f}")

first, last = rows[0], rows[-1]
n = len(rows)
print(f"\n{start} -> {last[0]}   ({n} epochs)")
print(f"rwd      {first[1]:.1f} -> {last[1]:.1f}")
print(f"eps_len  {first[2]:.1f} -> {last[2]:.1f}")
tail = [r for _, r, _ in rows[-50:]]
print(f"last-50 mean rwd {sum(tail) / len(tail):.1f}")
