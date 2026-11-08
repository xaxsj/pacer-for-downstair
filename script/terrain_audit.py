"""P0 terrain audit: build the curriculum terrain and measure what it really is.

Usage: python script/terrain_audit.py [config.yaml]

Reports, per tile: kind (straight ascend/descend/flat), measured riser, tread,
clear width W, step count, and the spawn-mask / walkable overlap.
"""
import sys

import numpy as np
import isaacgym  # noqa: F401  (must be imported before torch)
import torch
import yaml

try:
    from pacer.env.tasks.humanoid_pedestrain_terrain import Terrain
except ImportError:
    from env.tasks.humanoid_pedestrain_terrain import Terrain

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pacer/data/cfg/pacer_stage1_stairs.yaml"
env = yaml.safe_load(open(cfg_path))["env"]
tcfg = dict(env["terrain"])
tcfg["terrainType"] = "trimesh"

t = Terrain(tcfg, num_robots=2048, device="cpu")
hs, vs = t.horizontal_scale, t.vertical_scale
hf = t.height_field_raw
sm = t.spawn_mask_raw
wf = t.walkable_field_raw

print(f"map {hf.shape} px, border {t.border} px, tiles {t.env_rows}x{t.env_cols}")
print(f"walkable==0 pixels : {int((wf == 0).sum())}")
print(f"spawn mask pixels  : {int((sm > 0).sum())}")
print(f"overlap            : {int(((sm > 0) & (wf == 0)).sum())}")
print(f"unique heights     : {np.unique(hf)[:6]} ... {np.unique(hf)[-3:]}")

n_stair = 0
rows = []
for j in range(t.env_cols):
    for i in range(t.env_rows):
        x0 = t.border + i * t.length_per_env_pixels
        y0 = t.border + j * t.width_per_env_pixels
        tile = hf[x0:x0 + t.length_per_env_pixels, y0:y0 + t.width_per_env_pixels]
        tspawn = sm[x0:x0 + t.length_per_env_pixels, y0:y0 + t.width_per_env_pixels]
        has_spawn = int(tspawn.sum())
        if has_spawn == 0:
            rows.append((j, i, "flat/other", 0, 0, 0, 0, 0))
            continue
        n_stair += 1
        # corridor width: columns where the tile is not the pit
        pit_level = int(tile.min())
        col_ok = np.where((tile != pit_level).any(axis=0))[0]
        width = (col_ok.max() - col_ok.min() + 1) * hs if len(col_ok) else 0.0
        # along the corridor centre line: rises and treads
        cyc = (col_ok.min() + col_ok.max()) // 2 if len(col_ok) else 0
        line = tile[:, cyc].astype(float) * vs
        d = np.diff(line)
        rises = np.abs(d[d > 1e-3])
        riser = float(np.median(rises)) if len(rises) else 0.0
        idx = np.where(np.abs(d) > 1e-3)[0]
        tread = float(np.median(np.diff(idx)) * hs) if len(idx) > 1 else 0.0
        start_px = int(np.argmax(np.abs(d) > 1e-3)) if len(idx) else -1
        d_start = start_px * hs if start_px >= 0 else -1
        direction = "asc" if (len(idx) and d[idx[0]] > 0) else "desc"
        rows.append((j, i, direction, riser, tread, width, len(idx), d_start))

print(f"\nstair tiles: {n_stair} / {t.env_rows * t.env_cols}")
print("  j  i  dir   riser   tread   width   steps  D_start")
for r in rows[:14]:
    print("  {:2d} {:2d}  {:5s} {:6.3f}  {:6.3f}  {:6.2f}  {:5d}  {:6.2f}".format(*r))

st = [r for r in rows if r[2] in ("asc", "desc")]
if st:
    print("\nmeasured ranges over stair tiles:")
    for k, name in ((3, "riser"), (4, "tread"), (5, "clear width"), (6, "steps"), (7, "D_start")):
        vals = [r[k] for r in st]
        print(f"  {name:12s} min {min(vals):6.3f}  median {np.median(vals):6.3f}  max {max(vals):6.3f}")
    asc = sum(1 for r in st if r[2] == "asc")
    print(f"  ascend {asc} / {len(st)}")


