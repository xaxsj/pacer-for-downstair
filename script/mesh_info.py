"""Report riser / tread / corridor width for every stair mesh in data/mesh."""
import glob
import os
import pickle

import numpy as np


def analyse(path):
    md = pickle.load(open(path, "rb"))
    hm = np.asarray(md["heigthmap"], dtype=float)
    cam = np.asarray(md["cam_pos"], dtype=float)
    xs, ys = float(md["x_scale"]), float(md["y_scale"])
    nx, ny = hm.shape
    x_min = cam[0] - (nx / 2.0) / xs
    y_min = cam[1] - (ny / 2.0) / ys

    def h(x, y):
        px = int(np.clip(np.floor((x - x_min) * xs), 0, nx - 1))
        py = int(np.clip(np.floor((y - y_min) * ys), 0, ny - 1))
        return float(hm[px, py])

    # walk along +x at y=0, collect height plateaus
    hs = [h(x, 0.0) for x in np.arange(0.0, 6.0, 0.01)]
    levels = []
    for v in hs:
        if not levels or abs(v - levels[-1]) > 0.02:
            levels.append(v)
    rises = np.diff(levels)
    riser = float(np.median(rises)) if len(rises) else 0.0

    # tread: distance between consecutive equal-height plateaus
    trans = []
    for i in range(1, len(hs)):
        if abs(hs[i] - hs[i - 1]) > 0.02:
            trans.append(i * 0.01)
    tread = float(np.median(np.diff(trans))) if len(trans) > 1 else 0.0

    # corridor width: y-extent where height > 0.05 at x = first_stair + 0.6
    x_probe = (trans[0] + 0.6) if trans else 2.5
    ys_ok = [y for y in np.arange(-3.0, 3.0, 0.02) if h(x_probe, y) > 0.05]
    width = (max(ys_ok) - min(ys_ok)) if ys_ok else 0.0

    print(f"{os.path.basename(path):22s} riser={riser:5.3f}  tread={tread:5.3f}  "
          f"width={width:5.2f}  stair_start_x={trans[0] if trans else -1:5.2f}")


for p in sorted(glob.glob("data/mesh/*.pkl")):
    try:
        analyse(p)
    except Exception as exc:  # noqa: BLE001
        print(f"{os.path.basename(p):22s} FAILED {type(exc).__name__}: {exc}")
