"""Inspect the AMP reference motion file: how many clips and what they are."""
import pickle
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/amass/pkls/amass_isaac_train_stairs_flat.pkl"
try:
    import joblib; d = joblib.load(path)
except Exception:
    d = pickle.load(open(path, "rb"))
print("type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
    for k in list(d.keys())[:6]:
        v = d[k]
        print(f"  {k}: {type(v).__name__}"
              + (f" shape {np.asarray(v).shape}" if hasattr(v, '__len__') else ""))
elif isinstance(d, (list, tuple)):
    print("len:", len(d))
    for i, m in enumerate(d[:5]):
        keys = list(m.keys())[:8] if isinstance(m, dict) else type(m).__name__
        print(f"  [{i}] {keys}")
