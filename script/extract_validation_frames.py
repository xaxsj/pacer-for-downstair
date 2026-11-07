import os
import imageio.v2 as imageio
import numpy as np
for rel in ["stage1/narrow_stairs/seed_0", "release/narrow_stairs/seed_0", "stage1/wide_stairs/seed_0", "diverse/narrow_stairs/seed_0"]:
    path = f"experiment_result/validation/{rel}/video.mp4"
    if not os.path.exists(path):
        print("missing", path); continue
    r = imageio.get_reader(path)
    frames = []
    for i in [0, 30, 60, 90, 120, 150, 180, 240, 299]:
        try: frames.append(np.asarray(r.get_data(i))[..., :3])
        except Exception as e: print("frame", i, type(e).__name__); break
    r.close()
    if frames:
        h=min(f.shape[0] for f in frames); w=min(f.shape[1] for f in frames)
        sheet=np.concatenate([f[:h,:w] for f in frames],axis=1)
        out=f"experiment_result/validation/{rel.replace('/','_')}_sheet.png"
        imageio.imwrite(out,sheet); print("wrote",out,sheet.shape)
