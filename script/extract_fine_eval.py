import os
import imageio.v2 as imageio
import numpy as np
for name in ["base_mixed", "stage1_mixed"]:
    path = f"experiment_result/eval_stage1/{name}.mp4"
    reader = imageio.get_reader(path)
    frames = []
    for sec in [0, 2, 4, 6, 8, 10]:
        try:
            frames.append(np.asarray(reader.get_data(sec * 30))[..., :3])
        except Exception as e:
            print(name, sec, type(e).__name__, e)
    reader.close()
    if frames:
        h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
        sheet = np.concatenate([f[:h, :w] for f in frames], axis=1)
        out = f"experiment_result/eval_stage1/frames/{name}_fine.png"
        imageio.imwrite(out, sheet)
        print(out, sheet.shape)
