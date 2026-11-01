import pickle, sys, numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rs = np.asarray(d["root_states"]); rb = np.asarray(d["rigid_body_pos"])   # (T,E,13) (T,E,24,3)
pos = rs[..., 0:3]
T = pos.shape[0]
print("progress:", d.get("progress"))
print("dt:", d.get("dt"), "path_file:", d.get("path_file"))

# SMPL 24 joints: 0 pelvis, 12 neck, 15 head, 10/11 feet
pelvis = rb[:, 0, 0, :]; neck = rb[:, 0, 12, :]; head = rb[:, 0, 15, :]
lf = rb[:, 0, 10, :]; rf = rb[:, 0, 11, :]

up = head - pelvis
ang = np.degrees(np.arccos(np.clip(up[:, 2] / (np.linalg.norm(up, axis=1) + 1e-9), -1, 1)))
print("\n frame  root_z  head_z  foot_z  up_angle_deg  root_xy")
for i in list(range(0, 20)) + list(range(20, T, 20)):
    print("%6d %7.3f %7.3f %7.3f %8.1f      %6.2f %6.2f" % (
        i, pos[i, 0, 2], head[i, 2], min(lf[i, 2], rf[i, 2]), ang[i], pos[i, 0, 0], pos[i, 0, 1]))

print("\nupright frames (angle<30deg): %d / %d" % (int((ang < 30).sum()), T))
print("mean angle first 13 frames: %.1f" % ang[:13].mean())
print("root z first 13: ", np.round(pos[:13, 0, 2], 3).tolist())
