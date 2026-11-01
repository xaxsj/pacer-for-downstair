import pickle, sys, numpy as np

d = pickle.load(open(sys.argv[1], "rb"))
rb = np.asarray(d["rigid_body_pos"])   # (T,E,24,3)
rs = np.asarray(d["root_states"])
T = rb.shape[0]
print("shape", rb.shape, "root frame0", np.round(rs[0, 0].tolist(), 3))

b0 = rb[0, 0]           # (24,3)
print("\n=== frame 0: 24 bodies, sorted by height (z) ===")
order = np.argsort(b0[:, 2])[::-1]
for i in order:
    p = b0[i]
    print(" idx %2d  z=%.3f   x=%.3f y=%.3f   horiz_dist_from_pelvis=%.3f" % (
        i, p[2], p[0], p[1], np.linalg.norm(p[:2] - b0[0, :2])))

print("\n=== joint0 (pelvis) over frames ===")
print(np.round(rb[:15, 0, 0], 3).tolist())
print("\n=== highest body idx over first 15 frames ===")
for f in range(15):
    hi = int(np.argmax(rb[f, 0, :, 2])); lo = int(np.argmin(rb[f, 0, :, 2]))
    print(" f%2d  top idx=%2d z=%.3f | bottom idx=%2d z=%.3f | pelvis z=%.3f" % (
        f, hi, rb[f, 0, hi, 2], lo, rb[f, 0, lo, 2], rb[f, 0, 0, 2]))
print("\ndof_pos frame0[:8]", np.round(np.asarray(d["dof_pos"])[0, 0, :8], 3).tolist())
