import pickle, sys, numpy as np

p = sys.argv[1]
d = pickle.load(open(p, "rb"))
print("keys:", list(d.keys()))
rs = np.asarray(d["root_states"])      # (T,E,13)
pos = rs[..., 0:3]
T, E, _ = pos.shape
print("T=%d E=%d" % (T, E))
x = pos[:, 0, 0]; y = pos[:, 0, 1]; z = pos[:, 0, 2]
print("start (x,y,z) = %.3f %.3f %.3f" % (x[0], y[0], z[0]))
print("end   (x,y,z) = %.3f %.3f %.3f" % (x[-1], y[-1], z[-1]))
print("delta  x=%+.3f  y=%+.3f  z=%+.3f" % (x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]))
print("max |y| drift  = %.3f" % np.abs(y - y[0]).max())
print("min z / max z  = %.3f / %.3f" % (z.min(), z.max()))

# reset detection: z jumps up by > 0.4 m in one step
dz = np.diff(z)
resets = np.where(dz > 0.4)[0]
print("resets (z jump > 0.4) = %d at frames %s" % (len(resets), (resets+1).tolist()[:20]))

print("\n frame   x       y       z")
for i in range(0, T, max(1, T//20)):
    print("%6d %7.3f %7.3f %7.3f" % (i, x[i], y[i], z[i]))
print("%6d %7.3f %7.3f %7.3f" % (T-1, x[-1], y[-1], z[-1]))
