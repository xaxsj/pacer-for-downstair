"""CPU/GPU 物理下对比三角网格碰撞高度。"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/isaacgym_pkg/isaacgym/python"))
from isaacgym import gymapi


def make_sim(gym, use_gpu=False):
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.use_gpu_pipeline = use_gpu
    sp.physx.use_gpu = use_gpu
    return gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)


def add_mesh(gym, sim, data):
    v = np.asarray(data["vertices"], dtype=np.float32)
    f = np.asarray(data["faces"], dtype=np.uint32)
    tm = gymapi.TriangleMeshParams()
    tm.nb_vertices = v.shape[0]
    tm.nb_triangles = f.shape[0]
    tm.transform.p = gymapi.Vec3(0, 0, 0)
    tm.static_friction = 1.0
    tm.dynamic_friction = 1.0
    gym.add_triangle_mesh(sim, v.flatten(order="C"), f.flatten(order="C"), tm)


def expected_z(x):
    if x < -0.40:
        return None
    if x < 0.80:
        return 3.00
    idx = int((x - 0.80) // 0.32)
    if idx >= 20:
        return 0.00
    return 3.00 - (idx + 1) * 0.15


ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--steps", type=int, default=180)
ap.add_argument("--gpu", action="store_true")
args = ap.parse_args()

data = pickle.load(open(args.mesh, "rb"))
v = np.asarray(data["vertices"])
print("[mesh]", args.mesh, "gpu=", args.gpu)
print("  vertices", v.shape, "faces", np.asarray(data["faces"]).shape)

gym = gymapi.acquire_gym()
sim = make_sim(gym, use_gpu=args.gpu)
add_mesh(gym, sim, data)

sp = 30.0
env = gym.create_env(sim, gymapi.Vec3(-sp, -sp, 0.0), gymapi.Vec3(sp, sp, sp), 1)

xs = list(np.round(np.arange(-0.2, 8.21, 0.4), 2))
opts = gymapi.AssetOptions()
opts.density = 1000.0
R = 0.03
asset = gym.create_sphere(sim, R, opts)

actors = []
for x in xs:
    base = expected_z(x)
    if base is None:
        base = 0.0
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(x), 0.0, float(base + 0.30))
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actors.append(gym.create_actor(env, asset, pose, "p_%s" % x, 0, 0))

gym.prepare_sim(sim)
for _ in range(args.steps):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

print("\n    x    期望    实测")
bad = total = 0
for x, actor in zip(xs, actors):
    st = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    z = float(st["pose"]["p"]["z"][0]) - R
    exp = expected_z(x)
    if exp is None:
        continue
    total += 1
    flag = ""
    if abs(exp - z) > 0.05:
        bad += 1
        flag = "  <-- 不一致"
    print("  %5.2f  %6.2f  %6.2f%s" % (x, exp, z, flag))
print("\n不一致 %d / %d" % (bad, total))