#!/usr/bin/env python3
"""Probe PACER observation dimensions with a single environment.

Test 1 for the 624-vs-config mismatch:
- create exactly one env on GPU 4 (set CUDA_VISIBLE_DEVICES=4 externally)
- print get_obs_size / task obs detail / wrapper observation shape
- compare with the 624-wide actor input of output/release/pacer/Humanoid.pth
No training and no PACER source modification.
"""
from __future__ import annotations

import faulthandler
import os
import sys
faulthandler.dump_traceback_later(60, repeat=False)

sys.path.insert(0, os.getcwd())

# Parse exactly the same CLI fields that pacer/run.py uses.
sys.argv = [
    "probe_obs_dims.py",
    "--task", "HumanoidPedestrianTerrain",
    "--cfg_env", "pacer/data/cfg/pacer_probe_obs.yaml",
    "--cfg_train", "pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml",
    "--num_envs", "1",
    "--motion_file", "data/amass/pkls/amass_isaac_train_stairs_flat.pkl",
    "--headless",
    "--sim_device", "cuda:0",
    "--pipeline", "gpu",
    "--physx",
    "--rl_device", "cuda:0",
    "--graphics_device_id", "0",
]

from utils.config import get_args, load_cfg, parse_sim_params  # noqa: E402
from utils.parse_task import parse_task  # noqa: E402
from pacer.utils.flags import flags  # noqa: E402

for _k, _v in {
    "server_mode": False,
    "show_traj": False,
    "height_debug": False,
    "random_heading": False,
    "fixed": False,
    "follow": False,
    "divide_group": False,
    "small_terrain": False,
    "no_collision_check": False,
    "add_proj": False,
    "start_yaw": None,
    "real_mesh": False,
    "fixed_path": False,
    "real_path": False,
    "slow": False,
    "debug": False,
    "test": False,
    "auto_record": False,
    "auto_record_video": False,
    "record_frames": 300,
    "no_virtual_display": True,
}.items():
    setattr(flags, _k, _v)

args = get_args()
cfg, cfg_train, _ = load_cfg(args)
cfg["env"]["motion_file"] = args.motion_file
sim_params = parse_sim_params(args, cfg, cfg_train)
task, env = parse_task(args, cfg, cfg_train, sim_params)

print("=== CONFIG ===")
print("terrain_obs:", cfg["env"].get("terrain_obs", None))
print("terrain_obs_type:", cfg["env"].get("terrain_obs_type", None))
print("has_shape_obs:", cfg["env"].get("has_shape_obs", None))
print("numEnvs:", cfg["env"].get("numEnvs", None))
print("numTerrains:", cfg["env"].get("terrain", {}).get("numTerrains", None))

print("=== OBS DIMS ===")
print("task.get_self_obs_size():", task.get_self_obs_size())
print("task.get_task_obs_size():", task.get_task_obs_size())
print("task.get_task_obs_size_detail():", task.get_task_obs_size_detail())
print("task.get_obs_size():", task.get_obs_size())
print("task.num_height_points:", getattr(task, "num_height_points", None))
print("env.observation_space:", env.observation_space)
print("env.observation_space.shape:", env.observation_space.shape)
print("env.amp_observation_space.shape:", getattr(env.amp_observation_space, "shape", None))

print("=== CHECKPOINT EXPECTATION ===")
print("pacer actor input: 624")
print("match:", env.observation_space.shape[0] == 624)

sys.stdout.flush()
os._exit(0)
