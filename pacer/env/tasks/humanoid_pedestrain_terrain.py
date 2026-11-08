# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from shutil import ExecError
import torch
import torch.nn.functional as F
import numpy as np

import env.tasks.humanoid_traj as humanoid_traj
from isaacgym import gymapi
from isaacgym.torch_utils import *
from env.tasks.humanoid import dof_to_obs
from env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from pacer.utils.flags import flags
from utils import torch_utils
from isaacgym import gymtorch
import joblib
from poselib.poselib.core.rotation3d import quat_inverse, quat_mul
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from typing import OrderedDict
from pacer.utils.draw_utils import agt_color
from pacer.env.tasks.humanoid import compute_humanoid_observations_smpl_max, compute_humanoid_observations_smpl,\
    compute_humanoid_observations_max, compute_humanoid_observations,\
      ENABLE_MAX_COORD_OBS

HACK_MOTION_SYNC = False

class HumanoidPedestrianTerrain(humanoid_traj.HumanoidTraj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        ## ZL Hack to get the height map to load.
        self.real_mesh = cfg['args'].real_mesh
        self.device = "cpu"
        self.device_type = device_type
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)
        
        self.load_smpl_configs(cfg)
        self.cfg = cfg
        self.num_envs = cfg["env"]["numEnvs"]
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.headless = cfg["headless"]
        self.sensor_extent = cfg["env"].get("sensor_extent", 2)
        self.sensor_res = cfg["env"].get("sensor_res", 32)
        self.power_reward = cfg["env"].get("power_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.fuzzy_target = cfg["env"].get("fuzzy_target", False)

        # Terrain-aware target paths: instead of a random heading, start the
        # trajectory along the local terrain gradient (doc: the robot must
        # actually approach the stairs, not wander).
        #   stair_heading_mode: "random" | "uphill" | "downhill"
        self.stair_heading_mode = str(cfg["env"].get("stair_heading_mode", "random"))
        self.stair_path_turn_scale = float(cfg["env"].get("stair_path_turn_scale", 1.0))
        self.stair_heading_probe = float(cfg["env"].get("stair_heading_probe", 0.30))

        # Terrain-aware stepping rewards.  The individual terms are exposed in
        # reward_raw for diagnosis, but enter PPO through one bounded safety
        # module so that adding safety objectives cannot swamp trajectory/AMP.
        self.enable_collision_penalty = cfg["env"].get("enable_collision_penalty", False)
        self.enable_edge_penalty = cfg["env"].get("enable_edge_penalty", False)
        self.enable_safe_stepping_reward = cfg["env"].get("enable_safe_stepping_reward", False)
        self.enable_velocity_reward = cfg["env"].get("enable_velocity_reward", False)
        self.collision_reward_weight = float(cfg["env"].get("collision_reward_weight", 0.20))
        self.edge_reward_weight = float(cfg["env"].get("edge_reward_weight", 0.10))
        self.velocity_reward_weight = float(cfg["env"].get("velocity_reward_weight", 0.10))
        self.landing_reward_weight = float(cfg["env"].get("landing_reward_weight", 0.15))
        self.impact_reward_weight = float(cfg["env"].get("impact_reward_weight", 0.05))
        self.safe_reward_clip = float(cfg["env"].get("safe_reward_clip", 0.25))
        self.velocity_error_scale = float(cfg["env"].get("velocity_error_scale", 2.0))
        self.collision_reward_warmup_steps = int(cfg["env"].get("collision_reward_warmup_steps", 16000))
        self.edge_reward_warmup_steps = int(cfg["env"].get("edge_reward_warmup_steps", 16000))
        self.safe_reward_warmup_steps = int(cfg["env"].get("safe_reward_warmup_steps", 16000))
        self.collision_unsafe_distance = float(cfg["env"].get("collision_unsafe_distance", 0.20))
        self.collision_cone_deg = float(cfg["env"].get("collision_cone_deg", 30.0))
        self.collision_height_threshold = float(cfg["env"].get("collision_height_threshold", 0.03))
        self.collision_slope_threshold = float(cfg["env"].get("collision_slope_threshold", 0.5))
        self.collision_height_margin = float(cfg["env"].get("collision_height_margin", 0.04))
        self.collision_height_softness = float(cfg["env"].get("collision_height_softness", 0.015))
        self.collision_speed_scale = float(cfg["env"].get("collision_speed_scale", 1.0))
        self.collision_speed_floor = float(cfg["env"].get("collision_speed_floor", 0.25))
        self.foot_clearance_low = float(cfg["env"].get("foot_clearance_low", 0.02))
        self.foot_clearance_high = float(cfg["env"].get("foot_clearance_high", 0.06))
        self.leg_clearance_low = float(cfg["env"].get("leg_clearance_low", 0.03))
        self.leg_clearance_high = float(cfg["env"].get("leg_clearance_high", 0.10))
        self.crossing_height_threshold = float(cfg["env"].get("crossing_height_threshold", 0.025))
        self.foot_contact_force_threshold = float(cfg["env"].get("foot_contact_force_threshold", 10.0))
        self.impact_force_free = float(cfg["env"].get("impact_force_free", 20.0))
        self.impact_force_saturation = float(cfg["env"].get("impact_force_saturation", 150.0))
        self.knee_termination_force = float(cfg["env"].get("knee_termination_force", 150.0))
        self.knee_contact_frames = int(cfg["env"].get("knee_contact_frames", 3))
        self.critical_contact_force = float(cfg["env"].get("critical_contact_force", 50.0))
        self.fall_tilt_deg = float(cfg["env"].get("fall_tilt_deg", 45.0))
        self.fall_torso_height = float(cfg["env"].get("fall_torso_height", 0.45))
        self.edge_gradient_threshold = float(cfg["env"].get("edge_gradient_threshold", 0.04))
        self.edge_activation_height = float(cfg["env"].get("edge_activation_height", 0.12))
        self.edge_distance_scale = float(cfg["env"].get("edge_distance_scale", 0.12))
        self._collision_reward_step = 0
        self._edge_reward_step = 0
        self._safe_reward_step = 0


        self.square_height_points = self.init_square_height_points()
        self.terrain_obs_type = self.cfg['env'].get("terrain_obs_type",
                                                    "square")
        self.terrain_obs = self.cfg['env'].get("terrain_obs", False)
        self.terrain_obs_root = self.cfg['env'].get("terrain_obs_root",
                                                    "pelvis")
        if self.terrain_obs_type == "fov":
            self.height_points = self.init_fov_height_points()
        elif self.terrain_obs_type == "square_fov":
            self.height_points = self.init_square_fov_height_points()
        elif self.terrain_obs_type == "square":
            self.height_points = self.square_height_points
        self.root_points = self.init_root_points()

        self.center_height_points = self.init_center_height_points()
        self.height_meas_scale = 5

        self.show_sensors = self.cfg['args'].show_sensors
        if (not self.headless) and self.show_sensors:
            self._sensor_handles = [[] for _ in range(self.num_envs)]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.reward_raw = torch.zeros((self.num_envs, 7)).to(self.device)
        self._build_safe_stepping_tensors()

        # Reward decomposition monitor (--reward_debug): per-term batch means are
        # accumulated on the GPU and dumped to CSV after N control steps.
        self.reward_debug_names = [
            "location", "power", "velocity_raw", "velocity_weighted",
            "swing_raw", "landing_raw", "impact_raw", "safety_clipped",
            "edge", "task_total",
        ]
        self.reward_debug_on = bool(getattr(flags, "reward_debug", False))
        self._rwdbg_sum = None
        self._rwdbg_sq = None
        self._rwdbg_n = 0
        self._rwdbg_done = False
        if self.reward_debug_on:
            print("[reward_debug] enabled, "
                  f"{int(getattr(flags, 'reward_debug_steps', 600))} steps")

        if (not self.headless) and self.show_sensors:
            self._build_sensor_state_tensors()

        return

    def _build_safe_stepping_tensors(self):
        self._collision_body_ids = self._build_key_body_ids_tensor(
            ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"])
        self._edge_foot_ids = self._build_key_body_ids_tensor(
            ["L_Ankle", "R_Ankle"])
        self._knee_ids = self._build_key_body_ids_tensor(
            ["L_Knee", "R_Knee"])
        self._critical_body_ids = self._build_key_body_ids_tensor(
            ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"])
        self._pelvis_id = int(self._build_key_body_ids_tensor(["Pelvis"])[0].item())
        self._chest_id = int(self._build_key_body_ids_tensor(["Chest"])[0].item())
        self._collision_bottom_offsets = torch.tensor(
            [-0.0487, 0.0070, -0.0487, 0.0070],
            dtype=torch.float32, device=self.device)
        self._edge_center_offsets = torch.tensor(
            [[0.0596, 0.0132, -0.0279],
             [0.0617, -0.0127, -0.0241]],
            dtype=torch.float32, device=self.device)

        grid_x = torch.linspace(-0.10, 0.10, 5, device=self.device)
        grid_y = torch.linspace(-0.06, 0.06, 5, device=self.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
        self._edge_grid_size = 5
        self._edge_local_grid = torch.stack(
            [grid_x.flatten(), grid_y.flatten(),
             torch.zeros_like(grid_x).flatten()], dim=-1)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]], device=self.device) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self._edge_sobel_x = sobel_x.view(1, 1, 3, 3)
        self._edge_sobel_y = sobel_y.view(1, 1, 3, 3)
        self._risk_angles = torch.linspace(
            -self.collision_cone_deg, self.collision_cone_deg, 5,
            device=self.device) * (np.pi / 180.0)
        self._risk_distances = torch.linspace(
            self.collision_unsafe_distance / 5,
            self.collision_unsafe_distance, 5, device=self.device)
        self._leg_fractions = torch.tensor(
            [0.25, 0.50, 0.75], device=self.device).view(1, 1, 3, 1)
        self._leg_bottom_offsets = torch.zeros(6, device=self.device)

        self._previous_foot_contact = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device)
        self._previous_foot_ground = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._swing_crossed_edge = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device)
        self._swing_min_foot_clearance = torch.full(
            (self.num_envs, 2), float("inf"), device=self.device)
        self._swing_min_leg_clearance = torch.full(
            (self.num_envs, 2), float("inf"), device=self.device)
        self._knee_contact_count = torch.zeros(
            (self.num_envs, 2), dtype=torch.long, device=self.device)
        return

    def _sample_static_terrain(self, points):
        """Sample immutable terrain without cloning the complete height map."""
        batch_size, num_points, _ = points.shape
        if self.cfg["env"]["terrain"]["terrainType"] == "plane":
            return torch.zeros((batch_size, num_points), device=points.device,
                               dtype=points.dtype)
        # MeshTerrain.world_points_to_map historically modifies its input.
        # Never let a height query mutate a view into Isaac Gym state tensors.
        px, py = self.terrain.world_points_to_map(points.clone())
        heights1 = self.terrain.heightsamples[px, py]
        heights2 = self.terrain.heightsamples[px + 1, py + 1]
        heights = torch.minimum(heights1, heights2).to(points.dtype)
        return (heights * self.terrain.vertical_scale).view(batch_size, num_points)

    @staticmethod
    def _warmup_scale(step, warmup_steps):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))

    @staticmethod
    def _clearance_score(clearance, low, high):
        return torch.clamp((clearance - low) / max(high - low, 1e-4),
                           min=0.0, max=1.0)

    def _foot_contact_state(self):
        force = torch.linalg.norm(
            self._contact_forces[:, self._collision_body_ids, :], dim=-1)
        return force.view(self.num_envs, 2, 2).max(dim=-1).values > \
            self.foot_contact_force_threshold

    def _compute_riser_risk(self, point_pos, point_vel_xy, bottom_offsets,
                            active_mask, command_direction):
        """Dense forward-sweep risk for feet or sampled lower-leg points."""
        batch_size, num_points, _ = point_pos.shape
        speed = torch.linalg.norm(point_vel_xy, dim=-1)
        moving_direction = point_vel_xy / torch.clamp(
            speed.unsqueeze(-1), min=1e-6)
        fallback = command_direction.unsqueeze(1).expand(-1, num_points, -1)
        direction = torch.where((speed > 0.05).unsqueeze(-1),
                                moving_direction, fallback)

        num_rays = self._risk_angles.numel()
        num_distances = self._risk_distances.numel()
        angles = self._risk_angles
        cos_a = torch.cos(angles).view(1, 1, num_rays)
        sin_a = torch.sin(angles).view(1, 1, num_rays)
        dir_x = direction[..., 0].unsqueeze(-1)
        dir_y = direction[..., 1].unsqueeze(-1)
        ray_direction = torch.stack(
            [dir_x * cos_a - dir_y * sin_a,
             dir_x * sin_a + dir_y * cos_a], dim=-1)
        distances = self._risk_distances

        sample_xy = (point_pos[..., :2].unsqueeze(2).unsqueeze(3)
                     + ray_direction.unsqueeze(3)
                     * distances.view(1, 1, 1, num_distances, 1))
        sample_points = torch.zeros(
            (batch_size, num_points, num_rays, num_distances, 3),
            device=self.device, dtype=point_pos.dtype)
        sample_points[..., :2] = sample_xy
        sampled_height = self._sample_static_terrain(
            sample_points.view(batch_size, -1, 3)).view(
                batch_size, num_points, num_rays, num_distances)

        base_height = self._sample_static_terrain(point_pos).view(
            batch_size, num_points, 1, 1)
        previous_height = torch.cat(
            [base_height.expand(-1, -1, num_rays, -1),
             sampled_height[..., :-1]], dim=-1)
        distance_step = self.collision_unsafe_distance / num_distances
        local_slope = (sampled_height - previous_height) / distance_step
        candidate = torch.logical_and(
            sampled_height > base_height + self.collision_height_threshold,
            local_slope > self.collision_slope_threshold)

        candidate_distance = distances.view(1, 1, 1, num_distances).expand(
            batch_size, num_points, num_rays, num_distances)
        masked_distance = torch.where(
            candidate, candidate_distance,
            torch.full_like(candidate_distance, float("inf")))
        nearest_distance, nearest_idx = torch.min(
            masked_distance.view(batch_size, num_points, -1), dim=-1)
        valid = torch.isfinite(nearest_distance)
        flat_height = sampled_height.view(batch_size, num_points, -1)
        nearest_top = torch.gather(
            flat_height, -1, nearest_idx.unsqueeze(-1)).squeeze(-1)
        ray_idx = nearest_idx // num_distances
        nearest_direction = torch.gather(
            ray_direction, 2,
            ray_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        ).squeeze(2)

        approach_speed = torch.clamp(
            torch.sum(point_vel_xy * nearest_direction, dim=-1), min=0.0)
        normalized_speed = torch.clamp(
            approach_speed / max(self.collision_speed_scale, 1e-4),
            min=0.0, max=1.0)
        effective_speed = (self.collision_speed_floor
                           + (1.0 - self.collision_speed_floor)
                           * normalized_speed)
        distance_weight = torch.clamp(
            1.0 - nearest_distance / self.collision_unsafe_distance,
            min=0.0, max=1.0)
        point_bottom = point_pos[..., 2] + bottom_offsets.view(1, -1)
        height_gate = torch.sigmoid(
            (nearest_top + self.collision_height_margin - point_bottom)
            / max(self.collision_height_softness, 1e-4))
        risk = effective_speed * distance_weight * height_gate
        risk = torch.where(valid & active_mask, risk, torch.zeros_like(risk))
        return risk

    def _compute_landing_quality(self, touchdown):
        foot_pos = self._rigid_body_pos[:, self._edge_foot_ids, :]
        foot_rot = self._rigid_body_rot[:, self._edge_foot_ids, :]
        batch_size, num_feet, _ = foot_pos.shape
        num_grid = self._edge_local_grid.shape[0]
        center_offsets = self._edge_center_offsets.view(1, num_feet, 3).expand(
            batch_size, -1, -1)
        foot_center = foot_pos + quat_apply(
            foot_rot.reshape(-1, 4), center_offsets.reshape(-1, 3)
        ).view(batch_size, num_feet, 3)
        local_grid = self._edge_local_grid.view(1, 1, num_grid, 3).expand(
            batch_size, num_feet, -1, -1)
        grid_rot = foot_rot.unsqueeze(2).expand(
            -1, -1, num_grid, -1).reshape(-1, 4)
        world_grid = foot_center.unsqueeze(2) + quat_apply(
            grid_rot, local_grid.reshape(-1, 3)
        ).view(batch_size, num_feet, num_grid, 3)
        terrain_height = self._sample_static_terrain(
            world_grid.view(batch_size, -1, 3)).view(
                batch_size * num_feet, 1,
                self._edge_grid_size, self._edge_grid_size)

        center_ground = self._sample_static_terrain(foot_center).reshape(
            batch_size * num_feet, 1)
        support = (torch.abs(terrain_height.flatten(2)
                             - center_ground.unsqueeze(-1)) < 0.025
                   ).to(terrain_height.dtype).mean(dim=-1)
        padded = F.pad(terrain_height, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self._edge_sobel_x)
        grad_y = F.conv2d(padded, self._edge_sobel_y)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y).flatten(2) \
            > self.edge_gradient_threshold
        local_dist = torch.linalg.norm(self._edge_local_grid[:, :2], dim=-1)
        distances = local_dist.view(1, 1, -1).expand(
            batch_size, num_feet, -1).reshape(batch_size * num_feet, 1, -1)
        edge_distance = torch.where(
            edge, distances, torch.full_like(distances, float("inf"))
        ).min(dim=-1).values
        no_edge = ~torch.isfinite(edge_distance)
        margin = torch.clamp(edge_distance / 0.08, min=0.0, max=1.0)
        margin = torch.where(no_edge, torch.ones_like(margin), margin)
        downward_speed = torch.clamp(
            -self._rigid_body_vel[:, self._edge_foot_ids, 2], min=0.0)
        impact_quality = torch.exp(-2.0 * downward_speed * downward_speed)
        quality = (0.45 * support.view(batch_size, num_feet)
                   + 0.35 * margin.view(batch_size, num_feet)
                   + 0.20 * impact_quality)
        event_count = touchdown.to(quality.dtype).sum(dim=-1).clamp_min(1.0)
        return (quality * touchdown.to(quality.dtype)).sum(dim=-1) / event_count

    def _compute_safe_stepping_terms(self, root_pos, tar_pos):
        foot_contact = self._foot_contact_state()
        just_reset = self.progress_buf <= 1
        reset_mask = just_reset.unsqueeze(-1)
        self._previous_foot_contact = torch.where(
            reset_mask, foot_contact, self._previous_foot_contact)
        self._swing_crossed_edge = torch.where(
            reset_mask, torch.zeros_like(self._swing_crossed_edge),
            self._swing_crossed_edge)
        self._swing_min_foot_clearance = torch.where(
            reset_mask, torch.full_like(
                self._swing_min_foot_clearance, float("inf")),
            self._swing_min_foot_clearance)
        self._swing_min_leg_clearance = torch.where(
            reset_mask, torch.full_like(
                self._swing_min_leg_clearance, float("inf")),
            self._swing_min_leg_clearance)
        self._knee_contact_count = torch.where(
            reset_mask, torch.zeros_like(self._knee_contact_count),
            self._knee_contact_count)
        swing = ~foot_contact
        touchdown = foot_contact & ~self._previous_foot_contact & ~reset_mask

        ankle_pos = self._rigid_body_pos[:, self._edge_foot_ids, :]
        ankle_ground = self._sample_static_terrain(ankle_pos)
        ground_change = torch.abs(ankle_ground - self._previous_foot_ground)
        crossed_now = (swing & (ground_change > self.crossing_height_threshold)
                       & ~reset_mask)
        self._swing_crossed_edge |= crossed_now

        foot_clearance = ankle_pos[..., 2] - ankle_ground - 0.05
        knee_pos = self._rigid_body_pos[:, self._knee_ids, :]
        fractions = self._leg_fractions.to(dtype=ankle_pos.dtype)
        leg_points = knee_pos.unsqueeze(2) + fractions * (
            ankle_pos - knee_pos).unsqueeze(2)
        leg_ground = self._sample_static_terrain(
            leg_points.view(self.num_envs, -1, 3)).view(self.num_envs, 2, 3)
        leg_clearance = (leg_points[..., 2] - leg_ground).min(dim=-1).values
        track = swing & self._swing_crossed_edge
        self._swing_min_foot_clearance = torch.where(
            track, torch.minimum(self._swing_min_foot_clearance, foot_clearance),
            self._swing_min_foot_clearance)
        self._swing_min_leg_clearance = torch.where(
            track, torch.minimum(self._swing_min_leg_clearance, leg_clearance),
            self._swing_min_leg_clearance)

        valid_touchdown = touchdown & self._swing_crossed_edge
        foot_clear_score = self._clearance_score(
            self._swing_min_foot_clearance,
            self.foot_clearance_low, self.foot_clearance_high)
        leg_clear_score = self._clearance_score(
            self._swing_min_leg_clearance,
            self.leg_clearance_low, self.leg_clearance_high)
        foot_clear_event = torch.where(
            valid_touchdown, foot_clear_score, torch.zeros_like(foot_clear_score))
        leg_clear_event = torch.where(
            valid_touchdown, leg_clear_score, torch.zeros_like(leg_clear_score))

        command = tar_pos[..., :2] - root_pos[..., :2]
        command = command / torch.clamp(
            torch.linalg.norm(command, dim=-1, keepdim=True), min=1e-6)
        foot_pos = self._rigid_body_pos[:, self._collision_body_ids, :]
        foot_vel = self._rigid_body_vel[:, self._collision_body_ids, :2]
        foot_active = swing.unsqueeze(-1).expand(-1, -1, 2).reshape(
            self.num_envs, 4)
        foot_risk = self._compute_riser_risk(
            foot_pos, foot_vel, self._collision_bottom_offsets,
            foot_active, command).view(self.num_envs, 2, 2).max(dim=-1).values

        knee_vel = self._rigid_body_vel[:, self._knee_ids, :2]
        ankle_vel = self._rigid_body_vel[:, self._edge_foot_ids, :2]
        leg_vel = knee_vel.unsqueeze(2) + fractions[..., :1] * (
            ankle_vel - knee_vel).unsqueeze(2)
        leg_active = swing.unsqueeze(-1).expand(-1, -1, 3).reshape(
            self.num_envs, 6)
        leg_risk = self._compute_riser_risk(
            leg_points.reshape(self.num_envs, 6, 3),
            leg_vel.reshape(self.num_envs, 6, 2),
            self._leg_bottom_offsets.to(dtype=leg_points.dtype),
            leg_active, command).view(self.num_envs, 2, 3).max(dim=-1).values

        positive_event = (0.60 * foot_clear_event + 0.40 * leg_clear_event
                          ).max(dim=-1).values
        dense_risk = (0.60 * foot_risk + 0.40 * leg_risk).max(dim=-1).values
        swing_reward = torch.clamp(
            positive_event - dense_risk, min=-1.0, max=1.0)
        landing_reward = self._compute_landing_quality(touchdown)

        foot_force = torch.linalg.norm(
            self._contact_forces[:, self._collision_body_ids, :], dim=-1
        ).view(self.num_envs, 2, 2).max(dim=-1).values
        knee_force = torch.linalg.norm(
            self._contact_forces[:, self._knee_ids, :], dim=-1)
        denom = max(self.impact_force_saturation - self.impact_force_free, 1.0)
        foot_impact = torch.clamp(
            (foot_force - self.impact_force_free) / denom, 0.0, 1.0)
        # Foot force is an impact only at contact onset; otherwise normal body
        # weight during stance would be punished on every control step.
        foot_impact = foot_impact * touchdown.to(foot_impact.dtype)
        knee_impact = torch.clamp(
            (knee_force - self.impact_force_free) / denom, 0.0, 1.0)
        impact_reward = -(0.35 * foot_impact.max(dim=-1).values
                          + 0.65 * knee_impact.max(dim=-1).values)

        knee_touch = knee_force > self.impact_force_free
        self._knee_contact_count = torch.where(
            knee_touch, self._knee_contact_count + 1,
            torch.zeros_like(self._knee_contact_count))

        ended = touchdown
        self._swing_crossed_edge = torch.where(
            ended, torch.zeros_like(self._swing_crossed_edge),
            self._swing_crossed_edge)
        self._swing_min_foot_clearance = torch.where(
            ended, torch.full_like(self._swing_min_foot_clearance, float("inf")),
            self._swing_min_foot_clearance)
        self._swing_min_leg_clearance = torch.where(
            ended, torch.full_like(self._swing_min_leg_clearance, float("inf")),
            self._swing_min_leg_clearance)
        self._previous_foot_contact.copy_(foot_contact)
        self._previous_foot_ground.copy_(ankle_ground)
        return swing_reward, landing_reward, impact_reward

    def _compute_foot_collision_penalty(self):
        body_pos = self._rigid_body_pos[:, self._collision_body_ids, :]
        body_vel_xy = self._rigid_body_vel[:, self._collision_body_ids, :2]
        batch_size, num_feet, _ = body_pos.shape

        speed = torch.linalg.norm(body_vel_xy, dim=-1)
        direction = body_vel_xy / torch.clamp(speed.unsqueeze(-1), min=1e-6)
        num_rays, num_distances = 5, 5
        angles = torch.linspace(-self.collision_cone_deg,
                                self.collision_cone_deg, num_rays,
                                device=self.device) * (np.pi / 180.0)
        cos_a = torch.cos(angles).view(1, 1, num_rays)
        sin_a = torch.sin(angles).view(1, 1, num_rays)
        dir_x, dir_y = direction[..., 0].unsqueeze(-1), direction[..., 1].unsqueeze(-1)
        ray_direction = torch.stack(
            [dir_x * cos_a - dir_y * sin_a,
             dir_x * sin_a + dir_y * cos_a], dim=-1)
        distances = torch.linspace(
            self.collision_unsafe_distance / num_distances,
            self.collision_unsafe_distance, num_distances, device=self.device)

        sample_xy = (body_pos[..., :2].unsqueeze(2).unsqueeze(3)
                     + ray_direction.unsqueeze(3)
                     * distances.view(1, 1, 1, num_distances, 1))
        sample_points = torch.zeros(
            (batch_size, num_feet, num_rays, num_distances, 3),
            device=self.device, dtype=body_pos.dtype)
        sample_points[..., :2] = sample_xy
        sampled_height = self._sample_static_terrain(
            sample_points.view(batch_size, -1, 3)).view(
                batch_size, num_feet, num_rays, num_distances)

        base_height = self._sample_static_terrain(body_pos).view(
            batch_size, num_feet, 1, 1)
        previous_height = torch.cat(
            [base_height.expand(-1, -1, num_rays, -1), sampled_height[..., :-1]],
            dim=-1)
        distance_step = self.collision_unsafe_distance / num_distances
        local_slope = (sampled_height - previous_height) / distance_step
        candidate = torch.logical_and(
            sampled_height > base_height + self.collision_height_threshold,
            local_slope > self.collision_slope_threshold)

        candidate_distance = distances.view(1, 1, 1, num_distances).expand(
            batch_size, num_feet, num_rays, num_distances)
        masked_distance = torch.where(
            candidate, candidate_distance,
            torch.full_like(candidate_distance, float("inf")))
        nearest_distance, nearest_idx = torch.min(
            masked_distance.view(batch_size, num_feet, -1), dim=-1)
        valid = torch.isfinite(nearest_distance)

        flat_height = sampled_height.view(batch_size, num_feet, -1)
        nearest_top = torch.gather(
            flat_height, -1, nearest_idx.unsqueeze(-1)).squeeze(-1)
        ray_idx = nearest_idx // num_distances
        nearest_direction = torch.gather(
            ray_direction, 2,
            ray_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        ).squeeze(2)

        approach_speed = torch.clamp(
            torch.sum(body_vel_xy * nearest_direction, dim=-1), min=0.0)
        distance_weight = torch.clamp(
            1.0 - nearest_distance / self.collision_unsafe_distance,
            min=0.0, max=1.0)
        foot_bottom = body_pos[..., 2] + self._collision_bottom_offsets
        height_gate = torch.sigmoid(
            (nearest_top + self.collision_height_margin - foot_bottom)
            / max(self.collision_height_softness, 1e-4))
        normalized_speed = torch.clamp(
            approach_speed / max(self.collision_speed_scale, 1e-4),
            min=0.0, max=1.0)
        risk = normalized_speed * distance_weight * height_gate
        risk = torch.where(valid, risk, torch.zeros_like(risk))
        return -torch.max(risk, dim=-1).values

    def _compute_edge_stepping_penalty(self, root_pos, tar_pos):
        foot_pos = self._rigid_body_pos[:, self._edge_foot_ids, :]
        foot_rot = self._rigid_body_rot[:, self._edge_foot_ids, :]
        batch_size, num_feet, _ = foot_pos.shape
        num_grid = self._edge_local_grid.shape[0]

        center_offsets = self._edge_center_offsets.view(1, num_feet, 3).expand(
            batch_size, -1, -1)
        foot_center = foot_pos + quat_apply(
            foot_rot.reshape(-1, 4), center_offsets.reshape(-1, 3)
        ).view(batch_size, num_feet, 3)
        local_grid = self._edge_local_grid.view(1, 1, num_grid, 3).expand(
            batch_size, num_feet, -1, -1)
        grid_rot = foot_rot.unsqueeze(2).expand(-1, -1, num_grid, -1).reshape(-1, 4)
        world_grid = foot_center.unsqueeze(2) + quat_apply(
            grid_rot, local_grid.reshape(-1, 3)
        ).view(batch_size, num_feet, num_grid, 3)
        terrain_height = self._sample_static_terrain(
            world_grid.view(batch_size, -1, 3)).view(
                batch_size * num_feet, 1,
                self._edge_grid_size, self._edge_grid_size)

        padded_height = F.pad(terrain_height, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded_height, self._edge_sobel_x)
        grad_y = F.conv2d(padded_height, self._edge_sobel_y)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
        edge_weight = (grad_mag > self.edge_gradient_threshold).to(
            terrain_height.dtype).view(batch_size, num_feet, num_grid)
        edge_count = torch.sum(edge_weight, dim=-1)
        has_edge = edge_count > 0
        edge_centroid = torch.sum(
            world_grid[..., :2] * edge_weight.unsqueeze(-1), dim=2
        ) / torch.clamp(edge_count.unsqueeze(-1), min=1.0)
        edge_vector = edge_centroid - foot_center[..., :2]

        grad_x_mean = torch.sum(
            grad_x.view(batch_size, num_feet, num_grid) * edge_weight,
            dim=-1) / torch.clamp(edge_count, min=1.0)
        grad_y_mean = torch.sum(
            grad_y.view(batch_size, num_feet, num_grid) * edge_weight,
            dim=-1) / torch.clamp(edge_count, min=1.0)
        local_gradient = torch.stack(
            [grad_x_mean, grad_y_mean, torch.zeros_like(grad_x_mean)], dim=-1)
        world_gradient = quat_apply(
            foot_rot.reshape(-1, 4), local_gradient.reshape(-1, 3)
        ).view(batch_size, num_feet, 3)[..., :2]

        command = tar_pos[..., :2] - root_pos[..., :2]
        command = command / torch.clamp(
            torch.linalg.norm(command, dim=-1, keepdim=True), min=1e-6)
        command = command.unsqueeze(1).expand(-1, num_feet, -1)
        sign_correction = -torch.sign(torch.sum(world_gradient * command, dim=-1))
        edge_projection = torch.sum(edge_vector * command, dim=-1)
        raw_edge = sign_correction * torch.minimum(
            edge_projection, torch.zeros_like(edge_projection))
        raw_edge = torch.minimum(raw_edge, torch.zeros_like(raw_edge))

        center_ground = self._sample_static_terrain(foot_center).view(
            batch_size, num_feet)
        foot_clearance = torch.clamp(foot_center[..., 2] - center_ground, min=0.0)
        height_weight = torch.clamp(
            1.0 - foot_clearance / max(self.edge_activation_height, 1e-4),
            min=0.0, max=1.0)
        edge_penalty = torch.clamp(
            raw_edge / max(self.edge_distance_scale, 1e-4),
            min=-1.0, max=0.0) * height_weight
        edge_penalty = torch.where(has_edge, edge_penalty,
                                   torch.zeros_like(edge_penalty))
        return torch.min(edge_penalty, dim=-1).values

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless) and self.show_sensors:
            self._load_sensor_asset()
            self._build_sensor(env_id, env_ptr)

        return

    def _build_sensor(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self.num_height_points):
            marker_handle = self.gym.create_actor(env_ptr, self._sensor_asset,
                                                  default_pose, "marker",
                                                  self.num_envs + 1, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(*agt_color(env_id)))
            self._sensor_handles[env_id].append(marker_handle)

        return

    def _build_sensor_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._sensor_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 11:(11 + self.num_height_points), :]
        self._sensor_pos = self._sensor_states[..., :3]
        self._sensor_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._sensor_handles, dtype=torch.int32, device=self.device)
        self._sensor_actor_ids = self._sensor_actor_ids.flatten()
        return

    def _load_sensor_asset(self):
        asset_root = "pacer/data/assets/mjcf/"
        asset_file = "sensor_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._sensor_asset = self.gym.load_asset(self.sim, asset_root,
                                                 asset_file, asset_options)

        return

    def _draw_task(self):
        # cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        norm_states = self.get_head_pose()
        base_quat = norm_states[:, 3:7]
        if not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)
        heading_rot = torch_utils.calc_heading_quat(base_quat)

        points = quat_apply(
            heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
            self.height_points) + (norm_states[:, :3]).unsqueeze(1)

        if (not self.headless) and self.show_sensors:
            self._sensor_pos[:] = points
        # self._sensor_pos[..., 2] += 0.3
        # self._sensor_pos[..., 2] -= 5

        traj_samples = self._fetch_traj_samples()

        self._marker_pos[:] = traj_samples
        self._marker_pos[..., 2] = self._humanoid_root_states[..., 2:3]  # jp hack # ZL hack
        # self._marker_pos[..., 2] = 0.89
        # self._marker_pos[..., 2] = 0

        if (not self.headless) and self.show_sensors:
            comb_idx = torch.cat([self._sensor_actor_ids, self._marker_actor_ids])
        else:
            comb_idx = torch.cat([self._marker_actor_ids])

        if flags.show_traj:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(comb_idx), len(comb_idx))

            self.gym.clear_lines(self.viewer)

            for i, env_ptr in enumerate(self.envs):
                verts = self._traj_gen.get_traj_verts(i)
                verts[..., 2] = self._humanoid_root_states[i, 2]  # ZL Hack
                # verts[..., 2] = 0.89
                # verts[..., 2] = 0
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                # cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                cols = np.array(agt_color(i), dtype=np.float32)[None, ]
                curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
        else:
            self._marker_pos[:] = 0
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(comb_idx), len(comb_idx))

            self.gym.clear_lines(self.viewer)


        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (ENABLE_MAX_COORD_OBS):
            if (env_ids is None):
                body_pos = self._rigid_body_pos.clone()
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
            if self.smpl_humanoid:
                if (env_ids is None):
                    smpl_params = self.humanoid_betas
                    limb_weights = self.humanoid_limb_and_weights
                else:
                    smpl_params = self.humanoid_betas[env_ids]
                    limb_weights = self.humanoid_limb_and_weights[env_ids]

                if self._root_height_obs:
                    center_height = self.get_center_heights(torch.cat([body_pos[:, 0], body_rot[:, 0]], dim=-1), env_ids=env_ids).mean(dim=-1, keepdim=True)
                    body_pos[:, :, 2] = body_pos[:, :, 2] - center_height

                obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)
            else:
                obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs)

        else:
            if (env_ids is None):
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
                dof_pos = self._dof_pos
                dof_vel = self._dof_vel
                key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            else:
                root_pos = self._rigid_body_pos[env_ids][:, 0, :]
                root_rot = self._rigid_body_rot[env_ids][:, 0, :]
                root_vel = self._rigid_body_vel[env_ids][:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[env_ids][:, 0, :]
                dof_pos = self._dof_pos[env_ids]
                dof_vel = self._dof_vel[env_ids]
                key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

            if self.smpl_humanoid and self.self.has_shape_obs:
                if (env_ids is None):
                    smpl_params = self.humanoid_betas
                else:
                    smpl_params = self.humanoid_betas[env_ids]
                obs = compute_humanoid_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._dof_obs_size, self._dof_offsets, smpl_params, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs)
            else:
                obs = compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets)
        return obs

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):

            obs_size = 2 * self._num_traj_samples

            if self.terrain_obs:
                if self.velocity_map:
                    obs_size += self.num_height_points * 3
                else:
                    obs_size += self.num_height_points

            if self._divide_group and self._group_obs:
                obs_size += 5 * 11 * 3

        return obs_size

    def get_self_obs_size(self):
        return self._num_self_obs

    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()
        
        
        if (self._enable_task_obs):
            task_obs_detail['traj'] = 2 * self._num_traj_samples

        if self.terrain_obs:
            if self.velocity_map:
                task_obs_detail['heightmap_velocity'] = self.num_height_points * 3
            else:
                task_obs_detail['heightmap'] = self.num_height_points

        if self._divide_group and self._group_obs:
            task_obs_detail['people'] = 5 * 11 * 3

        return task_obs_detail

    def get_head_pose(self, env_ids=None):
        if self.smpl_humanoid:
            head_idx = self._body_names.index("Head")
        else:
            head_idx = 2
        head_pose = torch.cat([
            self._rigid_body_pos[:, head_idx], self._rigid_body_rot[:,
                                                                    head_idx]
        ],
                              dim=1)
        if (env_ids is None):
            return head_pose
        else:
            return head_pose[env_ids]

    def update_value_func(self, eval_value_func, actor_func):
        self.eval_value_func = eval_value_func
        self.actor_func = actor_func

    def query_value_gradient(self, env_ids, new_traj):
        # new_traj would be the same as self._fetch_traj_samples(env_ids)
        # return callable value function and update_obs (processed with mean and std)
        # value_func(obs)
        # new_traj of shape (num_envs, 10, 3)
        # TODO: implement this
        if "eval_value_func" in self.__dict__:
            sim_obs_size = self.get_self_obs_size()
            task_obs_detal = self.get_task_obs_size_detail()
            assert(task_obs_detal[0][0] == "traj")

            if (env_ids is None):
                root_states = self._humanoid_root_states
            else:
                root_states = self._humanoid_root_states[env_ids]

            new_traj_obs = compute_location_observations(root_states, new_traj.view(env_ids.shape[0], 10, -1), self._has_upright_start)
            buffered_obs = self.obs_buf[env_ids].clone()
            buffered_obs[:, sim_obs_size:(task_obs_detal[0][1] + sim_obs_size)] = new_traj_obs

            return buffered_obs, self.eval_value_func
        return None, None

    def live_plotter(self, img,  identifier='', pause_time=0.00000001):
        if not hasattr(self, 'imshow_obj'):
            plt.ion()

            self.fig = plt.figure(figsize=(1, 1), dpi = 350)
            ax = self.fig.add_subplot(111)
            self.imshow_obj = ax.imshow(img)
            # create a variable for the line so we can later update it
            # update plot label/title

            plt.title('{}'.format(identifier))
            plt.show()
        if not img is None:
            self.imshow_obj.set_data(img)

        # plt.pause(pause_time)
        self.fig.canvas.start_event_loop(0.001)

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        traj_samples = self._fetch_traj_samples(env_ids)

        obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)
        if self.terrain_obs:

            if self.terrain_obs_root == "head":
                head_pose = self.get_head_pose(env_ids=env_ids)
                self.measured_heights = self.get_heights(root_states=head_pose, env_ids=env_ids)
            else:
                self.measured_heights = self.get_heights(root_states=root_states, env_ids=env_ids)


            if flags.height_debug:
                # joblib.dump(self.measured_heights, "heights.pkl")
                if env_ids is None or len(env_ids) == self.num_envs:
                    heights = self.measured_heights.view(num_envs, -1, 3 if self.velocity_map else 1)
                    sensor_size = int(np.sqrt(self.num_height_points))
                    heights_show = heights.cpu().numpy()[self.viewing_env_idx, :, 0].reshape(sensor_size, sensor_size)
                    if heights_show.min() < 0:
                        heights_show -= heights_show.min()
                    self.live_plotter(heights_show)

            if self.cfg['env'].get("use_center_height", False):
                center_heights = self.get_center_heights(root_states=root_states, env_ids=env_ids)
                center_heights = center_heights.mean(dim=-1, keepdim=True)

                if self.velocity_map:
                    measured_heights = self.measured_heights.view(num_envs, -1, 3)
                    measured_heights[..., 0] = center_heights - measured_heights[..., 0]
                    heights = measured_heights.view(num_envs, -1)
                else:
                    heights = center_heights - self.measured_heights
                heights = torch.clip(heights, -3, 3.) * self.height_meas_scale  #

            else:
                heights = torch.clip(root_states[:, 2:3] - self.measured_heights, -3, 3.) * self.height_meas_scale  #

            obs = torch.cat([obs, heights], dim=1)

        if self._divide_group and self._group_obs:
            group_obs = compute_group_observation(self._rigid_body_pos, self._rigid_body_rot, self._rigid_body_vel, self.selected_group_jts, self._group_num_people, self._has_upright_start)
            # Group obs has to be computed as a whole. otherwise, the grouping breaks.
            if not (env_ids is None):
                group_obs = group_obs[env_ids]

            obs = torch.cat([obs, group_obs], dim=1)

        return obs


    def _compute_flip_task_obs(self, normal_task_obs, env_ids):

        # location_obs  20
        # Terrain obs: self.num_terrain_obs
        # group obs
        B, D = normal_task_obs.shape
        traj_samples_dim = 20
        obs_acc = []
        normal_task_obs = normal_task_obs.clone()
        traj_samples = normal_task_obs[:, :traj_samples_dim].view(B, 10, 2)
        traj_samples[..., 1] *= -1
        obs_acc.append(traj_samples.view(B, -1))
        if self.terrain_obs:
            if self.velocity_map:
                height_samples = normal_task_obs[..., traj_samples_dim: traj_samples_dim + self.num_height_points * 3]
                height_samples = height_samples.view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)), 3)
                height_samples[..., 0].flip(2)
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))
            else:
                height_samples = normal_task_obs[..., traj_samples_dim: traj_samples_dim + self.num_height_points].view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)))
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))

        obs = torch.cat(obs_acc, dim=1)

        if self._divide_group and self._group_obs:
            group_obs = normal_task_obs[..., traj_samples_dim + self.num_height_points: ].view(B, -1, 3)
            group_obs[..., 1] *= -1
            obs_acc.append(group_obs.view(B, -1))


        obs = torch.cat(obs_acc, dim=1)

        del obs_acc

        return obs

    def _terrain_gradient_heading(self, root_pos):
        """Heading along the local terrain gradient, for the trajectory start.

        The trajectory generator moves along [cos(theta), -sin(theta)] in world
        xy, so an uphill direction (gx, gy) corresponds to atan2(-gy, gx).
        """
        n = root_pos.shape[0]
        d = self.stair_heading_probe
        offs = torch.tensor([[d, 0.0, 0.0], [-d, 0.0, 0.0],
                             [0.0, d, 0.0], [0.0, -d, 0.0]],
                            device=root_pos.device, dtype=root_pos.dtype)
        pts = root_pos[:, None, :] + offs[None]
        h = self._sample_static_terrain(pts)
        gx = h[:, 0] - h[:, 1]
        gy = h[:, 2] - h[:, 3]
        if self.stair_heading_mode == "downhill":
            gx, gy = -gx, -gy
        # flat ground (or exactly on a plateau): keep walking along +x
        flat = (gx.abs() + gy.abs()) < 1e-4
        gx = torch.where(flat, torch.ones_like(gx), gx)
        gy = torch.where(flat, torch.zeros_like(gy), gy)
        return torch.atan2(-gy, gx)

    def _reset_task(self, env_ids):
        # super()._reset_task(env_ids) # Commented out to disable traj resetting
        if not flags.server_mode:
            root_pos = self._humanoid_root_states[env_ids, 0:3]
            headings = None
            turn_scale = self.stair_path_turn_scale
            if self.stair_heading_mode in ("uphill", "downhill"):
                headings = self._terrain_gradient_heading(root_pos)
            elif self.stair_heading_mode == "straight":
                # Fixed straight path along +x, matching the flat-flight corridor.
                headings = torch.zeros(root_pos.shape[0], device=root_pos.device,
                                       dtype=root_pos.dtype)
                turn_scale = 0.0
            self._traj_gen.reset(env_ids, root_pos, init_heading=headings,
                                 turn_scale=turn_scale)
        return


    def _sample_ref_state(self, env_ids, vel_min=1, vel_range=0.5):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidAMP.StateInit.Random
                or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (
                False
            ), "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init))

        if self.smpl_humanoid:
            curr_gender_betas = self.humanoid_betas[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_motionlib(
                motion_ids, motion_times, curr_gender_betas)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
                motion_ids, motion_times)
            rb_pos, rb_rot = None, None


        # TERRA: rotate the initial pose by a fixed yaw (relative).
        # NOTE: Isaac Gym stores quaternions as (x, y, z, w).
        # stair_heading_mode == "straight" pins body +X to +x so the robot faces
        # the straight corridor it is about to walk along. The motion clip has
        # its own heading, and this rotation is *relative*, so the correction has
        # to be computed per env: yaw_i = -heading_i(root_rot).
        straight_heading = getattr(self, "stair_heading_mode", "random") == "straight"
        if (flags.random_heading or getattr(flags, "start_yaw", None) is not None
                or straight_heading):
            start_yaw = getattr(flags, "start_yaw", None)
            if start_yaw is None and straight_heading:
                ex = torch_utils.my_quat_rotate(
                    root_rot,
                    torch.tensor([1.0, 0.0, 0.0], device=self.device,
                                 dtype=root_rot.dtype).repeat(root_rot.shape[0], 1))
                ang = torch.atan2(ex[:, 1], ex[:, 0])
                half = -ang / 2.0
                zero = torch.zeros(num_envs, device=self.device, dtype=root_rot.dtype)
                random_heading_quat = torch.stack(
                    [zero, zero, torch.sin(half), torch.cos(half)], dim=-1)
            elif start_yaw is not None:
                half = float(start_yaw) / 2.0
                zero = torch.zeros(num_envs, device=self.device, dtype=root_rot.dtype)
                cos_h = torch.full((num_envs,), float(np.cos(half)),
                                   device=self.device, dtype=root_rot.dtype)
                sin_h = torch.full((num_envs,), float(np.sin(half)),
                                   device=self.device, dtype=root_rot.dtype)
                random_heading_quat = torch.stack([zero, zero, sin_h, cos_h], dim=-1)
            else:
                random_rot = np.zeros([num_envs, 3])
                random_rot[:, 2] = np.pi * (2 * np.random.random([num_envs]) - 1.0)
                q_xyzw = sRot.from_euler("xyz", random_rot).as_quat()
                random_heading_quat = torch.from_numpy(
                    q_xyzw).float().to(self.device)
            random_heading_quat_repeat = random_heading_quat[:, None].repeat(1, 24, 1)
            root_rot = quat_mul(random_heading_quat, root_rot).clone()
            # 绕根位置旋转刚体，然后把根位置加回去（原代码漏了这一步）
            rb_pos = (quat_apply(random_heading_quat_repeat,
                                 rb_pos - root_pos[:, None, :])
                      + root_pos[:, None, :]).clone()
            key_pos = (quat_apply(random_heading_quat_repeat[:, :4, :],
                                  key_pos - root_pos[:, None, :])
                       + root_pos[:, None, :]).clone()
            rb_rot = quat_mul(random_heading_quat_repeat, rb_rot).clone()
            root_ang_vel = quat_apply(random_heading_quat, root_ang_vel).clone()
            if getattr(flags, "verbose_path", False):
                ex = torch_utils.my_quat_rotate(
                    root_rot, torch.tensor([[1.0, 0.0, 0.0]], device=root_rot.device).repeat(root_rot.shape[0], 1))
                import math as _m
                _ang = float(_m.degrees(_m.atan2(ex[0, 1].item(), ex[0, 0].item())))
                print(f"[start_yaw] requested={start_yaw} -> body+X heading = {_ang:.1f} deg")

            curr_heading = torch_utils.calc_heading_quat(root_rot)
            root_vel[:, 0] = torch.rand([num_envs]) * vel_range + vel_min
            root_vel = quat_apply(curr_heading, root_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = self._sample_ref_state(env_ids)
        ## Randomrized location setting
        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)
        # joblib.dump(self.terrain.sample_valid_locations(100000, torch.arange(100000)).detach().cpu(), "new_root_xy.pkl")
        # import ipdb; ipdb.set_trace()

        if flags.fixed:
            # new_root_xy[:, 0], new_root_xy[:, 1] = 0 , 0
            # new_root_xy[:, 0], new_root_xy[:, 1] = 134.8434 + env_ids , -28.9593
            # new_root_xy[:, 0], new_root_xy[:, 1] = 30 + env_ids * 4, 240
            new_root_xy[:, 0], new_root_xy[:, 1] = 10 + env_ids * 3, 10
            # new_root_xy[:, 0], new_root_xy[:, 1] = 95 + env_ids * 5, 307
            # new_root_xy[:, 0], new_root_xy[:, 1] = 27, 1 + env_ids * 2
            # x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
            # new_root_xy[:, 0], new_root_xy[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2
            # if env_ids[0] == 0:
            # new_root_xy[0, 0], new_root_xy[0, 1] = 34 , -81

        if flags.server_mode:
            new_traj = self._traj_gen.input_new_trajs(env_ids)
            new_root_xy[:, 0], new_root_xy[:, 1] = new_traj[:, 0, 0], new_traj[:, 0,  1]

        # TERRA: fixed spawn position (--start_xy "x,y"), used for "start at platform top".
        start_xy = getattr(flags, "start_xy", None)
        if start_xy is not None:
            new_root_xy[:, 0] = float(start_xy[0])
            new_root_xy[:, 1] = float(start_xy[1])



        diff_xy = new_root_xy - root_pos[:, 0:2]
        root_pos[:, 0:2] = new_root_xy

        root_states = torch.cat([root_pos, root_rot], dim=1)

        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)

        root_pos[:, 2] += center_height
        key_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[...,  2] += center_height[:, None]
        rb_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[..., 2] += center_height[:, None]

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        if flags.follow:
            self.start = True  ## Updating camera when reset

        return

    def init_center_height_points(self):
        # center_height_points
        y =  torch.tensor(np.linspace(-0.2, 0.2, 3),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.1, 0.1, 3),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_height_points(self):
        # 4mx4m square
        y =  torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent, self.sensor_res),device=self.device,requires_grad=False)
        x = torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent,
                                     self.sensor_res),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_fov_height_points(self):
        y = torch.tensor(np.linspace(-1, 1, 20),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.02, 1.98, 20),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                            device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_root_points(self):
        y = torch.tensor(np.linspace(-0.5, 0.5, 20),
                         device=self.device,
                         requires_grad=False)
        x = torch.tensor(np.linspace(-0.25, 0.25, 10),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_root_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_fov_height_points(self):
        # 3m x 3m fan shaped area
        rs =  np.exp(np.arange(0.2, 2, 0.1)) - 0.9
        rs = rs/rs.max() * 2

        max_angle = 110
        phi = np.exp(np.linspace(0.1, 1.5, 12)) - 1
        phi = phi/phi.max() * max_angle
        phi = np.concatenate([-phi[::-1],[0], phi]) * np.pi/180
        xs, ys = [], []
        for r in rs:
            xs.append(r * np.cos(phi)); ys.append(r * np.sin(phi))

        xs, ys = np.concatenate(xs), np.concatenate(ys)

        xs, ys = torch.from_numpy(xs).to(self.device), torch.from_numpy(ys).to(
            self.device)

        self.num_height_points = xs.shape[0]
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = xs
        points[:, :, 1] = ys
        return points

    def get_center_heights(self, root_states, env_ids=None):
        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_center_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.smpl_humanoid and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        if env_ids is None:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        heights = self.terrain.sample_height_points(points.clone(), env_ids=env_ids)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def get_heights(self, root_states, env_ids=None):

        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.smpl_humanoid and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        heading_rot = torch_utils.calc_heading_quat(base_quat)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        if self.velocity_map:
            root_states_all = self._humanoid_root_states
        else:
            root_states_all = None

        if (self._divide_group or flags.divide_group) and not self._group_obs and not self._disable_group_obs:
            heading_rot_all = torch_utils.calc_heading_quat(self._humanoid_root_states[:, 3:7])
            root_points = quat_apply(
                heading_rot_all.repeat(1, self.num_root_points).reshape(-1, 4),
                self.root_points) + (self._humanoid_root_states[:, :3]).unsqueeze(1)
            # update heights with root points
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states = root_states_all,
                root_points = root_points,
                env_ids=env_ids,
                num_group_people=self._group_num_people,
                group_ids = self._group_ids)
        else:
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states=root_states_all,
                root_points=None,
                env_ids=env_ids,
            )
        # heights = self.terrain.sample_height_points(points.clone(), None)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def _create_ground_plane(self):
        if self.real_mesh:
            self.create_mesh_ground()
        else:
            self.create_training_ground()

    def create_mesh_ground(self):
        # plane_params = gymapi.PlaneParams()
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.distance = 7.001
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        # self.gym.add_ground(self.sim, plane_params)

        # scene_name = "parking"
        # scene_name = "parking_with_cars"
        # scene_name = "with_less_car"
        # scene_name = "mesh-downtown-san-jose-mapaligned-cropped-bottom-part-global"
        scene_name = getattr(flags, "mesh_name", "") or "stairs20"
        self.mesh_data = joblib.load(f"data/mesh/{scene_name}.pkl")

        mesh_vertices = self.mesh_data["vertices"]
        mesh_triangles = self.mesh_data["faces"].astype(np.uint32)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = mesh_vertices.shape[0]
        tm_params.nb_triangles = mesh_triangles.shape[0]
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.plane_static_friction
        tm_params.dynamic_friction = self.plane_dynamic_friction
        tm_params.restitution = self.plane_restitution
        self.gym.add_triangle_mesh(self.sim, mesh_vertices.flatten(order='C'),
                                   mesh_triangles.flatten(order='C'),
                                   tm_params)

        self.terrain = MeshTerrain(self.mesh_data, self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).to(
            self.device)
        return

    def create_training_ground(self):
        if flags.small_terrain:
            self.cfg["env"]["terrain"]['mapLength'] = 8
            self.cfg["env"]["terrain"]['mapWidth'] = 8

        self.terrain = Terrain(self.cfg["env"]["terrain"],
                               num_robots=self.num_envs,
                               device=self.device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _compute_reset(self):
        env_ids = torch.arange(self.num_envs,
                               device=self.device,
                               dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, self._anchored_time(env_ids, 0))
        root_states = self._humanoid_root_states
        center_height = self.get_center_heights(
            root_states, env_ids=None).mean(dim=-1, keepdim=True)

        body_force = torch.linalg.norm(self._contact_forces, dim=-1)
        critical_contact = torch.any(
            body_force[:, self._critical_body_ids] > self.critical_contact_force,
            dim=-1)
        knee_force = body_force[:, self._knee_ids]
        knee_failure = torch.any(
            (knee_force > self.knee_termination_force)
            | (self._knee_contact_count >= self.knee_contact_frames), dim=-1)

        pelvis = self._rigid_body_pos[:, self._pelvis_id, :]
        chest = self._rigid_body_pos[:, self._chest_id, :]
        torso_axis = chest - pelvis
        upright_cos = torso_axis[:, 2] / torch.clamp(
            torch.linalg.norm(torso_axis, dim=-1), min=1e-6)
        tilt_limit = np.cos(np.deg2rad(self.fall_tilt_deg))
        torso_tilted = upright_cos < tilt_limit
        torso_low = (chest[:, 2] - center_height.squeeze(-1)) \
            < self.fall_torso_height
        safety_failure = critical_contact | knee_failure | torso_tilted | torso_low

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._rigid_body_pos, tar_pos,
            self.max_episode_length, self._fail_dist,
            self._enable_early_termination, safety_failure,
            flags.no_collision_check)
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, self._anchored_time(env_ids, 1))
        if self.fuzzy_target:
            location_reward = compute_location_reward_fuzzy(root_pos, tar_pos)
        else:
            location_reward = compute_location_reward(root_pos, tar_pos)

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self.power_coefficient * power
        new_reward_scale = self._warmup_scale(
            self._safe_reward_step, self.safe_reward_warmup_steps)

        velocity_reward = torch.zeros_like(location_reward)
        if self.enable_velocity_reward:
            tar_now = self._traj_gen.calc_pos(
                env_ids, self._anchored_time(env_ids, 0))
            command_velocity = ((tar_pos[..., :2] - tar_now[..., :2])
                                / max(self._traj_sample_timestep, 1e-4))
            velocity_error = self._humanoid_root_states[..., 7:9] - command_velocity
            velocity_reward = torch.exp(
                -self.velocity_error_scale
                * torch.sum(velocity_error * velocity_error, dim=-1))

        swing_reward = torch.zeros_like(location_reward)
        landing_reward = torch.zeros_like(location_reward)
        impact_reward = torch.zeros_like(location_reward)
        safety_reward = torch.zeros_like(location_reward)
        if self.enable_safe_stepping_reward:
            swing_reward, landing_reward, impact_reward = \
                self._compute_safe_stepping_terms(root_pos, tar_pos)
            safety_reward = new_reward_scale * torch.clamp(
                self.collision_reward_weight * swing_reward
                + self.landing_reward_weight * landing_reward
                + self.impact_reward_weight * impact_reward,
                min=-self.safe_reward_clip, max=self.safe_reward_clip)
        elif self.enable_collision_penalty:
            collision_scale = self._warmup_scale(
                self._collision_reward_step,
                self.collision_reward_warmup_steps)
            safety_reward = (self.collision_reward_weight * collision_scale
                             * self._compute_foot_collision_penalty())
            self._collision_reward_step += 1

        edge_reward = torch.zeros_like(location_reward)
        if self.enable_edge_penalty:
            edge_scale = self._warmup_scale(
                self._edge_reward_step, self.edge_reward_warmup_steps)
            edge_reward = (self.edge_reward_weight * edge_scale
                           * self._compute_edge_stepping_penalty(root_pos, tar_pos))
            self._edge_reward_step += 1

        if self.power_reward:
            self.rew_buf[:] = (location_reward + power_reward
                               + new_reward_scale * self.velocity_reward_weight
                               * velocity_reward
                               + safety_reward + edge_reward)
        else:
            self.rew_buf[:] = (location_reward
                               + new_reward_scale * self.velocity_reward_weight
                               * velocity_reward
                               + safety_reward + edge_reward)
        if self.enable_safe_stepping_reward or self.enable_velocity_reward:
            self._safe_reward_step += 1
        self.reward_raw[:] = torch.stack(
            [location_reward, power_reward, velocity_reward, swing_reward,
             landing_reward, impact_reward, safety_reward], dim=-1)

        if self.reward_debug_on and not self._rwdbg_done:
            self._accumulate_reward_debug(
                location_reward, power_reward, velocity_reward,
                new_reward_scale * self.velocity_reward_weight * velocity_reward,
                swing_reward, landing_reward, impact_reward,
                safety_reward, edge_reward, self.rew_buf)

        return

    def _accumulate_reward_debug(self, *terms):
        """Accumulate per-term batch means; dump a CSV once the step budget is met."""
        vec = torch.stack([t.detach().float().mean() for t in terms])
        if self._rwdbg_sum is None:
            self._rwdbg_sum = vec.clone()
            self._rwdbg_sq = vec * vec
        else:
            self._rwdbg_sum += vec
            self._rwdbg_sq += vec * vec
        self._rwdbg_n += 1
        if self._rwdbg_n >= int(getattr(flags, "reward_debug_steps", 600)):
            self._dump_reward_debug()

    def _dump_reward_debug(self):
        import csv
        import os

        n = max(self._rwdbg_n, 1)
        mean = (self._rwdbg_sum / n).cpu().numpy()
        var = (self._rwdbg_sq / n - (self._rwdbg_sum / n) ** 2).clamp(min=0.0)
        std = torch.sqrt(var).cpu().numpy()
        out = getattr(flags, "reward_debug_out", "") or "output/reward_debug.csv"
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        total = float(np.abs(mean).sum()) + 1e-12
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["term", "mean", "std", "abs_share"])
            for i, name in enumerate(self.reward_debug_names):
                writer.writerow([name, f"{mean[i]:.6f}", f"{std[i]:.6f}",
                                 f"{abs(mean[i]) / total:.4f}"])
        print(f"[reward_debug] {n} steps -> {out}")
        for i, name in enumerate(self.reward_debug_names):
            print(f"  {name:18s} mean {mean[i]:+.4f}  std {std[i]:.4f}")
        self._rwdbg_done = True

        return


from isaacgym.terrain_utils import *
from pacer.utils.draw_utils import *


def straight_stairs_terrain(terrain, step_width, step_height, stair_width,
                            num_steps, start_dist, platform_len=1.5,
                            side_depth=1.5, ascend=True, spawn_inset=0.20,
                            approach_buf=1.0):
    """One straight stair flight inside a single tile.

    Layout along +x (the walking direction):

        [0, approach_buf)                       flat buffer
        [.., + start_dist)                      flat approach
        num_steps steps of step_width/step_height
        [.., + platform_len)                    flat landing at the far level
        beyond                                  pit

    The robot spawns on a thin strip at the end of the buffer, so there is flat
    ground behind it (otherwise it stands on the lip of the previous tile's pit).

    ``ascend`` puts the approach at height 0 and climbs; ``ascend=False`` starts
    at the top level and descends.  The corridor is ``stair_width`` metres clear
    in y; everything outside is a pit ``side_depth`` metres deep, so W really
    changes the collision mesh and the walkable area (doc §7).

    Returns (walk, spawn) boolean masks with the tile's shape.
    """
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    n_rows, n_cols = terrain.height_field_raw.shape

    step_w = max(int(round(step_width / hs)), 1)
    step_h = int(round(step_height / vs))
    buf_px = max(int(round(approach_buf / hs)), 1)
    start_px = max(buf_px + int(round(start_dist / hs)), buf_px + 1)
    half_w = max(int(round(stair_width / 2.0 / hs)), 1)
    plat_px = max(int(round(platform_len / hs)), 0)
    pit = -int(round(side_depth / vs))

    cy = n_cols // 2
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, n_cols)
    top_h = num_steps * step_h

    hf = terrain.height_field_raw
    hf[:] = pit
    walk = np.zeros(hf.shape, dtype=bool)
    spawn = np.zeros(hf.shape, dtype=bool)

    # corridor floor at the approach level
    hf[:, y0:y1] = 0 if ascend else top_h
    walk[:, y0:y1] = True

    x = start_px
    for i in range(num_steps):
        level = (i + 1) * step_h if ascend else top_h - (i + 1) * step_h
        hf[x:x + step_w, y0:y1] = level
        x += step_w
    far_level = top_h if ascend else 0
    plat_end = min(x + plat_px, n_rows)
    hf[x:plat_end, y0:y1] = far_level

    # past the landing everything is a pit again
    hf[plat_end:, :] = pit
    walk[plat_end:, :] = False

    # spawn strip at the end of the buffer, inset from both corridor edges
    strip = max(min(int(round(0.10 / hs)), buf_px), 1)
    inset = int(round(spawn_inset / hs))
    sy0 = min(y0 + inset, y1 - 1)
    sy1 = max(y1 - inset, sy0 + 1)
    spawn[buf_px - strip:buf_px, sy0:sy1] = True
    return walk, spawn


def flat_flight_terrain(terrain, step_width, step_height, stair_width,
                        up_steps, down_steps, platform_len=1.5,
                        start_dist=0.6, approach_buf=1.0, runout_len=2.0,
                        side_depth=1.5, spawn_inset=0.20):
    """One straight corridor containing  up-stairs -> platform -> down-stairs.

    Layout along +x (the walking direction):

        [0, approach_buf)            flat buffer (spawn strip at its end)
        [.., + start_dist)           flat approach
        up_steps   steps of step_width x step_height      (rises to H)
        platform_len at H
        down_steps steps                                  (falls back to 0)
        runout_len flat
        beyond                       pit

    The corridor is ``stair_width`` metres clear in y; everything outside is a
    pit ``side_depth`` metres deep, so W changes the collision mesh and the
    walkable area.  Returns (walk, spawn) boolean masks.
    """
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    n_rows, n_cols = terrain.height_field_raw.shape

    step_w = max(int(round(step_width / hs)), 1)
    step_h = int(round(step_height / vs))
    buf_px = max(int(round(approach_buf / hs)), 1)
    first_px = max(buf_px + int(round(start_dist / hs)), buf_px + 1)
    half_w = max(int(round(stair_width / 2.0 / hs)), 1)
    plat_px = max(int(round(platform_len / hs)), 0)
    runout_px = max(int(round(runout_len / hs)), 0)
    pit = -int(round(side_depth / vs))

    cy = n_cols // 2
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, n_cols)
    top_h = up_steps * step_h

    hf = terrain.height_field_raw
    hf[:] = pit
    walk = np.zeros(hf.shape, dtype=bool)
    spawn = np.zeros(hf.shape, dtype=bool)

    # flat floor for the whole corridor, steps are carved below
    hf[:, y0:y1] = 0
    walk[:, y0:y1] = True

    x = first_px
    for i in range(up_steps):
        hf[x:x + step_w, y0:y1] = (i + 1) * step_h
        x += step_w
    top_end = min(x + plat_px, n_rows)
    hf[x:top_end, y0:y1] = top_h
    x = top_end
    for i in range(down_steps):
        hf[x:x + step_w, y0:y1] = top_h - (i + 1) * step_h
        x += step_w
    end = min(x + runout_px, n_rows)
    hf[x:end, y0:y1] = 0

    # past the run-out everything is a pit again
    hf[end:, :] = pit
    walk[end:, :] = False

    # spawn strip at the end of the buffer, inset from both corridor edges
    strip = max(min(int(round(0.10 / hs)), buf_px), 1)
    inset = int(round(spawn_inset / hs))
    sy0 = min(y0 + inset, y1 - 1)
    sy1 = max(y1 - inset, sy0 + 1)
    spawn[buf_px - strip:buf_px, sy0:sy1] = True
    return walk, spawn


def poles_terrain(terrain, difficulty=1):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length

    img = np.zeros((terrain.width, terrain.length), dtype=int)
    # disk, circle, curve, poly, ellipse
    base_prob = 1 / 2
    # probs = np.array([0.7, 0.7, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    probs = np.array([0.9, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    low, high = 200, 500
    num_mult = int(stop_x // 80)

    for i in range(len(probs)):
        p = probs[i]
        if i == 0:
            for _ in range(10 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_disk(img_size=terrain.width, max_r = 7) * int(np.random.uniform(low, high))
        # elif i == 1 and np.random.binomial(1, p):
        #     for _ in range(5 * num_mult):
        #         if np.random.binomial(1, p):

        #             img += draw_circle(img_size=terrain.width, max_r=5) * int(
        #                 np.random.uniform(low, high))
        elif i == 1 and np.random.binomial(1, p):
            for _ in range(3 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_curve(img_size=terrain.width) * int(np.random.uniform(low, high))
        elif i == 2 and np.random.binomial(1, p):
            for _ in range(1 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_polygon(img_size=terrain.width, max_sides=5) * int(np.random.uniform(low, high))
        elif i == 3 and np.random.binomial(1, p):
            for _ in range(5 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_ellipse(img_size=terrain.width,
                                        max_size=5) * int(
                                            np.random.uniform(low, high))

    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = img

    return terrain


class MeshTerrain:
    def __init__(self, heigthmap_data, device):
        self.border_size = 20
        self.border = 500
        self.sample_extent_x = 300
        self.sample_extent_y = 300
        self.vertical_scale = 1
        self.device = device

        self.heightsamples = torch.from_numpy(heigthmap_data['heigthmap']).to(device)
        self.walkable_map  = torch.from_numpy(heigthmap_data['walkable_map']).to(device)
        self.cam_pos = torch.from_numpy(heigthmap_data['cam_pos'])
        self.x_scale = heigthmap_data['x_scale']
        self.y_scale = heigthmap_data['y_scale']

        self.x_shape, self.y_shape = self.heightsamples.shape
        self.x_c = (self.x_shape / 2) / self.x_scale
        self.y_c = (self.y_shape / 2) / self.y_scale

        coord_x, coord_y = torch.where(self.walkable_map == 1) # Image coordinates, need to flip y and x
        coord_x, coord_y = coord_x.float(), coord_y.float()
        self.coord_x_scale = coord_x / self.x_scale  - self.x_c
        self.coord_y_scale = coord_y / self.y_scale  - self.y_c

        self.coord_x_scale += self.cam_pos[0]
        self.coord_y_scale += self.cam_pos[1]

        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, num_envs, env_ids):
        num_envs = env_ids.shape[0]
        idxes = np.random.randint(0, self.num_samples, size=num_envs)
        valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)
        return valid_locs

    def world_points_to_map(self, points):
        points[..., 0] -= self.cam_pos[0] - self.x_c
        points[..., 1] -= self.cam_pos[1] - self.y_c
        points[..., 0] *= self.x_scale
        points[..., 1] *= self.y_scale
        points = (points).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        px = torch.clip(px, 0, self.heightsamples.shape[0] -
                        2)  # image, so sampling 1 is for x
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py

    def sample_height_points(self,
                             points,
                             root_states = None,
                             root_points=None,
                             env_ids=None,
                             num_group_people=512,
                             group_ids=None):

        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        device = points.device
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0] / num_group_people)
            heightsamples_group = heightsamples[None, ].repeat(
                num_groups, 1, 1)


            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people * num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros(px.shape).to(px.device)

            if not root_states is None:
                linear_vel = root_states[:, 7:10]  # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1],
                                            2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape +
                                                 (3, )).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx],root_py[idx]] += torch.tensor(1.7 / self.vertical_scale)
                # heightsamples_group[idx][root_px[idx] + 1,root_py[idx] + 1] += torch.tensor(1.7 / self.vertical_scale)
                group_mask_env_ids = group_ids[env_ids] == idx  # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]

                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N)

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]

                    velocity_map_group[
                        idx, root_px[idx],
                        root_py[idx], :] = group_linear_vel.repeat(
                            1, root_points.shape[1]).view(-1, 3) # Make sure that the order is correct.

                    # Then sampling the points
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # this is one-to-one substraction of the agents in the group to mark the static terrain with relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            # import matplotlib.pyplot as plt; plt.imshow(heights[0].reshape(32, 32).cpu().numpy()); plt.show()
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim=-1)
        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)


        return heights * self.vertical_scale


class Terrain:
    def __init__(self, cfg, num_robots, device) -> None:

        self.type = cfg["terrainType"]
        self.device = device
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1 # resolution 0.1
        self.vertical_scale = 0.005
        self.border_size = 50
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [
            np.sum(cfg["terrainProportions"][:i + 1])
            for i in range(len(cfg["terrainProportions"]))
        ]
        # Randomized stair geometry (riser height and tread depth).
        self.stair_height_min = float(cfg.get("stair_height_min", 0.10))
        self.stair_height_max = float(cfg.get("stair_height_max", 0.22))
        self.stair_tread_min = float(cfg.get("stair_tread_min", 0.24))
        self.stair_tread_max = float(cfg.get("stair_tread_max", 0.38))
        # Flat area at the top of the pyramid stairs. This is the only
        # multi-metre flat patch inside a tile (spawns exclude the outer
        # border), so shrinking it shortens the run-up a robot can build up
        # before it meets the first riser.
        self.stair_platform_size = float(cfg.get("stair_platform_size", 3.0))

        # --- curriculum terrain (docs/安全奖励与万轮课程训练计划-20261108.md §7) --
        # stair_terrain: "pyramid" (original) or "straight" (one flight per tile
        # with an explicit clear width W, platform length and start distance).
        self.stair_terrain = str(cfg.get("stair_terrain", "pyramid"))
        self.stair_riser_min = float(cfg.get("stair_riser_min",
                                             cfg.get("stair_height_min", 0.10)))
        self.stair_riser_max = float(cfg.get("stair_riser_max",
                                             cfg.get("stair_height_max", 0.22)))
        # Clear width W is sampled from discrete bins when given (doc §7: uniform
        # sampling is not allowed, every bin needs a stable sample count).
        self.stair_width_bins = [float(v) for v in cfg.get("stair_width_bins", [])]
        self.stair_width_probs = [float(v) for v in cfg.get("stair_width_probs", [])]
        self.stair_width_min = float(cfg.get("stair_width_min", 2.4))
        self.stair_width_max = float(cfg.get("stair_width_max", 3.0))
        self.stair_steps_min = int(cfg.get("stair_steps_min", 5))
        self.stair_steps_max = int(cfg.get("stair_steps_max", 7))
        self.stair_platform_min = float(cfg.get("stair_platform_min",
                                                cfg.get("stair_platform_size", 3.0)))
        self.stair_platform_max = float(cfg.get("stair_platform_max",
                                                cfg.get("stair_platform_size", 3.0)))
        self.stair_start_dist_min = float(cfg.get("stair_start_dist_min", 0.4))
        self.stair_start_dist_max = float(cfg.get("stair_start_dist_max", 0.8))
        self.stair_up_steps_min = int(cfg.get("stair_up_steps_min", 5))
        self.stair_up_steps_max = int(cfg.get("stair_up_steps_max", 7))
        self.stair_down_steps_min = int(cfg.get("stair_down_steps_min", 5))
        self.stair_down_steps_max = int(cfg.get("stair_down_steps_max", 7))
        self.stair_approach_buf = float(cfg.get("stair_approach_buf", 1.0))
        self.stair_runout = float(cfg.get("stair_runout", 2.0))
        # Pin each env to one corridor: the spawn point is drawn once and reused
        # on every reset, so the stair parameters of an env never change mid-run.
        self.fixed_spawn_per_env = bool(cfg.get("fixed_spawn_per_env", True))
        self._env_spawn_locs = None
        self.stair_side_depth = float(cfg.get("stair_side_depth", 1.5))
        # Lateral inset from the corridor edge used for the spawn strip (doc
        # §6.1: at least 95% of spawns must be 0.20 m inside the edge).
        self.stair_spawn_inset = float(cfg.get("stair_spawn_inset", 0.20))

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length /
                                         self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(
            self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(
            self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        # 1 = allowed spawn point. Straight-stair tiles declare the approach strip
        # so the distance to the first riser equals stair_start_dist.
        self.spawn_mask_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int8)
        if cfg["curriculum"]:
            self.curiculum(num_robots,
                           num_terrains=self.env_cols,
                           num_levels=self.env_rows)
        else:
            self.randomized_terrain()
        self.heightsamples = torch.from_numpy(self.height_field_raw).to(self.device) # ZL: raw height field, first dimension is x, second is y
        self.walkable_field = torch.from_numpy(self.walkable_field_raw).to(self.device)
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale,cfg["slopeTreshold"])
        self.sample_extent_x = int((self.tot_rows - self.border * 2) * self.horizontal_scale)
        self.sample_extent_y = int((self.tot_cols - self.border * 2) * self.horizontal_scale)

        coord_x, coord_y = torch.where(self.walkable_field == 0)
        coord_x_scale = coord_x * self.horizontal_scale
        coord_y_scale = coord_y * self.horizontal_scale
        # Keep points inside the tile atlas. The old version subtracted a 50 m
        # border from the bbox of the walkable pixels, which silently deleted the
        # straight-stair approach strips near a tile edge.
        x_lo = self.border * self.horizontal_scale
        x_hi = (self.border + self.env_rows * self.length_per_env_pixels) * self.horizontal_scale
        y_lo = self.border * self.horizontal_scale
        y_hi = (self.border + self.env_cols * self.width_per_env_pixels) * self.horizontal_scale
        walkable_subset = torch.logical_and(
                torch.logical_and(coord_x_scale >= x_lo, coord_x_scale < x_hi),
                torch.logical_and(coord_y_scale >= y_lo, coord_y_scale < y_hi)
            )
        # import ipdb; ipdb.set_trace()
        # joblib.dump(self.walkable_field_raw, "walkable_field.pkl")

        self.coord_x_scale = coord_x_scale[walkable_subset]
        self.coord_y_scale = coord_y_scale[walkable_subset]
        self.num_samples = self.coord_x_scale.shape[0]

        # Prefer spawn points that actually sit on stairs. A point counts as
        # "on a stair" when the terrain height varies by at least
        # stair_spawn_min_spread inside a stair_spawn_window half-window, i.e.
        # there is a riser within reach rather than a flat plateau. The robot
        # then spends its rollout going up/down steps instead of walking along
        # the top of a pyramid.
        self.stair_spawn_bias = float(cfg.get("stair_spawn_bias", 0.0))
        self.stair_spawn_idx = np.zeros((0,), dtype=np.int64)
        if self.stair_spawn_bias > 0.0:
            from scipy.ndimage import maximum_filter, minimum_filter
            window = int(round(float(cfg.get("stair_spawn_window", 0.5))
                               / self.horizontal_scale))
            min_spread = int(round(float(cfg.get("stair_spawn_min_spread", 0.05))
                                   / self.vertical_scale))
            size = 2 * window + 1
            hmax = maximum_filter(self.height_field_raw, size=size)
            hmin = minimum_filter(self.height_field_raw, size=size)
            spread = hmax - hmin
            px = coord_x[walkable_subset].cpu().numpy()
            py = coord_y[walkable_subset].cpu().numpy()
            self.stair_spawn_idx = np.where(spread[px, py] >= min_spread)[0]
            if self.stair_spawn_idx.size == 0:
                print("[stair_spawn] WARNING no stair points found; "
                      "falling back to uniform spawns")
                self.stair_spawn_bias = 0.0
            else:
                print(f"[stair_spawn] {self.stair_spawn_idx.size} / "
                      f"{self.num_samples} spawn points on stairs "
                      f"(bias {self.stair_spawn_bias:.2f})")

        # Straight-stair tiles declare their own approach strip; use it directly
        # so the run-up before the first riser is exactly stair_start_dist.
        if self.spawn_mask_raw.any():
            px = coord_x[walkable_subset].cpu().numpy()
            py = coord_y[walkable_subset].cpu().numpy()
            picked = np.where(self.spawn_mask_raw[px, py] > 0)[0]
            if picked.size:
                self.stair_spawn_idx = picked
                self.stair_spawn_bias = 1.0
                print(f"[stair_spawn] {picked.size} / {self.num_samples} spawn "
                      f"points on the straight-stair approach")
            else:
                print("[stair_spawn] WARNING spawn mask has no walkable point; "
                      "falling back to uniform spawns")

    def _sample_stair_width(self):
        """Clear width W: discrete bins when configured, uniform range otherwise."""
        if (self.stair_width_bins
                and len(self.stair_width_bins) == len(self.stair_width_probs)):
            probs = np.asarray(self.stair_width_probs, dtype=float)
            total = probs.sum()
            if total > 0:
                return float(np.random.choice(self.stair_width_bins,
                                              p=probs / total))
        return float(np.random.uniform(self.stair_width_min,
                                       self.stair_width_max))

    def sample_valid_locations(self, max_num_envs, env_ids, group_num_people = 16, sample_groups = False):
        if sample_groups:
            num_groups = max_num_envs// group_num_people
            group_centers = torch.stack([torch_rand_float(0., self.sample_extent_x, (num_groups, 1),device=self.device).squeeze(1), torch_rand_float(0., self.sample_extent_y, (num_groups, 1),device=self.device).squeeze(1)], dim = -1)
            group_diffs = torch.stack([torch_rand_float(-8., 8, (num_groups, group_num_people) ,device=self.device), torch_rand_float(8., -8, (num_groups, group_num_people),device=self.device)], dim = -1)
            valid_locs = (group_centers[:, None, ] + group_diffs).reshape(max_num_envs, -1)

            if not env_ids is None:
                valid_locs = valid_locs[env_ids]
        else:
            num_envs = env_ids.shape[0]
            # Fixed spawn per env: draw the whole set once and reuse it for every
            # reset, so an env always returns to the same corridor (same h/d/W)
            # instead of teleporting to a different tile. Variation then comes
            # from env to env, not from reset to reset.
            if (getattr(self, "fixed_spawn_per_env", False)
                    and self._env_spawn_locs is not None
                    and self._env_spawn_locs.shape[0] >= int(env_ids.max()) + 1):
                return self._env_spawn_locs[env_ids]

            draw = max_num_envs if getattr(self, "fixed_spawn_per_env", False) else num_envs
            idxes = np.random.randint(0, self.num_samples, size=draw)
            if getattr(self, "stair_spawn_bias", 0.0) > 0.0 and \
                    self.stair_spawn_idx.size > 0:
                take = np.random.rand(draw) < self.stair_spawn_bias
                n_take = int(take.sum())
                if n_take > 0:
                    idxes[take] = self.stair_spawn_idx[
                        np.random.randint(0, self.stair_spawn_idx.size,
                                          size=n_take)]
            valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)
            if getattr(self, "fixed_spawn_per_env", False):
                self._env_spawn_locs = valid_locs
                valid_locs = valid_locs[env_ids]

        return valid_locs

    def world_points_to_map(self, points):
        points = (points / self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0] - 2)
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py


    def sample_height_points(self, points, root_states = None, root_points=None, env_ids = None, num_group_people = 512, group_ids = None):
        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0]/num_group_people)
            heightsamples_group = heightsamples[None, ].repeat(num_groups, 1, 1)

            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people *  num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros_like(px)

            if not root_states is None:
                linear_vel = root_states[:, 7:10] # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1], 2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape + (3,)).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx], root_py[idx]] += torch.tensor(1.7 / self.vertical_scale).short()
                group_mask_env_ids = group_ids[env_ids] == idx # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]
                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N).long()

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]
                    velocity_map_group[idx, root_px[idx], root_py[idx], :] = group_linear_vel.repeat(1, root_points.shape[1]).view(-1, 3)

                    # Sample the points for each agent's px and py
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # for each agent's velocity map, minus it's own velocity to get the relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)

            if root_states is None:
                return heights * self.vertical_scale
            else:
                velocity_map = torch.zeros((B, N, 2)).to(points)
                linear_vel = root_states[env_ids, 7:10]
                root_rot = root_states[env_ids, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                linear_vel_ego = torch_utils.my_quat_rotate(heading_rot, linear_vel)
                velocity_map[:] = velocity_map[:] - linear_vel_ego[:, None, :2] # Flip velocity to be in agent's point of view
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0.1, 1)
            slope = difficulty * 0.7
            discrete_obstacles_height = 0.025 + difficulty * 0.15
            stepping_stones_size = 2 - 1.8 * difficulty
            step_height = np.random.uniform(self.stair_height_min, self.stair_height_max)
            step_tread = np.random.uniform(self.stair_tread_min, self.stair_tread_max)
            if choice < self.proportions[0]:
                if choice < 0.05:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            elif choice < self.proportions[1]:
                if choice < 0.15:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                random_uniform_terrain(terrain,
                                       min_height=-0.1,
                                       max_height=0.1,
                                       step=0.025,
                                       downsampled_scale=0.2)
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                pyramid_stairs_terrain(terrain,
                                       step_width=step_tread,
                                       step_height=step_height,
                                       platform_size=self.stair_platform_size)
            elif choice < self.proportions[4]:
                discrete_obstacles_terrain(terrain,
                                           discrete_obstacles_height,
                                           1.,
                                           2.,
                                           40,
                                           platform_size=3.)
            elif choice < self.proportions[5]:
                stepping_stones_terrain(terrain,
                                        stone_size=stepping_stones_size,
                                        stone_distance=0.1,
                                        max_height=0.,
                                        platform_size=3.)
            elif choice < self.proportions[6]:
                poles_terrain(terrain=terrain, difficulty=difficulty)
                self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)

            elif choice < self.proportions[7]:
                # plain walking terrain
                pass

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in tqdm(range(num_terrains)):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.7
                step_height = np.random.uniform(self.stair_riser_min,
                                                self.stair_riser_max)
                step_tread = np.random.uniform(self.stair_tread_min, self.stair_tread_max)
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty

                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels

                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                    random_uniform_terrain(terrain,
                                           min_height=-0.1,
                                           max_height=0.1,
                                           step=0.025,
                                           downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if self.stair_terrain == "flat_flight":
                        # up stairs -> platform -> down stairs, one straight line
                        width_i = self._sample_stair_width()
                        up_i = int(np.random.randint(self.stair_up_steps_min,
                                                     self.stair_up_steps_max + 1))
                        down_i = int(np.random.randint(self.stair_down_steps_min,
                                                       self.stair_down_steps_max + 1))
                        start_i = float(np.random.uniform(self.stair_start_dist_min,
                                                          self.stair_start_dist_max))
                        plat_i = float(np.random.uniform(self.stair_platform_min,
                                                         self.stair_platform_max))
                        walk, spawn = flat_flight_terrain(
                            terrain,
                            step_width=step_tread,
                            step_height=abs(step_height),
                            stair_width=width_i,
                            up_steps=up_i,
                            down_steps=down_i,
                            platform_len=plat_i,
                            start_dist=start_i,
                            approach_buf=self.stair_approach_buf,
                            runout_len=self.stair_runout,
                            side_depth=self.stair_side_depth,
                            spawn_inset=self.stair_spawn_inset)
                        self.walkable_field_raw[start_x:end_x,
                                                start_y:end_y] = (~walk).astype(int)
                        self.spawn_mask_raw[start_x:end_x,
                                            start_y:end_y] = spawn.astype(np.int8)
                    elif self.stair_terrain == "straight":
                        # h / d / W / steps / platform / start distance are all
                        # randomised per tile (doc §7: no uniform-only width).
                        width_i = self._sample_stair_width()
                        steps_i = int(np.random.randint(self.stair_steps_min,
                                                        self.stair_steps_max + 1))
                        start_i = float(np.random.uniform(self.stair_start_dist_min,
                                                          self.stair_start_dist_max))
                        plat_i = float(np.random.uniform(self.stair_platform_min,
                                                         self.stair_platform_max))
                        # Slot [proportions[1], proportions[2]) ascends, the next
                        # slot descends (doc §7 Stage 1 = 60% up / 25% down).
                        ascend = bool(choice < self.proportions[2])
                        walk, spawn = straight_stairs_terrain(
                            terrain,
                            step_width=step_tread,
                            step_height=abs(step_height),
                            stair_width=width_i,
                            num_steps=steps_i,
                            start_dist=start_i,
                            platform_len=plat_i,
                            side_depth=self.stair_side_depth,
                            ascend=ascend,
                            spawn_inset=self.stair_spawn_inset)
                        self.walkable_field_raw[start_x:end_x,
                                                start_y:end_y] = (~walk).astype(int)
                        self.spawn_mask_raw[start_x:end_x,
                                            start_y:end_y] = spawn.astype(np.int8)
                    else:
                        # Pyramid stairs. The old code negated the step height
                        # for every tile in the first stair slot, so the slot
                        # labelled "stairs up" actually descended. Decide the
                        # direction from the slot instead (doc §7: 60% up).
                        ascend = bool(choice < self.proportions[2])
                        pyramid_stairs_terrain(terrain,
                                               step_width=step_tread,
                                               step_height=(abs(step_height)
                                                            if ascend else -abs(step_height)),
                                               platform_size=self.stair_platform_size)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain,
                                               discrete_obstacles_height,
                                               1.,
                                               2.,
                                               40,
                                               platform_size=3.)
                elif choice < self.proportions[5]:
                    stepping_stones_terrain(terrain,
                                            stone_size=stepping_stones_size,
                                            stone_distance=0.1,
                                            max_height=0.,
                                            platform_size=3.)
                elif choice < self.proportions[6]:
                    poles_terrain(terrain=terrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)
                    # self.walkable_field_raw[start_x:end_x, start_y:end_y] = 1
                elif choice < self.proportions[7]:
                    # plain walking terrain
                    pass

                # Heightfield coordinate system
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(
                    terrain.height_field_raw[x1:x2,
                                             y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [
                    env_origin_x, env_origin_y, env_origin_z
                ]

        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)




@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, safety_failure,
                           disableCollision):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        has_fallen = safety_failure
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[...,0:2]  # also reset if toooo far away from the target trajectory
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        if disableCollision:
            has_fallen[:] = False
        has_failed = torch.logical_or(has_fallen, tar_fail)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# Task-location
@torch.jit.script
def compute_location_observations(root_states, traj_samples, upright):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]))
    heading_rot_exp = torch.reshape(
        heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1],
                          heading_rot_exp.shape[2]))
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
         traj_samples_delta.shape[2]))

    local_traj_pos = torch_utils.my_quat_rotate(heading_rot_exp,
                                                traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(local_traj_pos,
                        (traj_samples.shape[0],
                         traj_samples.shape[1] * local_traj_pos.shape[1]))
    return obs



@torch.jit.script
def compute_location_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward

@torch.jit.script
def compute_location_reward_fuzzy(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0
    radius = 0.0025
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_err[pos_err < radius] = 0 # 5cm radius around target is perfect.

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward

 


@torch.jit.script
def compute_group_observation(body_pos, body_rot, body_vel, selected_jts, num_group_people, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
    # joints + root velocities
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    top_k = 5
    selected_jts = [0, 1, 5, 9, 3, 7, 16, 21, 18, 23]
    num_selected_jts = len(selected_jts)

    B, J, _ = body_pos.shape
    group_pos = body_pos.view(-1, num_group_people, J, 3)
    group_vel = body_vel.view(-1, num_group_people, J, 3)
    group_root_pos = group_pos[..., 0, :]
    repeated_root_pos = group_root_pos.repeat(1, num_group_people, 1).view(-1, num_group_people, num_group_people, 3)

    indexes = torch.arange(B).to(body_pos.device)
    dist = torch.norm(group_root_pos[..., None, :] - repeated_root_pos, dim = -1)
    topk_dist, topk_idx = torch.topk(dist, top_k + 1, dim = -1, largest = False)
    topk_dist, topk_idx = topk_dist[..., 1:], topk_idx[..., 1:]
    topk_mask = (topk_dist > 10).view(-1)

    repeated_indexes = indexes.view(-1, num_group_people).repeat(1, num_group_people).view(-1, num_group_people, num_group_people)
    selected_idxes = torch.gather(repeated_indexes, -1, topk_idx).flatten()

    selected_pos = body_pos[selected_idxes][:, selected_jts].view(B, -1, 3)
    selected_vel = body_vel[selected_idxes][:, [0]].view(B, -1, 3)

    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = selected_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.view(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)

    flat_body_vel = selected_vel.view(selected_vel.shape[0] * selected_vel.shape[1], selected_vel.shape[2])
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_vel.shape[1], 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)

    local_body_pos = flat_local_body_pos.view(-1, num_selected_jts, 3)
    local_body_vel = flat_local_body_vel.view(-1, 1, 3)

    local_body_pos[topk_mask], local_body_vel[topk_mask] = 0, 0
    group_obs = torch.cat([local_body_pos.view(B, -1, top_k, 3), local_body_vel.view(B, -1, top_k, 3)], dim = 1).view(B, -1)

    return group_obs
