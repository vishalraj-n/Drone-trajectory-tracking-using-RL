"""Nominal Crazyflie CTBT trajectory-tracking environment."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi
from isaaclab_assets import CRAZYFLIE_CFG

from ctbt_drone.dynamics import QuadrotorGeometry, project_ctbt_to_rotor_thrust, rotor_thrust_to_body_torque
from ctbt_drone.trajectories.generators import feasible, random_waypoint_spline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = PROJECT_ROOT / "assets" / "trajectories" / "train.npz"


def official_crazyflie_cfg() -> ArticulationCfg:
    cfg = copy.deepcopy(CRAZYFLIE_CFG)
    cfg.prim_path = "/World/envs/env_.*/Robot"
    schema_uri = CRAZYFLIE_CFG.spawn.usd_path.replace("cf2x.usd", "configuration/cf2x_robot_schema.usd")
    cache = PROJECT_ROOT / "assets" / "cache"
    cfg.spawn.usd_path = retrieve_file_path(CRAZYFLIE_CFG.spawn.usd_path, download_dir=str(cache), force_download=False)
    retrieve_file_path(schema_uri, download_dir=str(cache / "configuration"), force_download=False)
    return cfg


LOCAL_CRAZYFLIE_CFG = official_crazyflie_cfg()


@configclass
class CTBTTrajectoryEnvCfg(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 2
    action_space = 4
    current_dim = 17  # position error, velocity error, quaternion, rates, previous action
    trajectory_horizon = 100
    trajectory_dim = 6
    # mass, diagonal inertia, thrust-to-weight, K_F, wind xyz
    privileged_dim = 9
    observation_space = {
        "current": [current_dim],
        "trajectory": [1, trajectory_horizon, trajectory_dim],
        "critic_current": [current_dim],
        "privileged": [privileged_dim],
    }
    state_space = {}
    debug_vis = False
    trajectory_path = str(DEFAULT_BANK)
    # -1 preserves the training/evaluation behavior of sampling a bank entry.
    # Inference can set this to a non-negative bank index to follow one entry.
    trajectory_index = -1
    trajectory_loop = False
    # Evaluation may use a shorter bank than the requested playback duration.
    # In that case, clamp to the final waypoint instead of wrapping to the
    # beginning and creating a discontinuous target.
    allow_trajectory_hold = False
    # Training can generate a fresh randomized quintic spline at every reset.
    # Evaluation/playback explicitly disable this and use trajectory_path.
    online_trajectory_generation = False
    terminate_on_crash = True
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )
    robot: ArticulationCfg = LOCAL_CRAZYFLIE_CFG

    nominal_mass = 0.027
    arm_length = 0.046
    max_rotor_thrust = 0.12
    max_torque = (0.002, 0.002, 0.0008)
    thrust_coefficient = 1.30e-7
    torque_coefficient = 1.56e-9
    max_tilt_rad = math.radians(55.0)
    position_scale = 1.0
    velocity_scale = 1.0
    angular_rate_scale = 2.0
    max_position_error = 3.0
    max_velocity = 5.0
    max_angular_rate = 20.0
    reward_position_margin = 0.75
    reward_position_weight = 0.50
    reward_proximity_margin = 0.05
    reward_proximity_weight = 0.1625
    reward_velocity_margin = 0.50
    reward_velocity_weight = 0.20
    reward_angular_velocity_margin = 0.50
    reward_angular_velocity_weight = 0.0625
    reward_smoothness_margin = 0.50
    reward_smoothness_weight = 0.0375
    reward_yaw_margin = 0.50
    reward_yaw_weight = 0.0375
    crash_penalty = -100.0
    fixed_trajectory_type = "mixed"
    seed = 42


class CTBTTrajectoryEnv(DirectRLEnv):
    """Actor sees task state and trajectory; critic additionally sees truth."""

    cfg: CTBTTrajectoryEnvCfg

    def __init__(self, cfg: CTBTTrajectoryEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg, render_mode, **kwargs)
        body_ids, _ = self._robot.find_bodies("body")
        self._body_id = body_ids if len(body_ids) else self._robot.find_bodies(".*")[0]
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        loaded_bank = self._load_bank(cfg.trajectory_path).to(self.device)
        self._bank_count, self._trajectory_length, _ = loaded_bank.shape
        required_length = cfg.trajectory_horizon + int(cfg.episode_length_s / self.step_dt)
        if not cfg.trajectory_loop and not cfg.allow_trajectory_hold and self._trajectory_length <= required_length:
            raise ValueError("Trajectory bank entries must outlive an entire episode plus the CNN horizon")
        if cfg.trajectory_index >= self._bank_count:
            raise IndexError(
                f"trajectory_index={cfg.trajectory_index} is outside the bank "
                f"[0, {self._bank_count - 1}]"
            )
        self._online_trajectory_generation = bool(cfg.online_trajectory_generation and cfg.trajectory_index < 0)
        if self._online_trajectory_generation:
            self._trajectory_bank = torch.zeros(
                self.num_envs, self._trajectory_length, 12, device=self.device
            )
            self._bank_count = self.num_envs
        else:
            self._trajectory_bank = loaded_bank
        self._trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._trajectory_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._trajectory_reset_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reference = torch.zeros(self.num_envs, cfg.trajectory_horizon, 6, device=self.device)
        self._desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_acc = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_jerk = torch.zeros(self.num_envs, 3, device=self.device)
        self._geometry = QuadrotorGeometry(cfg.arm_length, cfg.thrust_coefficient, cfg.torque_coefficient)
        self._allocation = self._geometry.allocation_matrix(self.device)
        self._max_thrust = torch.full((self.num_envs, 4), cfg.max_rotor_thrust, device=self.device)
        self._last_rotor_thrust = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_ctbt = torch.zeros(self.num_envs, 4, device=self.device)
        self._saturation = torch.zeros(self.num_envs, device=self.device)
        self._mass = torch.full((self.num_envs,), cfg.nominal_mass, device=self.device)
        self._inertia = torch.tensor((1.4e-5, 1.4e-5, 2.2e-5), device=self.device).expand(self.num_envs, -1)
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("position", "proximity", "velocity", "angular_velocity", "smoothness", "yaw", "crash", "saturation")
        }

    def _generate_online_trajectory(self, env_id: int) -> None:
        """Generate one feasible adaptive-style trajectory for a reset."""
        reset_number = int(self._trajectory_reset_counts[env_id].item())
        seed = int(self.cfg.seed) + 1009 * env_id + 9176 * reset_number
        rng = np.random.default_rng(seed)
        for _ in range(100):
            trajectory = random_waypoint_spline(rng, self._trajectory_length, self.step_dt)
            if feasible(trajectory, self.step_dt, max_speed=1.5, max_acceleration=3.5, max_jerk=12.0):
                self._trajectory_bank[env_id] = torch.as_tensor(trajectory, device=self.device)
                self._trajectory_reset_counts[env_id] += 1
                return
        raise RuntimeError("Could not generate a feasible online trajectory after 100 attempts")

    @staticmethod
    def _load_bank(path: str) -> torch.Tensor:
        bank_path = Path(path).expanduser().resolve()
        if not bank_path.is_file():
            raise FileNotFoundError(f"Trajectory bank not found: {bank_path}. Run scripts/generate_trajectories.py first.")
        data = np.load(bank_path)
        trajectories = data["trajectories"] if isinstance(data, np.lib.npyio.NpzFile) else data
        if trajectories.ndim != 3 or trajectories.shape[-1] < 6:
            raise ValueError("Trajectory bank must have shape [num_trajectories, length, >=6]")
        return torch.as_tensor(trajectories[:, :, :12], dtype=torch.float32)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        ground_cfg = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.025))
        self.scene.clone_environments(copy_from_source=False)
        sim_utils.DomeLightCfg(intensity=1800.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        hover = self._mass * abs(float(self.sim.cfg.gravity[2]))
        collective = hover * (1.0 + 0.75 * self._actions[:, 0])
        torque_scale = torch.tensor(self.cfg.max_torque, device=self.device)
        ctbt = torch.cat((collective[:, None], self._actions[:, 1:] * torque_scale), dim=-1)
        thrust, self._saturation = project_ctbt_to_rotor_thrust(ctbt, self._allocation, self._max_thrust)
        # Nominal actuator equation: T_i = K_F * Omega_i^2. Motor lag and
        # command delay are deliberately deferred to the next phase.
        rotor_omega_squared = thrust / self.cfg.thrust_coefficient
        thrust = self.cfg.thrust_coefficient * rotor_omega_squared
        torque = rotor_thrust_to_body_torque(thrust, self._geometry)
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = thrust.sum(dim=-1)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques[:, 0] = torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=forces, torques=torques
        )
        self._last_rotor_thrust = thrust
        self._last_ctbt = ctbt

    def _trajectory_at(self, env_ids: torch.Tensor, horizon: int) -> torch.Tensor:
        starts = self.episode_length_buf[env_ids].long() + self._trajectory_offsets[env_ids]
        indexes = starts[:, None] + torch.arange(horizon, device=self.device)[None, :]
        if self.cfg.trajectory_loop:
            indexes = indexes.remainder(self._trajectory_length)
        else:
            indexes = indexes.clamp_max(self._trajectory_length - 1)
        return self._trajectory_bank[self._trajectory_ids[env_ids]][torch.arange(len(env_ids), device=self.device)[:, None], indexes]

    def _refresh_reference(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        data = self._trajectory_at(env_ids, self.cfg.trajectory_horizon)
        position = self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        velocity = self._robot.data.root_lin_vel_w[env_ids]
        self._reference[env_ids] = torch.cat(
            ((data[:, :, :3] - position[:, None]) / self.cfg.position_scale,
             (data[:, :, 3:6] - velocity[:, None]) / self.cfg.velocity_scale), dim=-1
        )
        self._desired_pos[env_ids] = data[:, 0, :3]
        self._desired_vel[env_ids] = data[:, 0, 3:6]
        self._desired_acc[env_ids] = data[:, 0, 6:9]
        self._desired_jerk[env_ids] = data[:, 0, 9:12]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_reference()
        position = self._robot.data.root_pos_w - self.scene.env_origins
        position_error_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_pos - position)
        velocity_error_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_vel - self._robot.data.root_lin_vel_w)
        attitude = self._robot.data.root_quat_w
        attitude = torch.where(attitude[:, :1] < 0.0, -attitude, attitude)
        current = torch.cat(
            (
                position_error_b / self.cfg.position_scale,
                velocity_error_b / self.cfg.velocity_scale,
                attitude,
                self._robot.data.root_ang_vel_b / self.cfg.angular_rate_scale,
                self._actions,
            ), dim=-1
        )
        gravity = abs(float(self.sim.cfg.gravity[2]))
        thrust_to_weight = (4.0 * self.cfg.max_rotor_thrust) / (self._mass * gravity)
        nominal_thrust_to_weight = (4.0 * self.cfg.max_rotor_thrust) / (self.cfg.nominal_mass * gravity)
        nominal_inertia = torch.tensor((1.4e-5, 1.4e-5, 2.2e-5), device=self.device)
        privileged = torch.cat(
            (
                self._mass[:, None] / self.cfg.nominal_mass,
                self._inertia / nominal_inertia,
                (thrust_to_weight / nominal_thrust_to_weight)[:, None],
                torch.full((self.num_envs, 1), self.cfg.thrust_coefficient / 1.30e-7, device=self.device),
                torch.zeros(self.num_envs, 3, device=self.device),  # wind is disabled in phase one
            ), dim=-1
        )
        return {
            "current": current,
            "trajectory": self._reference[:, None],
            "critic_current": current,
            "privileged": privileged,
        }

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_reference()
        position = self._robot.data.root_pos_w - self.scene.env_origins
        position_error = torch.linalg.vector_norm(self._desired_pos - position, dim=-1)
        velocity_error = torch.linalg.vector_norm(self._desired_vel - self._robot.data.root_lin_vel_w, dim=-1)
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        yaw_error = wrap_to_pi(-yaw).abs()
        angular_velocity_error = torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, dim=-1)
        action_smoothness = torch.linalg.vector_norm(self._actions - self._previous_actions, dim=-1)
        tilt = torch.linalg.vector_norm(self._robot.data.projected_gravity_b[:, :2], dim=-1)
        crash = (position[:, 2] < 0.08) | (position[:, 2] > 3.0) | (tilt > math.sin(self.cfg.max_tilt_rad))
        # dm_control Gaussian tolerance: value is 1 at zero error and 0.1 at
        # the configured margin, matching adapt-drones' reward helper.
        gaussian = lambda error, margin: torch.pow(
            torch.tensor(0.1, device=self.device), torch.square(error / margin)
        )
        position_reward = gaussian(position_error, self.cfg.reward_position_margin)
        proximity_reward = gaussian(position_error, self.cfg.reward_proximity_margin)
        velocity_reward = gaussian(velocity_error, self.cfg.reward_velocity_margin)
        angular_velocity_reward = gaussian(angular_velocity_error, self.cfg.reward_angular_velocity_margin)
        smoothness_reward = gaussian(action_smoothness, self.cfg.reward_smoothness_margin)
        yaw_reward = gaussian(yaw_error, self.cfg.reward_yaw_margin)
        reward = (
            self.cfg.reward_position_weight * position_reward
            + self.cfg.reward_proximity_weight * proximity_reward
            + self.cfg.reward_velocity_weight * velocity_reward
            + self.cfg.reward_angular_velocity_weight * angular_velocity_reward
            + self.cfg.reward_smoothness_weight * smoothness_reward
            + self.cfg.reward_yaw_weight * yaw_reward
            + self.cfg.crash_penalty * crash.float()
        )
        for name, value in (("position", position_reward), ("proximity", proximity_reward), ("velocity", velocity_reward), ("angular_velocity", angular_velocity_reward), ("smoothness", smoothness_reward), ("yaw", yaw_reward)):
            self._episode_sums[name] += value
        self._episode_sums["crash"] += crash.float()
        self._episode_sums["saturation"] += self._saturation
        self.extras = {"episode": {f"Episode/{name}": value.mean() for name, value in self._episode_sums.items()}}
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._robot.data.root_pos_w - self.scene.env_origins
        tilt = torch.linalg.vector_norm(self._robot.data.projected_gravity_b[:, :2], dim=-1)
        died = (position[:, 2] < 0.08) | (position[:, 2] > 3.0) | (tilt > math.sin(self.cfg.max_tilt_rad))
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.terminate_on_crash:
            died = torch.zeros_like(died)
        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = env_ids.to(self.device)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self._online_trajectory_generation:
            for env_id in env_ids.tolist():
                self._generate_online_trajectory(int(env_id))
            self._trajectory_ids[env_ids] = env_ids
            self._trajectory_offsets[env_ids] = 0
        elif self.cfg.trajectory_index >= 0:
            self._trajectory_ids[env_ids] = self.cfg.trajectory_index
            self._trajectory_offsets[env_ids] = 0
        else:
            self._trajectory_ids[env_ids] = torch.randint(self._bank_count, (len(env_ids),), device=self.device)
            max_offset = self._trajectory_length - self.cfg.trajectory_horizon - self.max_episode_length
            self._trajectory_offsets[env_ids] = torch.randint(max(1, max_offset), (len(env_ids),), device=self.device)
        default = self._robot.data.default_root_state[env_ids].clone()
        first = self._trajectory_bank[self._trajectory_ids[env_ids], self._trajectory_offsets[env_ids], :3]
        default[:, :3] = self.scene.env_origins[env_ids] + first
        default[:, 7:13] = 0.0
        default[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
        )
        self._robot.write_root_pose_to_sim(default[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default[:, 7:], env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_rotor_thrust[env_ids] = 0.0
        self._last_ctbt[env_ids] = 0.0
        self._saturation[env_ids] = 0.0
        for value in self._episode_sums.values():
            value[env_ids] = 0.0
