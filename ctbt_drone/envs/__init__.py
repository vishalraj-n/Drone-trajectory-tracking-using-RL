"""Gymnasium registrations for the CTBT drone tasks."""

import gymnasium as gym

from .ctbt_trajectory_env import CTBTTrajectoryEnv, CTBTTrajectoryEnvCfg

gym.register(
    id="Isaac-Crazyflie-CTBT-Trajectory-v0",
    entry_point="ctbt_drone.envs:CTBTTrajectoryEnv",
    disable_env_checker=True,
)

