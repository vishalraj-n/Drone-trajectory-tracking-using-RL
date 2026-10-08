"""Train the CTBT trajectory policy with the shared Isaac Lab install."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--trajectory_bank", type=str, default=str(PROJECT_ROOT / "assets/trajectories/train.npz"))
parser.add_argument(
    "--online_trajectories",
    action="store_true",
    help="Opt into fresh online spline generation. Default training uses the fixed adaptive-reference pool.",
)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_steps_per_env", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import ctbt_drone.envs  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ctbt_drone.agents.ppo_cfg import CTTrajectoryPPORunnerCfg  # noqa: E402
from ctbt_drone.envs.ctbt_trajectory_env import CTBTTrajectoryEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = CTBTTrajectoryEnvCfg()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.trajectory_path = str(pathlib.Path(args.trajectory_bank).expanduser().resolve())
    env_cfg.online_trajectory_generation = args.online_trajectories
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(CTTrajectoryPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args.learning_rate
    if args.device is not None:
        agent_cfg.device = args.device

    env = gym.make("Isaac-Crazyflie-CTBT-Trajectory-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = args.log_dir or str(PROJECT_ROOT / "logs" / "crazyflie_ctbt_trajectory")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    print(f"Training with bank: {env_cfg.trajectory_path}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
