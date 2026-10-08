"""Headless multi-seed CTBT policy evaluation with CSV telemetry."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--trajectory_bank", type=str, default=str(PROJECT_ROOT / "assets/trajectories/test.npz"))
parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "evaluation.csv"))
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import ctbt_drone.envs  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ctbt_drone.agents.ppo_cfg import CTTrajectoryPPORunnerCfg  # noqa: E402
from ctbt_drone.envs.ctbt_trajectory_env import CTBTTrajectoryEnvCfg  # noqa: E402


def main() -> None:
    cfg = CTBTTrajectoryEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.trajectory_path = str(pathlib.Path(args.trajectory_bank).expanduser().resolve())
    cfg.online_trajectory_generation = False
    cfg.scene.clone_in_fabric = False
    if args.device is not None:
        cfg.sim.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(CTTrajectoryPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    if args.device is not None:
        agent_cfg.device = args.device
    env = gym.make("Isaac-Crazyflie-CTBT-Trajectory-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(str(pathlib.Path(args.checkpoint).expanduser().resolve()), metadata.version("rsl-rl-lib"))
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    totals = torch.zeros(env.unwrapped.num_envs, 4, device=env.unwrapped.device)
    counts = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    rows = []
    target_episodes = args.episodes * env.unwrapped.num_envs
    while counts.sum().item() < target_episodes:
        with torch.inference_mode():
            action = policy(obs)
            obs, reward, dones, _ = env.step(action)
            if hasattr(policy, "reset"):
                policy.reset(dones)
        position = env.unwrapped._robot.data.root_pos_w - env.unwrapped.scene.env_origins
        error = torch.linalg.vector_norm(position - env.unwrapped._desired_pos, dim=-1)
        totals[:, 0] += error
        totals[:, 1] += error.square()
        totals[:, 2] += env.unwrapped._saturation
        totals[:, 3] += reward
        counts += dones.float()
        for i in torch.where(dones)[0].tolist():
            rows.append({"env": i, "episode": int(counts[i].item()), "mean_position_error": float(totals[i, 0].item() / max(env.unwrapped.max_episode_length, 1)), "rmse_position_error": float((totals[i, 1].item() / max(env.unwrapped.max_episode_length, 1)) ** 0.5), "mean_saturation": float(totals[i, 2].item() / max(env.unwrapped.max_episode_length, 1)), "return": float(totals[i, 3].item())})
            totals[i] = 0.0
    output = pathlib.Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} episode results to {output}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
