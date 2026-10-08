"""Run a trained CTBT policy interactively in Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--trajectory_bank", type=str, default=str(PROJECT_ROOT / "assets/trajectories/test.npz"))
parser.add_argument(
    "--trajectory_index",
    type=int,
    default=None,
    help="Exact bank entry to follow (zero-based); overrides --trajectory_type.",
)
parser.add_argument(
    "--trajectory_type",
    choices=("random", "circle", "figure8", "spline", "straight", "lissajous"),
    default="random",
    help="Select the first bank entry from this family, or randomize when set to random.",
)
parser.add_argument("--episode_seconds", type=float, default=20.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--top_view", action="store_true", help="Use a top-down camera for trajectory inspection.")
parser.add_argument("--trail_points", type=int, default=300, help="Number of actual-path marker points.")
parser.add_argument("--reference_stride", type=int, default=4, help="Stride used to draw the reference path.")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize before exiting.")
parser.add_argument("--video", action="store_true", help="Save the first inference episode as an MP4 video.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=str(PROJECT_ROOT / "videos" / "inference"),
    help="Directory where the recorded MP4 is written.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Maximum recorded steps; 0 records until the first episode ends.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
# RecordVideo needs camera rendering enabled before Isaac Sim is launched.
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
import ctbt_drone.envs  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ctbt_drone.agents.ppo_cfg import CTTrajectoryPPORunnerCfg  # noqa: E402
from ctbt_drone.envs.ctbt_trajectory_env import CTBTTrajectoryEnvCfg  # noqa: E402


def main() -> None:
    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bank_path = pathlib.Path(args.trajectory_bank).expanduser().resolve()
    bank_data = np.load(bank_path)
    family_ids = None
    if isinstance(bank_data, np.lib.npyio.NpzFile) and "family_ids" in bank_data.files:
        family_ids = np.asarray(bank_data["family_ids"]).astype(str)
    env_cfg = CTBTTrajectoryEnvCfg()
    env_cfg.episode_length_s = args.episode_seconds
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.trajectory_path = str(bank_path)
    if args.trajectory_index is not None:
        env_cfg.trajectory_index = args.trajectory_index
    elif args.trajectory_type != "random":
        if family_ids is None:
            raise ValueError("This trajectory bank has no family_ids. Regenerate it with scripts/generate_trajectories.py.")
        matches = np.flatnonzero(family_ids == args.trajectory_type)
        if len(matches) == 0:
            raise ValueError(f"No trajectory family {args.trajectory_type!r} exists in {bank_path}.")
        env_cfg.trajectory_index = int(matches[0])
    else:
        env_cfg.trajectory_index = -1
    # The bundled test bank contains roughly 16 seconds of samples while the
    # default playback is 20 seconds.  Do not wrap sample N back to sample 0:
    # that would create a discontinuous target and make the drone appear to
    # teleport.  The environment holds the final waypoint until the episode
    # ends.  Set this to True only for a deliberately seamless periodic bank.
    env_cfg.trajectory_loop = False
    env_cfg.allow_trajectory_hold = True
    env_cfg.online_trajectory_generation = False
    env_cfg.terminate_on_crash = False
    env_cfg.seed = args.seed
    env_cfg.scene.clone_in_fabric = False
    env_cfg.debug_vis = True
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(CTTrajectoryPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    if args.device is not None:
        agent_cfg.device = args.device
    env = gym.make(
        "Isaac-Crazyflie-CTBT-Trajectory-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args.video else None,
    )
    if args.video:
        video_dir = pathlib.Path(args.video_dir).expanduser().resolve()
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "disable_logger": True,
        }
        if args.video_length > 0:
            video_kwargs["video_length"] = args.video_length
        print(f"Recording inference video in {video_dir}.", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.top_view:
        env.unwrapped.sim.set_camera_view(eye=(0.0, 0.0, 8.0), target=(0.0, 0.0, 0.0))
    else:
        env.unwrapped.sim.set_camera_view(eye=(1.8, 1.8, 1.4), target=(0.0, 0.0, 0.8))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(str(checkpoint), metadata.version("rsl-rl-lib")))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    # Draw the selected reference path as blue spheres and the executed path
    # as red spheres. Markers are intentionally sampled sparsely to keep the
    # viewport responsive for long trajectories.
    unwrapped = env.unwrapped
    reference_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CTBT/reference_trajectory",
        markers={
            "reference": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 1.0)),
            )
        },
    )
    actual_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CTBT/actual_trajectory",
        markers={
            "actual": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.08, 0.03)),
            )
        },
    )
    reference_marker = VisualizationMarkers(reference_cfg)
    actual_marker = VisualizationMarkers(actual_cfg)
    def draw_reference_path() -> None:
        trajectory = unwrapped._trajectory_bank[unwrapped._trajectory_ids[0]]
        trajectory_start = int(unwrapped._trajectory_offsets[0].item())
        reference_positions = trajectory[trajectory_start:: max(1, args.reference_stride), :3].clone()
        reference_positions[:, :3] += unwrapped.scene.env_origins[0]
        reference_marker.visualize(translations=reference_positions)

    draw_reference_path()
    actual_points = torch.zeros(args.trail_points, 3, device=unwrapped.device)
    actual_count = 0
    completed_episodes = 0
    initial_position = unwrapped._robot.data.root_pos_w[0].detach()
    actual_points[:] = initial_position
    actual_marker.visualize(translations=actual_points)
    print(
        f"Running {checkpoint}; trajectory index={env_cfg.trajectory_index}; "
        f"playback duration={args.episode_seconds:.1f} s. Close Isaac Sim or press Ctrl+C to stop.",
        flush=True,
    )
    step = 0
    try:
        while simulation_app.is_running():
            start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            if bool(dones[0].item()):
                completed_episodes += 1
                print(f"Episode {completed_episodes} finished; reset={completed_episodes < args.episodes}.", flush=True)
                if completed_episodes >= args.episodes:
                    break
                # DirectRLEnv has already reset the state and selected a new
                # trajectory. Keep the visualization synchronized with it.
                draw_reference_path()
                current_position = unwrapped._robot.data.root_pos_w[0].detach()
                actual_points[:] = current_position
                actual_count = 0
            current_position = unwrapped._robot.data.root_pos_w[0].detach()
            if actual_count < args.trail_points:
                actual_points[actual_count] = current_position
                actual_count += 1
            else:
                actual_points[:-1] = actual_points[1:].clone()
                actual_points[-1] = current_position
            actual_marker.visualize(translations=actual_points)
            if step % args.print_every == 0:
                position = env.unwrapped._robot.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
                target = env.unwrapped._desired_pos[0]
                error = torch.linalg.vector_norm(position - target).item()
                print(f"step={step:06d} pos={position.detach().cpu().numpy()} target={target.detach().cpu().numpy()} error={error:.3f} reward={reward[0].item():.3f}", flush=True)
            step += 1
            if args.real_time:
                time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
