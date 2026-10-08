# CTBT Crazyflie trajectory control

Public Isaac Lab project for PPO-based Crazyflie position and trajectory
tracking. The actor outputs normalized CTBT commands: collective thrust and
body roll, pitch, and yaw torques. Rotor allocation and rigid-body simulation
are handled by the environment.

This project uses an existing Isaac Sim/Isaac Lab installation. It does not
install or clone Isaac Lab. Isaac Sim and the Crazyflie asset are subject to
their respective NVIDIA licenses.

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaaclab/python
source /path/to/isaaclab/venv/bin/activate
cd "$ISAACLAB_ROOT"
```

From this project directory, generate trajectory banks first:

```bash
cd /path/to/ctbt-drone-trajectory
"$ISAACLAB_PYTHON" scripts/generate_trajectories.py --count 256
```

Training uses a fixed pool of 16-second randomized five-waypoint quintic trajectories matching the adaptive-drone reference distribution. Each reset samples a trajectory and a valid random starting offset. To opt into fresh online spline generation instead, add `--online_trajectories`.

Run a short training smoke test:

```bash
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p "/path/to/ctbt-drone-trajectory/scripts/train.py" \
  --num_envs 32 --max_iterations 20
```

Run a normal training job after the smoke test succeeds:

```bash
./isaaclab.sh -p "/path/to/ctbt-drone-trajectory/scripts/train.py" \
  --num_envs 512 --max_iterations 4000
```

Run inference with a checkpoint:

```bash
./isaaclab.sh -p "/path/to/ctbt-drone-trajectory/scripts/play.py" \
  --checkpoint "/absolute/path/to/model_XXXX.pt" \
  --trajectory_bank "/path/to/ctbt-drone-trajectory/assets/trajectories/test.npz" \
  --trajectory_type figure8 \
  --episode_seconds 20 \
  --top_view \
  --episodes 1
```

Add `--video` to save the first episode as an MP4. The default output directory is `videos/inference`; use `--video_dir /path/to/output` to change it. Add `--video_length N` to cap the recording at `N` simulation steps.

```bash
./isaaclab.sh -p "/path/to/ctbt-drone-trajectory/scripts/play.py" \
  --checkpoint "/absolute/path/to/model_XXXX.pt" \
  --video --video_dir "/absolute/path/to/videos"
```

Inference follows the selected trajectory family for 20 seconds. Use `--trajectory_type circle|figure8|spline|straight|lissajous` or `--trajectory_index N` to select an exact bank entry. `--trajectory_index` takes precedence. If the bank entry is shorter than the requested duration, it is looped continuously. Playback is one 20-second episode by default; use `--episodes` for repeated runs.

Evaluate headlessly:

```bash
./isaaclab.sh -p "/path/to/ctbt-drone-trajectory/scripts/evaluate.py" \
  --checkpoint "/absolute/path/to/model_XXXX.pt" --num_envs 16
```

The actor action is always normalized CTBT `[collective thrust, roll torque, pitch torque, yaw torque]`. Rotor allocation happens inside the Isaac Lab environment. The critic receives additional simulator-truth observations; those are never passed to the actor.
