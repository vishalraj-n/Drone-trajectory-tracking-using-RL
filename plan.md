# Phase-One CTBT Quadrotor Trajectory-Control Project

## Objective

Train an Isaac Lab Crazyflie-class quadrotor to track physically feasible 3-D reference trajectories using PPO. The actor must output normalized CTBT commands:

`[collective thrust, body roll torque, body pitch torque, body yaw torque]`

The first phase uses nominal actuator-aware Crazyflie dynamics and broad trajectory variation, but does not yet use extensive dynamics randomization, motor-delay randomization, wind, payloads, or adaptive latent dynamics.

The first demonstration must show one trained policy following several trajectories, including a circle and a figure-eight.

## Environment and dynamics

- Use the existing Isaac Sim 5.1.0 and Isaac Lab installation described in `~/isaac/ISAAC_SETUP.md`.
- Never reinstall Isaac Sim or Isaac Lab and do not clone Isaac Lab into this project.
- Implement the task as an Isaac Lab `DirectRLEnv` with RSL-RL PPO.
- Use the official Isaac Lab Crazyflie asset.
- Use a 5 ms physics step and a 10 ms policy step, giving a 100 Hz control loop.
- Model the vehicle as a free-floating rigid body with nominal mass `m`, diagonal inertia `J`, quaternion attitude `q`, world linear velocity `v`, and body angular velocity `omega`. Apply the paper dynamics `p_dot=v`, `q_dot=0.5 q tensor omega`, `v_dot=-g e3 + (1/m) q tensor T_B tensor q*`, and `omega_dot=J^-1(tau - omega cross (J omega))` through Isaac PhysX body-wrench integration.
- Model each rotor explicitly as `T_i=K_F Omega_i^2` and `tau_i=K_M Omega_i^2`; use the CTBT allocator to obtain rotor thrust, recover nominal rotor speed, and compute body yaw torque from `K_M Omega_i^2`. Motor delay and lag remain deferred.
- Convert the four-dimensional normalized actor output into physical CTBT, then allocate CTBT to four nonnegative rotor thrusts using an explicit mixer and maximum-thrust saturation.
- Keep phase-one motor response nominal and deterministic. Do not randomize motor delay, motor lag, thrust curves, inertia, mass, arm length, wind, or payload.
- Document and test the FLU convention: x-forward, y-left, z-up. Document quaternion ordering, body angular-rate signs, CTBT torque signs, rotor ordering, and mixer geometry.

## Policy architecture

The policy is not the published RAPTOR motor-RPM policy. RAPTOR is used as inspiration for 100 Hz control, FLU observations, previous-action context, and later history-aware adaptation. Its public action is normalized per-motor RPM, which conflicts with the required CTBT interface. [RAPTOR repository](https://github.com/rl-tools/raptor)

### Actor

Use a feed-forward MLP actor whose inputs are:

- current proprioceptive/task observation;
- the learned trajectory embedding from the trajectory CNN;
- the previous normalized CTBT action.

The actor must not receive privileged dynamics information.

Recommended phase-one actor:

- current-observation MLP: 2–3 layers, 128 units, ELU or Tanh;
- trajectory encoder: three 1-D convolutional layers, kernel size 3, 32 channels, ReLU/ELU, followed by a linear layer to a 32-D embedding;
- concatenate current features and trajectory embedding;
- final MLP: two or three 128-unit layers;
- four-dimensional Gaussian policy head with bounded/squashed output in `[-1, 1]`.

The CNN should encode future reference position/velocity errors over a configurable horizon. Use 100 control samples by default, matching the reference implementation and the scale-aware paper; allow a shorter horizon for smoke tests.

### Critic

Use a separate MLP critic. It receives the actor-visible features plus privileged information through a separate critic observation group:

- current actor observation;
- the future trajectory window through the same CNN trajectory-encoder architecture used by the actor;
- privileged dynamics information unavailable to the actor: true mass, diagonal inertia, thrust-to-weight ratio, thrust coefficient, and wind xyz;
- true world-frame position/velocity and body-frame rates if these differ from the actor’s estimator-style observation.

The critic must never expose privileged values to the actor during inference. With fixed nominal dynamics, the privileged parameter subset is constant, but the true-state and reference-derivative information remains useful for value estimation and establishes the interface needed for phase two.

Configure RSL-RL observation groups so the actor and critic both receive their own CNN-encoded `trajectory` input. The actor receives `current` and `trajectory`; the critic receives `critic_current`, `trajectory`, and `privileged`. The two CNN instances have the same architecture but independent PPO-learned weights unless a custom shared actor-critic implementation is added later.

## Trajectory dataset and generation

Training must use many physically feasible trajectories rather than training only on one circle and one figure-eight. The adaptive-control repository confirms that its environment expects precomputed train/evaluation arrays and a 100-step trajectory window, but the tracked checkout does not include the expected training arrays. It does include the generator code and evaluation assets. [Adaptive-control repository](https://github.com/varadVaidya/adapt-drones)

Recreate the useful part locally instead of depending on that repository at runtime.

### Generator families

Implement `scripts/generate_trajectories.py` and `ctbt_drone/trajectories/generators.py` using analytic or minimum-snap generation for:

- hover and fixed waypoint;
- horizontal circles with varied radius, period, phase, and altitude;
- horizontal and 3-D lemniscates/figure-eights;
- random minimum-snap waypoint splines;
- random straight waypoint paths;
- random looped paths;
- Lissajous 3-D paths;
- Store each generated bank entry's family label in `family_ids` so inference can select `circle`, `figure8`, `spline`, `straight`, or `lissajous` without hard-coded indices. Inference defaults to one 20-second episode and loops a shorter bank entry if necessary; an exact `--trajectory_index` remains available.
- Training uses a fixed pool generated from the adaptive-drone reference distribution: five-waypoint, 16-second closed quintic splines with the reference waypoint and interior-velocity ranges. Every reset samples a pool entry and a valid random starting offset. The reference training arrays are not distributed in that repository, so the generator code is reproduced locally; fresh online generation remains an optional `--online_trajectories` ablation.
- optional octahedron and satellite-orbit paths for evaluation.

These families correspond to generators present in the cited repository, including `lemniscate_trajectory`, `random_trajectory`, `random_straight_trajectory`, `random_looped_trajectory`, `octahedron_trajectory`, `satellite_orbit`, and Lissajous generators.

Each generated sample must contain timestamp, position, velocity, acceleration, and jerk. The policy input uses position/velocity; acceleration and jerk are retained for feasibility filtering, privileged critic input, diagnostics, and future feed-forward extensions.

### Feasibility filtering

Before saving a trajectory, reject or resample it if it violates configured limits for:

- workspace bounds and minimum altitude;
- maximum speed;
- maximum acceleration;
- maximum jerk;
- minimum trajectory duration;
- discontinuities at segment joins;
- the Crazyflie’s available thrust margin.

The generator must print a summary of accepted/rejected trajectories and the resulting speed, acceleration, and jerk ranges.

### Dataset split

Generate deterministic, seedable datasets:

- `assets/trajectories/train.npz`: mixed trajectory families and parameter ranges;
- `assets/trajectories/validation.npz`: same families with held-out seeds and parameters;
- `assets/trajectories/test.npz`: held-out combinations and optional families such as Lissajous, octahedron, and satellite orbit.

During training, each parallel environment samples a trajectory and a random starting index/window. Do not expose the test trajectory bank to training. The evaluation script must support both saved banks and on-the-fly seeded generation.

The first training mixture should emphasize smooth circles, figure-eights, and minimum-snap splines. Increase the fraction of complex 3-D paths only after hover and smooth tracking are stable. This is trajectory/task diversity, not dynamics domain randomization.

## Observations and rewards

The actor observation should contain normalized:

- position error to the current reference;
- velocity error to the current reference;
- attitude representation;
- body angular velocity;
- previous CTBT action;
- the future reference position/velocity window supplied to the CNN.

Use the scale-aware paper’s weighted `dm_control` Gaussian-tolerance reward. For each nonnegative error `e` and margin `m`, define `G(e,m)=0.1^((e/m)^2)`, so the term is 1 at zero error and 0.1 at the margin, then compute:

`r = 0.50 G(r_pos,0.75) + 0.1625 G(r_close-pos,0.05) + 0.20 G(r_vel,0.50) + 0.0625 G(r_omega,0.50) + 0.0375 G(r_smooth,0.50) + 0.0375 G(r_yaw,0.50) + r_crash`

where `r_crash=-100` on crash and zero otherwise. The reward has no full attitude-error term. Attitude remains part of the physical state and actor observation because thrust direction depends on attitude; required roll/pitch should emerge from position and velocity tracking. Tilt is still a termination safety condition.

Log every reward component separately. Terminate on ground contact, excessive tilt, workspace violation, numerical instability, or timeout.

## Project structure and interfaces

Create these modules:

- `ctbt_drone/envs/ctbt_trajectory_env.py`: Isaac Lab environment, reset, observations, rewards, termination, CTBT application, and trajectory sampling.
- `ctbt_drone/dynamics.py`: rotor geometry, CTBT allocation, thrust/torque conversion, saturation, and frame checks.
- `ctbt_drone/trajectories/generators.py`: analytic and minimum-snap trajectory generators.
- `ctbt_drone/trajectories/dataset.py`: train/validation/test bank loading, window sampling, and normalization.
- `ctbt_drone/agents/trajectory_encoder.py`: CNN trajectory encoder with configurable horizon and 32-D output.
- `ctbt_drone/agents/ppo_cfg.py`: actor MLP, trajectory encoder wiring, privileged critic groups, and PPO settings.
- `scripts/generate_trajectories.py`: deterministic bank generation and feasibility filtering.
- `scripts/train.py`: shared Isaac Lab launch, PPO training, checkpoints, and TensorBoard logs.
- `scripts/play.py`: visual checkpoint playback with selectable trajectory family, seed, radius, period, and target options.
- `scripts/evaluate.py`: headless multi-seed evaluation and telemetry export.

The public action remains exactly four-dimensional CTBT. Rotor conversion is internal to the environment and must not become the policy’s public action interface.

## Validation and acceptance

1. Run syntax and import checks without launching training.
2. Unit-test mixer collective, roll, pitch, yaw, saturation, and sign conventions.
3. Unit-test every trajectory generator, derivative continuity, feasibility filtering, and deterministic seeding.
4. Run an Isaac Lab smoke test with 16–32 environments.
5. Verify open-loop nominal hover using a known hover CTBT command.
6. Train a hover/waypoint baseline.
7. Train the mixed-trajectory policy.
8. Evaluate held-out circle, figure-eight, spline, and 3-D trajectories over multiple seeds.
9. Produce telemetry and plots for position RMSE, maximum error, velocity RMSE, overshoot, settling time, final error, crash rate, CTBT saturation, and attitude/rate behavior.
10. Demonstrate the checkpoint interactively in Isaac Sim.

The phase-one result is accepted only when one checkpoint can reliably hover and follow multiple circle and figure-eight variants, with repeatable metrics and no unexplained mixer or frame errors.

## Deferred phase two

After the nominal mixed-trajectory demo works, add motor delay and first-order motor lag, randomized mass/inertia/arm length/thrust curves, wind, payloads, scale-aware dynamics randomization, latent dynamics encoding, history-based adaptation, and eventually RAPTOR-style recurrent teacher/student distillation. The scale-aware work provides the trajectory-encoder and CTBT design direction, while RAPTOR provides a later foundation-policy/meta-learning direction.

## Assumptions

- RAPTOR means RAPTOR-inspired architecture and timing, not direct use of its RPM-output checkpoint.
- The actor is feed-forward MLP plus CNN trajectory embedding, as specified above.
- The critic is a separate MLP with privileged simulator information.
- Broad trajectory diversity is required in phase one; broad dynamics randomization is not.
- The cited adaptive-control training arrays are not assumed to be available, so they will be regenerated locally from its generator patterns.
- All commands and documentation assume the existing environment at `~/isaac/env_isaaclab` and Isaac Lab at `~/isaac/IsaacLab`.
