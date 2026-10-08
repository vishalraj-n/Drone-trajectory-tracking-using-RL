"""Physically bounded analytic and minimum-snap-like trajectory generators."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _derivatives(position: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.gradient(position, dt, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=1)
    return np.concatenate((position, velocity, acceleration, jerk), axis=-1).astype(np.float32)


def circle(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) * dt
    radius = rng.uniform(0.20, 0.65)
    period = rng.uniform(5.0, 10.0)
    phase = rng.uniform(-np.pi, np.pi)
    center = np.array((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(0.65, 1.15)))
    w = 2.0 * np.pi / period
    angle = w * t + phase
    position = center + np.stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(t)), axis=-1)
    return _derivatives(position, dt)


def figure_eight(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) * dt
    radius = rng.uniform(0.20, 0.60)
    period = rng.uniform(6.0, 11.0)
    phase = rng.uniform(-np.pi, np.pi)
    center = np.array((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(0.65, 1.15)))
    angle = 2.0 * np.pi * t / period + phase
    position = center + np.stack((radius * np.sin(angle), 0.5 * radius * np.sin(2.0 * angle), np.zeros_like(t)), axis=-1)
    return _derivatives(position, dt)


def closed_circle(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    """Generate one exactly closed circle over the available sample window."""
    period = max((length - 1) * dt, dt)
    t = np.linspace(0.0, period, length, dtype=np.float32)
    radius = rng.uniform(0.20, 0.65)
    phase = rng.uniform(-np.pi, np.pi)
    center = np.array((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(0.65, 1.15)))
    angle = 2.0 * np.pi * t / period + phase
    position = center + np.stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(t)), axis=-1)
    return _derivatives(position, dt)


def closed_figure_eight(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    """Generate one exactly closed figure-eight over the sample window."""
    period = max((length - 1) * dt, dt)
    t = np.linspace(0.0, period, length, dtype=np.float32)
    radius = rng.uniform(0.20, 0.60)
    phase = rng.uniform(-np.pi, np.pi)
    center = np.array((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(0.65, 1.15)))
    angle = 2.0 * np.pi * t / period + phase
    position = center + np.stack((radius * np.sin(angle), 0.5 * radius * np.sin(2.0 * angle), np.zeros_like(t)), axis=-1)
    return _derivatives(position, dt)


def lissajous(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) * dt
    center = np.array((rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0.75, 1.15)))
    amplitudes = rng.uniform((0.20, 0.15, 0.05), (0.55, 0.45, 0.20))
    # Keep the highest third-harmonic component within the phase-one jerk
    # bound; faster variants are reserved for later dynamics randomization.
    frequency = rng.uniform(0.15, 0.35)
    phase = rng.uniform(-np.pi, np.pi, 3)
    position = center + amplitudes * np.stack(
        (np.sin(2.0 * np.pi * frequency * t + phase[0]),
         np.sin(3.0 * np.pi * frequency * t + phase[1]),
         np.sin(1.5 * np.pi * frequency * t + phase[2])), axis=-1
    )
    return _derivatives(position, dt)


def quintic_waypoints(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    """Generate a closed, smooth waypoint trajectory with zero endpoint rates."""

    n_points = 5
    points = np.empty((n_points, 3), dtype=np.float32)
    points[:, :2] = rng.uniform(-0.75, 0.75, (n_points, 2))
    points[:, 2] = rng.uniform(0.65, 1.25, n_points)
    points[-1] = points[0]
    segment_count = n_points - 1
    samples_per_segment = length // segment_count
    chunks = []
    for i in range(segment_count):
        count = samples_per_segment if i < segment_count - 1 else length - samples_per_segment * (segment_count - 1)
        s = np.linspace(0.0, 1.0, count, endpoint=i == segment_count - 1, dtype=np.float32)
        smooth = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        chunks.append(points[i] + (points[i + 1] - points[i]) * smooth[:, None])
    return _derivatives(np.concatenate(chunks, axis=0), dt)


def random_waypoint_spline(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    """Generate one random closed quintic spline for online training.

    This follows the adaptive-drone reference distribution: five equally
    spaced waypoints, random position and interior velocity constraints, and
    zero acceleration constraints.  Unlike the named evaluation families,
    every reset produces a new smooth path.
    """
    num_points = 5
    total_time = max(length * dt, 1.0)
    waypoint_times = np.linspace(0.0, total_time, num_points, dtype=np.float64)
    waypoints = np.empty((num_points, 3), dtype=np.float64)
    # These ranges match adapt-drones' random_trajectory() generator.
    waypoints[:, :2] = rng.uniform(-1.5, 1.5, (num_points, 2))
    waypoints[:, 2] = rng.uniform(0.5, 1.0, num_points)
    waypoints[-1] = waypoints[0]

    velocities = np.zeros((num_points, 3), dtype=np.float64)
    velocities[1:-1, :2] = rng.uniform(-1.0, 1.0, (num_points - 2, 2))
    velocities[1:-1, 2] = rng.uniform(-0.75, 0.75, num_points - 2)
    accelerations = np.zeros_like(velocities)

    samples = np.arange(length, dtype=np.float64) * dt
    position = np.empty((length, 3), dtype=np.float64)
    velocity = np.empty_like(position)
    acceleration = np.empty_like(position)
    jerk = np.empty_like(position)
    for segment in range(num_points - 1):
        start_t, end_t = waypoint_times[segment : segment + 2]
        duration = end_t - start_t
        mask = (samples >= start_t) & ((samples < end_t) if segment < num_points - 2 else (samples <= end_t))
        tau = samples[mask] - start_t
        # Polynomial coefficients are ordered from t^5 to t^0.
        boundary = np.stack(
            (waypoints[segment], velocities[segment], accelerations[segment],
             waypoints[segment + 1], velocities[segment + 1], accelerations[segment + 1]),
            axis=0,
        )
        matrix = np.array(
            [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0],
             [duration**5, duration**4, duration**3, duration**2, duration, 1],
             [5 * duration**4, 4 * duration**3, 3 * duration**2, 2 * duration, 1, 0],
             [20 * duration**3, 12 * duration**2, 6 * duration, 2, 0, 0]],
            dtype=np.float64,
        )
        coefficients = np.linalg.solve(matrix, boundary)
        # Explicit derivatives avoid finite-difference spikes at segment joins.
        position[mask] = np.stack([np.polyval(coefficients[:, axis], tau) for axis in range(3)], axis=-1)
        velocity[mask] = np.stack([np.polyval(np.polyder(coefficients[:, axis]), tau) for axis in range(3)], axis=-1)
        acceleration[mask] = np.stack([np.polyval(np.polyder(coefficients[:, axis], 2), tau) for axis in range(3)], axis=-1)
        jerk[mask] = np.stack([np.polyval(np.polyder(coefficients[:, axis], 3), tau) for axis in range(3)], axis=-1)
    return np.concatenate((position, velocity, acceleration, jerk), axis=-1).astype(np.float32)


def straight(rng: np.random.Generator, length: int, dt: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    start = np.array((rng.uniform(-0.7, 0.2), rng.uniform(-0.7, 0.2), rng.uniform(0.7, 1.0)))
    end = np.array((rng.uniform(-0.2, 0.7), rng.uniform(-0.2, 0.7), rng.uniform(0.7, 1.3)))
    smooth = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return _derivatives(start + (end - start) * smooth[:, None], dt)


GENERATORS: dict[str, Callable[[np.random.Generator, int, float], np.ndarray]] = {
    "adaptive": random_waypoint_spline,
    "circle": circle,
    "figure8": figure_eight,
    "lissajous": lissajous,
    "spline": quintic_waypoints,
    "straight": straight,
}

# Used only for playback.  The training bank retains the original diverse
# trajectory distribution; playback needs a mathematically periodic reference
# so the first and last samples join without a target jump.
CLOSED_GENERATORS: dict[str, Callable[[np.random.Generator, int, float], np.ndarray]] = {
    "circle": closed_circle,
    "figure8": closed_figure_eight,
    "spline": quintic_waypoints,
}


def feasible(trajectory: np.ndarray, dt: float, max_speed: float, max_acceleration: float, max_jerk: float) -> bool:
    position = trajectory[:, :3]
    speed = np.linalg.norm(trajectory[:, 3:6], axis=-1).max()
    acceleration = np.linalg.norm(trajectory[:, 6:9], axis=-1).max()
    jerk = np.linalg.norm(trajectory[:, 9:12], axis=-1).max()
    return bool(
        position[:, 2].min() > 0.20
        and position[:, 2].max() < 2.0
        and np.abs(position[:, :2]).max() < 1.5
        and speed <= max_speed
        and acceleration <= max_acceleration
        and jerk <= max_jerk
        and np.isfinite(trajectory).all()
    )


def generate_trajectory_bank(
    seed: int,
    count: int,
    length: int = 1200,
    dt: float = 0.01,
    families: tuple[str, ...] = tuple(GENERATORS),
    max_speed: float = 1.5,
    max_acceleration: float = 3.5,
    max_jerk: float = 12.0,
) -> np.ndarray:
    bank, _ = generate_labeled_trajectory_bank(
        seed=seed,
        count=count,
        length=length,
        dt=dt,
        families=families,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )
    return bank


def generate_labeled_trajectory_bank(
    seed: int,
    count: int,
    length: int = 1200,
    dt: float = 0.01,
    families: tuple[str, ...] = tuple(GENERATORS),
    max_speed: float = 1.5,
    max_acceleration: float = 3.5,
    max_jerk: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a bank and retain the accepted generator family per entry."""
    rng = np.random.default_rng(seed)
    bank: list[np.ndarray] = []
    labels: list[str] = []
    attempts = 0
    while len(bank) < count and attempts < count * 100:
        attempts += 1
        family = families[int(rng.integers(0, len(families)))]
        trajectory = GENERATORS[family](rng, length, dt)
        if feasible(trajectory, dt, max_speed, max_acceleration, max_jerk):
            bank.append(trajectory.astype(np.float32))
            labels.append(family)
    if len(bank) != count:
        raise RuntimeError(f"Accepted {len(bank)} of {count} trajectories after {attempts} attempts")
    return np.stack(bank, axis=0), np.asarray(labels)
