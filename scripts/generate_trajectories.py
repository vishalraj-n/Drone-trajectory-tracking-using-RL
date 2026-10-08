"""Generate train/validation/test trajectory banks without Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ctbt_drone.trajectories.generators import GENERATORS, generate_labeled_trajectory_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/trajectories"))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--length", type=int, default=1600, help="Samples per trajectory; 1600 is 16 seconds at 100 Hz, matching adapt-drones.")
    parser.add_argument(
        "--train_family",
        choices=("adaptive", "mixed"),
        default="adaptive",
        help="Training pool distribution. adaptive reproduces the reference random-waypoint distribution.",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    families = tuple(name for name in GENERATORS if name != "adaptive")
    train_families = ("adaptive",) if args.train_family == "adaptive" else families
    for name, count, seed, selected in (
        ("train", args.count, args.seed, train_families),
        ("validation", max(32, args.count // 4), args.seed + 1, families),
        ("test", max(64, args.count // 2), args.seed + 2, families),
    ):
        bank, family_ids = generate_labeled_trajectory_bank(seed, count, args.length, args.dt, selected)
        np.savez_compressed(
            args.output / f"{name}.npz",
            trajectories=bank,
            dt=args.dt,
            families=np.asarray(selected),
            family_ids=family_ids,
        )
        speed = np.linalg.norm(bank[:, :, 3:6], axis=-1).max()
        acceleration = np.linalg.norm(bank[:, :, 6:9], axis=-1).max()
        jerk = np.linalg.norm(bank[:, :, 9:12], axis=-1).max()
        print(f"{name}: {bank.shape}, max speed={speed:.3f}, max acceleration={acceleration:.3f}, max jerk={jerk:.3f}")


if __name__ == "__main__":
    main()
