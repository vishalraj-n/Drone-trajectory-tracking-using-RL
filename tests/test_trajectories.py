import numpy as np

from ctbt_drone.trajectories.generators import generate_trajectory_bank


def test_bank_is_deterministic_and_finite() -> None:
    first = generate_trajectory_bank(7, 4, length=240, families=("circle", "figure8"))
    second = generate_trajectory_bank(7, 4, length=240, families=("circle", "figure8"))
    assert np.array_equal(first, second)
    assert first.shape == (4, 240, 12)
    assert np.isfinite(first).all()

