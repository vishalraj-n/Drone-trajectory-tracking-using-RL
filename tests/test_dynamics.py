import torch

from ctbt_drone.dynamics import QuadrotorGeometry, project_ctbt_to_rotor_thrust, rotor_thrust_to_body_torque


def test_collective_is_symmetric() -> None:
    geometry = QuadrotorGeometry()
    allocation = geometry.allocation_matrix(torch.device("cpu"))
    thrust, saturation = project_ctbt_to_rotor_thrust(torch.tensor([[0.4, 0.0, 0.0, 0.0]]), allocation, 1.0)
    assert torch.allclose(thrust[0], torch.full((4,), 0.1), atol=1e-6)
    assert saturation.item() == 0.0


def test_equal_rotors_have_zero_body_torque() -> None:
    torque = rotor_thrust_to_body_torque(torch.ones(2, 4), QuadrotorGeometry())
    assert torch.allclose(torque, torch.zeros_like(torque), atol=1e-7)


def test_saturation_is_reported() -> None:
    geometry = QuadrotorGeometry()
    allocation = geometry.allocation_matrix(torch.device("cpu"))
    _, saturation = project_ctbt_to_rotor_thrust(torch.tensor([[2.0, 0.0, 0.0, 0.0]]), allocation, 0.1)
    assert saturation.item() > 0.0

