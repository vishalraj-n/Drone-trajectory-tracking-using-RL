"""Nominal quadrotor CTBT allocation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuadrotorGeometry:
    """X-configuration geometry in an FLU body frame.

    Rotor order is front-left, rear-left, rear-right, front-right. Verify the
    ordering against any hardware firmware before deployment.
    """

    arm_length: float = 0.046
    thrust_coefficient: float = 1.30e-7
    torque_coefficient: float = 1.56e-9
    spin_signs: tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)

    @property
    def yaw_torque_ratio(self) -> float:
        return self.torque_coefficient / self.thrust_coefficient

    def allocation_matrix(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        a = self.arm_length / 2.0**0.5
        positions = torch.tensor(((a, a), (-a, a), (-a, -a), (a, -a)), device=device, dtype=dtype)
        matrix = torch.zeros((4, 4), device=device, dtype=dtype)
        matrix[0] = 1.0
        matrix[1] = positions[:, 1]
        matrix[2] = -positions[:, 0]
        matrix[3] = torch.tensor(self.spin_signs, device=device, dtype=dtype) * self.yaw_torque_ratio
        return matrix


def project_ctbt_to_rotor_thrust(
    ctbt: torch.Tensor, allocation: torch.Tensor, max_thrust: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map [collective, roll, pitch, yaw] to clipped rotor thrusts."""

    desired = ctbt @ torch.linalg.pinv(allocation).T
    maximum = torch.as_tensor(max_thrust, device=ctbt.device, dtype=ctbt.dtype)
    clipped = torch.minimum(desired, maximum)
    actual = torch.clamp(clipped, min=0.0)
    saturation = (actual - desired).abs().sum(dim=-1)
    return actual, saturation


def rotor_thrust_to_body_torque(thrust: torch.Tensor, geometry: QuadrotorGeometry) -> torch.Tensor:
    """Return torque using T_i=K_F*Omega_i^2 and tau_i=K_M*Omega_i^2."""

    a = geometry.arm_length / 2.0**0.5
    roll = a * (thrust[..., 0] + thrust[..., 1] - thrust[..., 2] - thrust[..., 3])
    pitch = a * (-thrust[..., 0] + thrust[..., 1] + thrust[..., 2] - thrust[..., 3])
    spin = torch.tensor(geometry.spin_signs, device=thrust.device, dtype=thrust.dtype)
    omega_squared = thrust / geometry.thrust_coefficient
    yaw = geometry.torque_coefficient * (omega_squared * spin).sum(dim=-1)
    return torch.stack((roll, pitch, yaw), dim=-1)
