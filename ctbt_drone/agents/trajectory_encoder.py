"""RSL-RL adapter: CNN trajectory encoder followed by an MLP actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class TrajectoryMLPModel(CNNModel):
    """Actor model: flat current observation + CNN trajectory embedding."""

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        flat = MLPModel.get_latent(self, obs, masks, hidden_state)
        encoded = [self.cnns[group](obs[group]) for group in self.obs_groups_2d]
        return torch.cat([flat, *encoded], dim=-1)

