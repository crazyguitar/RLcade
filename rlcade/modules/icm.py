"""Intrinsic Curiosity Module (Pathak et al., 2017).

Provides intrinsic reward based on prediction error in a learned feature space.
The agent is rewarded for encountering states it cannot predict, encouraging
exploration of novel areas.

Paper: https://arxiv.org/abs/1705.05363
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcade.modules.encoders import CNNEncoder


class ICM(nn.Module):
    """Intrinsic Curiosity Module.

    Components:
    - Feature encoder: maps observations to a compact feature space
    - Forward model: predicts next features from (features, action)
    - Inverse model: predicts action from (features, next_features)

    Intrinsic reward = ||predicted_next_features - actual_next_features||^2
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, feature_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        self.encoder = CNNEncoder(obs_shape)
        self.feature_proj = nn.Linear(self.encoder.out_features, feature_dim)

        # Forward model: (features + one_hot_action) -> predicted next features
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Inverse model: (features + next_features) -> predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_proj(self.encoder(obs))

    def forward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward, forward loss, and inverse loss.

        Returns:
            intrinsic_reward: (batch,) prediction error as curiosity signal
            forward_loss: scalar MSE loss for forward model
            inverse_loss: scalar cross-entropy loss for inverse model
        """
        features = self.encode(obs)
        next_features = self.encode(next_obs)

        # Forward model
        action_onehot = F.one_hot(actions.long(), self.n_actions).float()
        predicted_next = self.forward_model(torch.cat([features, action_onehot], dim=-1))
        forward_loss = F.mse_loss(predicted_next, next_features.detach(), reduction="mean")

        # Intrinsic reward = per-sample prediction error
        intrinsic_reward = (predicted_next - next_features.detach()).pow(2).mean(dim=-1)

        # Inverse model
        predicted_action_logits = self.inverse_model(torch.cat([features, next_features], dim=-1))
        inverse_loss = F.cross_entropy(predicted_action_logits, actions.long())

        return intrinsic_reward, forward_loss, inverse_loss
