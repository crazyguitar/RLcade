from __future__ import annotations

import torch
import torch.nn as nn

from rlcade.modules.encoders import create_encoder
from rlcade.modules.heads import DuelingHead, DistributionalDuelingHead, NoisyLinear


class QNet(nn.Module):
    """Dueling Q-Network: encoder + dueling head."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, encoder: str = "cnn", **encoder_kwargs):
        super().__init__()
        self.encoder = create_encoder(encoder, obs_shape, **encoder_kwargs)
        self.head = DuelingHead(self.encoder.out_features, n_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class RainbowQNet(nn.Module):
    """Distributional Dueling Q-Network with NoisyNet: encoder + distributional dueling head."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        num_atoms: int = 51,
        v_min: float = -200.0,
        v_max: float = 200.0,
        noise_std: float = 0.5,
        encoder: str = "cnn",
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = create_encoder(encoder, obs_shape, **encoder_kwargs)
        self.head = DistributionalDuelingHead(
            self.encoder.out_features,
            n_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            noise_std=noise_std,
        )
        self.support = self.head.support

        # Xavier init for CNN only — NoisyLinear has its own init
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        return self.head(self.encoder(x), log=log)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.dist(self.encoder(x))

    def reset_noise(self):
        self.head.reset_noise()
