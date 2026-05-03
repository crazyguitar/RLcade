from __future__ import annotations

import torch
import torch.nn as nn

from rlcade.modules.encoders import create_encoder
from rlcade.modules.heads import ValueHead


class Critic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], encoder: str = "cnn", **encoder_kwargs):
        super().__init__()
        self.encoder = create_encoder(encoder, obs_shape, **encoder_kwargs)
        self.head = ValueHead(self.encoder.out_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
