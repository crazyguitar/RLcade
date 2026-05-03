from __future__ import annotations

import torch
import torch.nn as nn

from rlcade.modules.encoders import create_encoder
from rlcade.modules.heads import PolicyHead


class Actor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, encoder: str = "cnn", **encoder_kwargs):
        super().__init__()
        self.encoder = create_encoder(encoder, obs_shape, **encoder_kwargs)
        self.head = PolicyHead(self.encoder.out_features, n_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy logits. Callers wrap in ``Categorical(logits=...)`` as needed.

        Returning a raw tensor lets CUDAGraphWrapper capture the forward pass --
        distribution objects aren't graph-replayable.
        """
        return self.head(self.encoder(x))
