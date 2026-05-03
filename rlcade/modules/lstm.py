from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rlcade.modules.encoders import CNNEncoder


class LstmActorCritic(nn.Module):
    """Shared CNN -> LSTMCell -> policy/value heads (A3C-style)."""

    def __init__(self, obs_shape, n_actions, lstm_hidden=256):
        super().__init__()
        self.cnn = CNNEncoder(obs_shape)
        self.lstm = nn.LSTMCell(self.cnn.out_features, lstm_hidden)
        self.policy = nn.Linear(lstm_hidden, n_actions)
        self.value = nn.Linear(lstm_hidden, 1)
        self.lstm_hidden = lstm_hidden
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTMCell):
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)
                # Set forget gate bias to 1 (Jozefowicz et al., 2015)
                nn.init.constant_(m.bias_ih[self.lstm_hidden : 2 * self.lstm_hidden], 1)
                nn.init.constant_(m.bias_hh[self.lstm_hidden : 2 * self.lstm_hidden], 1)

    def forward(self, obs, hx_cx):
        features = self.cnn(obs)
        hx, cx = self.lstm(features, hx_cx)
        dist = Categorical(logits=self.policy(hx))
        return dist, self.value(hx).squeeze(-1), hx, cx

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.lstm_hidden, device=device)
        return h, torch.zeros_like(h)
