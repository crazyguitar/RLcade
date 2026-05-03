"""Encoders — feature extractors that map observations to a latent vector."""

from __future__ import annotations

import torch
import torch.nn as nn

# IMPALA ResNet building blocks (Espeholt et al., 2018)
# https://arxiv.org/abs/1802.01561
#
# All convolutions use 3x3 kernels — the smallest kernel that captures
# spatial adjacency in every direction. Two stacked 3x3 convs cover
# the same 5x5 receptive field as a single 5x5 conv but with fewer
# parameters and an extra non-linearity (Simonyan & Zisserman, 2014).


class ResidualBlock(nn.Module):
    """ReLU -> Conv3x3 -> ReLU -> Conv3x3 + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvSequence(nn.Module):
    """Conv2d(3x3) -> MaxPool2d(3x3, stride=2) -> 2x ResidualBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ResNetEncoder(nn.Module):
    """IMPALA-style ResNet encoder. Maps (C, H, W) images to a flat feature vector."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        channels: list[int] | None = None,
        out_dim: int = 256,
    ):
        super().__init__()
        c, h, w = obs_shape
        if channels is None:
            channels = [16, 32, 32]

        stages = []
        in_ch = c
        for out_ch in channels:
            stages.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            flat_dim = self.flatten(self.relu(self.stages(torch.zeros(1, c, h, w)))).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)
        self.out_features = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = self.relu(x)
        x = self.flatten(x)
        return torch.relu(self.fc(x))


class CNNEncoder(nn.Module):
    """4-layer strided CNN encoder. Maps (C, H, W) images to a flat feature vector."""

    def __init__(self, obs_shape: tuple[int, ...], channels: int = 32):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_features = self.net(torch.zeros(1, c, h, w)).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMEncoder(nn.Module):
    """CNN feature extractor followed by an LSTM for temporal modeling.

    Maintains hidden state across calls. Call reset_hidden() on episode boundaries.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        channels: int = 32,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.cnn = CNNEncoder(obs_shape, channels)
        self.lstm = nn.LSTM(self.cnn.out_features, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.out_features = lstm_hidden
        self._hidden = None

    def reset_hidden(self, batch_size: int = 1, device: torch.device | None = None):
        """Reset LSTM hidden state. Call at episode start."""
        device = device or next(self.parameters()).device
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        self._hidden = (h, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        # Add time dimension: (B, F) -> (B, 1, F)
        if self._hidden is None:
            self.reset_hidden(features.shape[0], features.device)
        elif self._hidden[0].shape[1] != features.shape[0]:
            self.reset_hidden(features.shape[0], features.device)
        out, self._hidden = self.lstm(features.unsqueeze(1), self._hidden)
        return out.squeeze(1)

    def detach_hidden(self):
        """Detach hidden state from graph (call between rollout steps to avoid BPTT across rollouts)."""
        if self._hidden is not None:
            self._hidden = (self._hidden[0].detach(), self._hidden[1].detach())


_ENCODERS = {
    "cnn": CNNEncoder,
    "lstm": LSTMEncoder,
    "resnet": ResNetEncoder,
}


def create_encoder(name: str, obs_shape: tuple[int, ...], **kwargs) -> nn.Module:
    """Create an encoder by name. Raises KeyError if unknown."""
    return _ENCODERS[name](obs_shape, **kwargs)


def parse_channels(val) -> list[int] | None:
    """Parse channel list from CLI string (e.g. '16,32,32') or passthrough list."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return [int(x) for x in val.split(",")]


def build_encoder_kwargs(config) -> dict:
    """Extract encoder-specific kwargs from an agent config."""
    kw = {"encoder": config.encoder}
    if config.encoder == "resnet":
        kw["channels"] = getattr(config, "resnet_channels", [16, 32, 32])
        kw["out_dim"] = getattr(config, "resnet_out_dim", 256)
    return kw
