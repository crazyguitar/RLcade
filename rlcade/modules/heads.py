"""Heads — map latent features to action distributions, values, or Q-values."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """MLP head that outputs action logits."""

    def __init__(self, in_features: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """MLP head that outputs a scalar state value."""

    def __init__(self, in_features: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DuelingHead(nn.Module):
    """Dueling head: separate value and advantage streams, outputs Q-values."""

    def __init__(self, in_features: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


# NoisyNet + Distributional components for Rainbow DQN


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device | None = None) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        device = self.weight_epsilon.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DistributionalDuelingHead(nn.Module):
    """Distributional dueling head with NoisyNet layers for Rainbow DQN.

    Outputs either Q-values, log-probabilities, or raw distributions
    over the C51 support atoms.
    """

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        num_atoms: int = 51,
        v_min: float = -200.0,
        v_max: float = 200.0,
        noise_std: float = 0.5,
        hidden: int = 512,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_atoms = int(num_atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.value = nn.Sequential(
            NoisyLinear(in_features, hidden, noise_std),
            nn.ReLU(),
            NoisyLinear(hidden, num_atoms, noise_std),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(in_features, hidden, noise_std),
            nn.ReLU(),
            NoisyLinear(hidden, n_actions * num_atoms, noise_std),
        )

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        v = self.value(x).view(batch, 1, self.num_atoms)
        a = self.advantage(x).view(batch, self.n_actions, self.num_atoms)
        return v + a - a.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        """Return Q-values, or log-probs if log=True."""
        logits = self._logits(x)
        if log:
            return F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax distribution: (batch, n_actions, num_atoms)."""
        return F.softmax(self._logits(x), dim=-1)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
