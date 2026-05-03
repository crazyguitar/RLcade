"""Replay buffers for off-policy learning."""

from __future__ import annotations

import operator

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular replay buffer. Stores on CPU, samples to target device."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.pos = 0
        self.size = 0

        self.obs = torch.zeros(capacity, *obs_shape)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, *obs_shape)
        self.dones = torch.zeros(capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos].copy_(next_obs)
        self.dones[self.pos] = float(bool(done))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            "obs": self.obs[idx].to(self.device),
            "actions": self.actions[idx].to(self.device),
            "rewards": self.rewards[idx].to(self.device),
            "next_obs": self.next_obs[idx].to(self.device),
            "dones": self.dones[idx].to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class SegmentTree:
    """Segment tree for efficient range queries."""

    def __init__(self, capacity: int, operation, init_value: float):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0
        self.capacity = capacity
        self.tree = [init_value] * (2 * capacity)
        self.operation = operation

    def _query(self, start: int, end: int, node: int, ns: int, ne: int) -> float:
        if start == ns and end == ne:
            return self.tree[node]
        mid = (ns + ne) // 2
        if end <= mid:
            return self._query(start, end, 2 * node, ns, mid)
        if start > mid:
            return self._query(start, end, 2 * node + 1, mid + 1, ne)
        return self.operation(
            self._query(start, mid, 2 * node, ns, mid),
            self._query(mid + 1, end, 2 * node + 1, mid + 1, ne),
        )

    def query(self, start: int = 0, end: int | None = None) -> float:
        end = (end or self.capacity) - 1
        return self._query(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


class SumTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operator.add, 0.0)

    def sum(self) -> float:
        return self.query()

    def retrieve(self, value: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, min, float("inf"))

    def min(self) -> float:
        return self.query()


class PrioritizedReplayBuffer:
    """PER buffer using sum/min segment trees for O(log n) sampling."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        alpha: float = 0.6,
        device: str = "cpu",
    ):
        # Round up to power of 2 for segment tree
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity <<= 1

        self.capacity = capacity
        self.alpha = alpha
        self.device = torch.device(device)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumTree(self.tree_capacity)
        self.min_tree = MinTree(self.tree_capacity)

        self.obs = torch.zeros(capacity, *obs_shape)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, *obs_shape)
        self.dones = torch.zeros(capacity)

    def add(self, obs, action, reward, next_obs, done):
        priority = self.max_priority**self.alpha
        self.sum_tree[self.pos] = priority
        self.min_tree[self.pos] = priority

        self.obs[self.pos].copy_(obs)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos].copy_(next_obs)
        self.dones[self.pos] = float(bool(done))

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, torch.Tensor | np.ndarray]:
        indices = self._sample_proportional(batch_size)
        weights = self._compute_weights(indices, beta)
        return {
            "obs": self.obs[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "rewards": self.rewards[indices].to(self.device),
            "next_obs": self.next_obs[indices].to(self.device),
            "dones": self.dones[indices].to(self.device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            "indices": indices,
        }

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        indices = np.empty(batch_size, dtype=np.int64)
        total = self.sum_tree.sum()
        segment = total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx = self.sum_tree.retrieve(value)
            indices[i] = min(idx, self.size - 1)
        return indices

    def _compute_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        min_prob = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (self.size * min_prob) ** (-beta)
        total = self.sum_tree.sum()

        weights = np.empty(len(indices), dtype=np.float32)
        for i, idx in enumerate(indices):
            prob = self.sum_tree[idx] / total
            weights[i] = (self.size * prob) ** (-beta) / max_weight
        return weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            p = priority**self.alpha
            self.sum_tree[idx] = p
            self.min_tree[idx] = p
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.size
