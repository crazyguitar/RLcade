import torch

from rlcade.utils.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), device="cpu")
        assert len(buf) == 0
        buf.add(torch.zeros(2), 0, 1.0, torch.zeros(2), False)
        assert len(buf) == 1

    def test_circular_wrap(self):
        buf = ReplayBuffer(capacity=4, obs_shape=(2,), device="cpu")
        for i in range(6):
            buf.add(torch.tensor([float(i), 0.0]), i % 3, float(i), torch.zeros(2), False)
        assert len(buf) == 4
        assert buf.pos == 2  # 6 % 4

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=16, obs_shape=(4, 3, 3), device="cpu")
        for i in range(16):
            buf.add(torch.randn(4, 3, 3), i % 5, 1.0, torch.randn(4, 3, 3), False)
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, 4, 3, 3)
        assert batch["next_obs"].shape == (4, 4, 3, 3)
        assert batch["actions"].shape == (4,)
        assert batch["rewards"].shape == (4,)
        assert batch["dones"].shape == (4,)

    def test_sample_device(self):
        buf = ReplayBuffer(capacity=8, obs_shape=(2,), device="cpu")
        for _ in range(8):
            buf.add(torch.zeros(2), 0, 0.0, torch.zeros(2), False)
        batch = buf.sample(2)
        assert batch["obs"].device == torch.device("cpu")

    def test_done_stored_as_float(self):
        buf = ReplayBuffer(capacity=4, obs_shape=(2,), device="cpu")
        buf.add(torch.zeros(2), 0, 0.0, torch.zeros(2), True)
        buf.add(torch.zeros(2), 0, 0.0, torch.zeros(2), False)
        assert buf.dones[0] == 1.0
        assert buf.dones[1] == 0.0


import numpy as np

from rlcade.utils.replay_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    def test_add_and_size(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        assert len(buf) == 0
        buf.add(torch.zeros(2), 0, 1.0, torch.zeros(2), False)
        assert len(buf) == 1

    def test_sample_shape(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        for i in range(16):
            buf.add(torch.randn(2), i % 3, 1.0, torch.randn(2), False)
        batch = buf.sample(4, beta=0.4)
        assert batch["obs"].shape == (4, 2)
        assert batch["weights"].shape == (4,)
        assert batch["indices"].shape == (4,)

    def test_weights_are_normalized(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        for _ in range(16):
            buf.add(torch.randn(2), 0, 1.0, torch.randn(2), False)
        batch = buf.sample(8, beta=0.4)
        assert batch["weights"].max().item() <= 1.0 + 1e-6

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        for _ in range(16):
            buf.add(torch.randn(2), 0, 1.0, torch.randn(2), False)
        batch = buf.sample(4, beta=0.4)
        new_priorities = np.array([100.0, 0.01, 0.01, 0.01])
        buf.update_priorities(batch["indices"], new_priorities)
        assert buf.max_priority >= 100.0

    def test_high_priority_sampled_more(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        for i in range(16):
            buf.add(torch.full((2,), float(i)), 0, 1.0, torch.zeros(2), False)
        buf.update_priorities(np.array([0]), np.array([1000.0]))
        counts = np.zeros(16)
        for _ in range(200):
            batch = buf.sample(4, beta=0.4)
            for idx in batch["indices"]:
                counts[idx] += 1
        assert counts[0] > counts[1:][:].mean() * 2

    def test_circular_overwrite(self):
        buf = PrioritizedReplayBuffer(capacity=4, obs_shape=(2,), alpha=0.6)
        for i in range(6):
            buf.add(torch.tensor([float(i), 0.0]), 0, 1.0, torch.zeros(2), False)
        assert len(buf) == 4
        assert buf.pos == 2

    def test_beta_1_gives_uniform_weights(self):
        buf = PrioritizedReplayBuffer(capacity=16, obs_shape=(2,), alpha=0.6)
        for _ in range(16):
            buf.add(torch.randn(2), 0, 1.0, torch.randn(2), False)
        batch = buf.sample(8, beta=1.0)
        assert torch.allclose(batch["weights"], torch.ones(8), atol=1e-5)
