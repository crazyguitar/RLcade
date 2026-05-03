import torch

from rlcade.modules.qnet import QNet, RainbowQNet
from rlcade.modules.heads import NoisyLinear


class TestQNet:
    def test_output_shape(self):
        net = QNet(obs_shape=(4, 84, 84), n_actions=12)
        x = torch.randn(2, 4, 84, 84)
        out = net(x)
        assert out.shape == (2, 12)

    def test_dueling_mean_subtraction(self):
        net = QNet(obs_shape=(4, 84, 84), n_actions=6)
        x = torch.randn(3, 4, 84, 84)
        with torch.no_grad():
            features = net.encoder(x)
            value = net.head.value(features)
            advantage = net.head.advantage(features)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            out = net(x)
        torch.testing.assert_close(q, out)


class TestRainbowQNet:
    def test_forward_q_values(self):
        net = RainbowQNet((4, 15, 16), 6, num_atoms=11)
        q = net(torch.randn(2, 4, 15, 16))
        assert q.shape == (2, 6)

    def test_forward_log_probs(self):
        net = RainbowQNet((4, 15, 16), 6, num_atoms=11)
        log_p = net(torch.randn(2, 4, 15, 16), log=True)
        assert log_p.shape == (2, 6, 11)
        assert (log_p <= 0).all()

    def test_dist_sums_to_one(self):
        net = RainbowQNet((4, 15, 16), 6, num_atoms=11)
        dist = net.dist(torch.randn(2, 4, 15, 16))
        assert dist.shape == (2, 6, 11)
        assert torch.allclose(dist.sum(dim=-1), torch.ones(2, 6), atol=1e-5)

    def test_reset_noise(self):
        net = RainbowQNet((4, 15, 16), 6, num_atoms=11)
        x = torch.randn(1, 4, 15, 16)
        net.train()
        q1 = net(x)
        net.reset_noise()
        q2 = net(x)
        assert not torch.equal(q1, q2)


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(16, 8)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 8)

    def test_eval_is_deterministic(self):
        layer = NoisyLinear(16, 8)
        layer.eval()
        x = torch.randn(2, 16)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)

    def test_train_uses_noise(self):
        layer = NoisyLinear(16, 8)
        layer.train()
        x = torch.randn(2, 16)
        out1 = layer(x)
        layer.reset_noise()
        out2 = layer(x)
        assert not torch.equal(out1, out2)

    def test_reset_noise_changes_epsilon(self):
        layer = NoisyLinear(16, 8)
        eps_before = layer.weight_epsilon.clone()
        layer.reset_noise()
        assert not torch.equal(eps_before, layer.weight_epsilon)
