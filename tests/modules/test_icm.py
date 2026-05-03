import torch

from rlcade.modules.icm import ICM


class TestICM:
    def test_output_shapes(self):
        icm = ICM((4, 15, 16), n_actions=12, feature_dim=128)
        obs = torch.randn(4, 4, 15, 16)
        next_obs = torch.randn(4, 4, 15, 16)
        actions = torch.randint(0, 12, (4,))
        intrinsic, fwd_loss, inv_loss = icm(obs, next_obs, actions)
        assert intrinsic.shape == (4,)
        assert fwd_loss.dim() == 0
        assert inv_loss.dim() == 0

    def test_intrinsic_reward_positive(self):
        icm = ICM((4, 15, 16), n_actions=6, feature_dim=64)
        obs = torch.randn(2, 4, 15, 16)
        next_obs = torch.randn(2, 4, 15, 16)
        actions = torch.randint(0, 6, (2,))
        intrinsic, _, _ = icm(obs, next_obs, actions)
        assert (intrinsic >= 0).all()

    def test_losses_are_differentiable(self):
        icm = ICM((4, 15, 16), n_actions=6, feature_dim=64)
        obs = torch.randn(2, 4, 15, 16)
        next_obs = torch.randn(2, 4, 15, 16)
        actions = torch.randint(0, 6, (2,))
        _, fwd_loss, inv_loss = icm(obs, next_obs, actions)
        loss = fwd_loss + inv_loss
        loss.backward()
        grad_count = sum(1 for p in icm.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_encode_output_shape(self):
        icm = ICM((4, 15, 16), n_actions=6, feature_dim=128)
        features = icm.encode(torch.randn(3, 4, 15, 16))
        assert features.shape == (3, 128)
