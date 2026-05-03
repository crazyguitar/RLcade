"""Tests for ResNet encoder."""

import torch
import pytest

from rlcade.modules.encoders import ResNetEncoder, ResidualBlock, ConvSequence, create_encoder


class TestResidualBlock:
    def test_output_shape_unchanged(self):
        block = ResidualBlock(16)
        x = torch.randn(2, 16, 21, 21)
        assert block(x).shape == x.shape

    def test_skip_connection(self):
        block = ResidualBlock(16)
        # Zero-init all conv weights/biases -> conv outputs are zero
        # block(x) = x + (zero conv outputs) = x
        for p in block.parameters():
            p.data.zero_()
        x = torch.randn(2, 16, 10, 10)
        assert torch.allclose(block(x), x)


class TestConvSequence:
    def test_spatial_halving(self):
        seq = ConvSequence(4, 16)
        x = torch.randn(1, 4, 84, 84)
        out = seq(x)
        assert out.shape == (1, 16, 42, 42)

    def test_channel_change(self):
        seq = ConvSequence(3, 32)
        x = torch.randn(1, 3, 20, 20)
        out = seq(x)
        assert out.shape[1] == 32


class TestResNetEncoder:
    def test_output_shape_default(self):
        enc = ResNetEncoder((4, 84, 84))
        x = torch.randn(2, 4, 84, 84)
        out = enc(x)
        assert out.shape == (2, 256)

    def test_small_input(self):
        enc = ResNetEncoder((4, 15, 16))
        x = torch.randn(1, 4, 15, 16)
        out = enc(x)
        assert out.shape == (1, 256)

    def test_out_features_matches(self):
        enc = ResNetEncoder((4, 84, 84))
        x = torch.randn(1, 4, 84, 84)
        assert enc(x).shape[1] == enc.out_features

    def test_gradient_flow(self):
        enc = ResNetEncoder((4, 84, 84))
        x = torch.randn(1, 4, 84, 84)
        out = enc(x)
        out.sum().backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_custom_channels(self):
        enc = ResNetEncoder((4, 84, 84), channels=[32, 64])
        x = torch.randn(1, 4, 84, 84)
        out = enc(x)
        assert out.shape == (1, 256)
        assert enc.out_features == 256

    def test_custom_out_dim(self):
        enc = ResNetEncoder((4, 84, 84), out_dim=128)
        x = torch.randn(1, 4, 84, 84)
        out = enc(x)
        assert out.shape == (1, 128)
        assert enc.out_features == 128

    def test_create_encoder_registry(self):
        enc = create_encoder("resnet", (4, 84, 84))
        assert isinstance(enc, ResNetEncoder)
        assert enc.out_features == 256


class TestResNetIntegration:
    """Test ResNet encoder plugged into Actor, Critic, and QNet."""

    def test_actor_with_resnet(self):
        from rlcade.modules.actor import Actor

        actor = Actor((4, 84, 84), n_actions=7, encoder="resnet")
        x = torch.randn(2, 4, 84, 84)
        logits = actor(x)
        assert logits.shape == (2, 7)

    def test_critic_with_resnet(self):
        from rlcade.modules.critic import Critic

        critic = Critic((4, 84, 84), encoder="resnet")
        x = torch.randn(2, 4, 84, 84)
        val = critic(x)
        assert val.shape == (2,)

    def test_qnet_with_resnet(self):
        from rlcade.modules.qnet import QNet

        qnet = QNet((4, 84, 84), n_actions=7, encoder="resnet")
        x = torch.randn(2, 4, 84, 84)
        q = qnet(x)
        assert q.shape == (2, 7)
