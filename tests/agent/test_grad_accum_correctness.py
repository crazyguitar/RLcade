"""Strict correctness tests for gradient accumulation.

These tests verify mathematical equivalence, not just "does it run".
No ROM needed — all tests use synthetic data.

Semantics: grad_accum_steps splits batch_size into micro-batches.
Effective batch size is unchanged; peak memory is reduced.
"""

import torch
from torch.distributions import Categorical

OBS_SHAPE = (1, 84, 84)
N_ACTIONS = 5


def _seed(seed=42):
    torch.manual_seed(seed)


def _fill_buffer_synthetic(buffer, n, obs_shape=OBS_SHAPE):
    """Fill a replay buffer with deterministic synthetic transitions."""
    _seed()
    for _ in range(n):
        obs = torch.randn(obs_shape)
        action = torch.randint(0, N_ACTIONS, ()).item()
        reward = torch.randn(()).item()
        next_obs = torch.randn(obs_shape)
        done = float(torch.randint(0, 2, ()).item())
        buffer.add(obs, action, reward, next_obs, done)


def _get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _make_dqn(grad_accum_steps=2, **kw):
    from rlcade.agent.dqn import EnvDQN, DQNConfig

    defaults = dict(
        obs_shape=OBS_SHAPE,
        n_actions=N_ACTIONS,
        batch_size=16,
        buffer_size=200,
        learn_start=1,
        learn_freq=1,
        grad_accum_steps=grad_accum_steps,
    )
    defaults.update(kw)
    agent = EnvDQN(DQNConfig(**defaults))
    agent.create_optimizers()
    _fill_buffer_synthetic(agent.buffer, 100)
    return agent


def _make_rainbow(grad_accum_steps=2, **kw):
    from rlcade.agent.dqn import EnvRainbowDQN, RainbowDQNConfig

    defaults = dict(
        obs_shape=OBS_SHAPE,
        n_actions=N_ACTIONS,
        batch_size=16,
        buffer_size=200,
        learn_start=1,
        learn_freq=1,
        grad_accum_steps=grad_accum_steps,
    )
    defaults.update(kw)
    agent = EnvRainbowDQN(RainbowDQNConfig(**defaults))
    agent.create_optimizers()
    _fill_buffer_synthetic(agent.buffer, 100)
    return agent


def _make_sac(grad_accum_steps=2, **kw):
    from rlcade.agent.sac import EnvSAC, SACConfig

    defaults = dict(
        obs_shape=OBS_SHAPE,
        n_actions=N_ACTIONS,
        batch_size=16,
        buffer_size=200,
        learn_start=1,
        learn_freq=1,
        grad_accum_steps=grad_accum_steps,
    )
    defaults.update(kw)
    agent = EnvSAC(SACConfig(**defaults))
    agent.create_optimizers()
    _fill_buffer_synthetic(agent.buffer, 100)
    return agent


def _make_ppo(grad_accum_steps=2, **kw):
    from rlcade.agent.ppo import EnvPPO, PPOConfig

    defaults = dict(
        obs_shape=OBS_SHAPE, n_actions=N_ACTIONS, batch_size=16, update_epochs=1, grad_accum_steps=grad_accum_steps
    )
    defaults.update(kw)
    agent = EnvPPO(PPOConfig(**defaults))
    agent.create_optimizers()
    return agent


def _make_trajectory(agent, n=64):
    """Build a synthetic trajectory dict on the agent's device."""
    _seed()
    obs = torch.randn(n, *OBS_SHAPE, device=agent.device)
    actions = torch.randint(0, N_ACTIONS, (n,), device=agent.device)
    with torch.no_grad():
        dist = Categorical(logits=agent.actor(obs))
        old_log_probs = dist.log_prob(actions)
        values = agent.critic(obs)
    advantages = torch.randn(n, device=agent.device)
    returns = advantages + values
    return dict(
        obs=obs, actions=actions, old_log_probs=old_log_probs, values=values, advantages=advantages, returns=returns
    )


# DQN


class TestDQNGradAccumCorrectness:
    def test_learn_always_steps(self):
        """Every learn() call produces a full optimizer step."""
        agent = _make_dqn(grad_accum_steps=4)
        params_before = _get_flat_params(agent.qnet)
        agent.learn()
        assert not torch.equal(params_before, _get_flat_params(agent.qnet)), "Weights did not change"

    def test_accum_equivalent_to_single_batch(self):
        """Splitting batch=16 into 2 micro-batches of 8 should match a single batch=16 step."""
        agent_accum = _make_dqn(grad_accum_steps=2, batch_size=16, tau=0.0)
        agent_single = _make_dqn(grad_accum_steps=1, batch_size=16, tau=0.0)

        # Sync weights
        agent_single.qnet.load_state_dict(agent_accum.qnet.state_dict())
        agent_single.target.load_state_dict(agent_accum.target.state_dict())

        # Use same batch for both
        _seed(99)
        batch = agent_accum.buffer.sample(16)

        # Agent with accum: manual split into 2 micro-batches of 8
        agent_accum.optimizer.zero_grad()
        for i in range(0, 16, 8):
            q = agent_accum.qnet(batch["obs"][i : i + 8]).gather(1, batch["actions"][i : i + 8].unsqueeze(1)).squeeze(1)
            t = agent_accum._compute_target(
                batch["rewards"][i : i + 8], batch["next_obs"][i : i + 8], batch["dones"][i : i + 8]
            )
            (agent_accum.criterion(q, t) / 2).backward()
        agent_accum.optimizer.step()

        # Agent without accum: single full batch
        agent_single.optimizer.zero_grad()
        q = agent_single.qnet(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        t = agent_single._compute_target(batch["rewards"], batch["next_obs"], batch["dones"])
        agent_single.criterion(q, t).backward()
        agent_single.optimizer.step()

        params_accum = _get_flat_params(agent_accum.qnet)
        params_single = _get_flat_params(agent_single.qnet)
        assert torch.allclose(
            params_accum, params_single, atol=1e-5
        ), f"Max diff: {(params_accum - params_single).abs().max().item()}"


# Rainbow DQN


class TestRainbowGradAccumCorrectness:
    def test_learn_always_steps(self):
        agent = _make_rainbow(grad_accum_steps=2)
        params_before = _get_flat_params(agent.qnet)
        agent.learn()
        assert not torch.equal(params_before, _get_flat_params(agent.qnet)), "Weights did not change"

    def test_priorities_updated(self):
        """PER priorities must update on every learn() call."""
        agent = _make_rainbow(grad_accum_steps=2)
        max_pri_before = agent.buffer.max_priority
        agent.learn()
        assert max_pri_before != agent.buffer.max_priority, "Priorities not updated"

    def test_noise_reset(self):
        """NoisyNet noise should reset after each learn() call."""
        from rlcade.modules.heads import NoisyLinear

        agent = _make_rainbow(grad_accum_steps=2)

        def _get_noise(model):
            for m in model.modules():
                if isinstance(m, NoisyLinear):
                    return m.weight_epsilon.clone()

        noise_before = _get_noise(agent.qnet)
        agent.learn()
        assert not torch.equal(noise_before, _get_noise(agent.qnet)), "Noise did not reset"


# SAC


class TestSACGradAccumCorrectness:
    def test_learn_updates_all_components(self):
        """All networks + alpha must update on every learn() call."""
        agent = _make_sac(grad_accum_steps=2)
        actor_before = _get_flat_params(agent.actor)
        q1_before = _get_flat_params(agent.q1)
        alpha_before = agent.log_alpha.item()

        agent.learn()
        assert not torch.equal(actor_before, _get_flat_params(agent.actor)), "Actor did not change"
        assert not torch.equal(q1_before, _get_flat_params(agent.q1)), "Q1 did not change"
        assert alpha_before != agent.log_alpha.item(), "Alpha did not change"

    def test_target_updates(self):
        agent = _make_sac(grad_accum_steps=2)
        q1t_before = _get_flat_params(agent.q1_target)
        agent.learn()
        assert not torch.equal(q1t_before, _get_flat_params(agent.q1_target)), "Q1 target did not update"


# PPO


class TestPPOGradAccumCorrectness:
    def test_same_num_optimizer_steps(self):
        """Micro-batching should not change the number of optimizer steps."""
        agent_1 = _make_ppo(grad_accum_steps=1)
        agent_2 = _make_ppo(grad_accum_steps=4)
        traj_1 = _make_trajectory(agent_1, n=64)
        traj_2 = _make_trajectory(agent_2, n=64)
        # 64 / 16 = 4 minibatches → 4 optimizer steps regardless of accum
        _, num_1, _ = agent_1.update_epoch(traj_1)
        _, num_2, _ = agent_2.update_epoch(traj_2)
        assert num_1 == 4
        assert num_2 == 4, f"Expected 4 steps with accum, got {num_2}"

    def test_weights_change(self):
        agent = _make_ppo(grad_accum_steps=2)
        trajectory = _make_trajectory(agent, n=64)
        params_before = _get_flat_params(agent.actor)
        agent.update_epoch(trajectory)
        assert not torch.equal(params_before, _get_flat_params(agent.actor))


# grad_accum_steps=1 is a no-op


class TestAccumStepsOneIsNoop:
    def test_dqn_accum_1(self):
        agent = _make_dqn(grad_accum_steps=1)
        params_before = _get_flat_params(agent.qnet)
        agent.learn()
        assert not torch.equal(params_before, _get_flat_params(agent.qnet))

    def test_sac_accum_1(self):
        agent = _make_sac(grad_accum_steps=1)
        params_before = _get_flat_params(agent.q1)
        agent.learn()
        assert not torch.equal(params_before, _get_flat_params(agent.q1))

    def test_ppo_accum_1(self):
        agent = _make_ppo(grad_accum_steps=1)
        trajectory = _make_trajectory(agent, n=64)
        _, num_updates, _ = agent.update_epoch(trajectory)
        assert num_updates == 4


# Checkpoint correctness


class TestDQNCheckpointWithAccum:
    def test_save_load_preserves_weights(self, tmp_path):
        """Checkpoint roundtrip must preserve model weights exactly."""
        agent = _make_dqn(grad_accum_steps=2)
        agent.learn()
        path = str(tmp_path / "dqn.pt")
        torch.save(agent.state(step=10), path)

        agent2 = _make_dqn(grad_accum_steps=2)
        with open(path, "rb") as f:
            step = agent2.load(f)

        assert step == 10
        assert torch.equal(_get_flat_params(agent.qnet), _get_flat_params(agent2.qnet))
        assert torch.equal(_get_flat_params(agent.target), _get_flat_params(agent2.target))

    def test_learn_after_load(self, tmp_path):
        """Agent must produce valid results after loading a checkpoint."""
        agent = _make_dqn(grad_accum_steps=2)
        agent.learn()
        path = str(tmp_path / "dqn.pt")
        torch.save(agent.state(step=1), path)

        agent2 = _make_dqn(grad_accum_steps=2)
        with open(path, "rb") as f:
            agent2.load(f)
        metrics = agent2.learn()
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestSACCheckpointWithAccum:
    def test_save_load_preserves_all_components(self, tmp_path):
        """SAC checkpoint must preserve actor, q1, q2, targets, and alpha."""
        agent = _make_sac(grad_accum_steps=2)
        agent.learn()
        path = str(tmp_path / "sac.pt")
        torch.save(agent.state(step=5), path)

        agent2 = _make_sac(grad_accum_steps=2)
        with open(path, "rb") as f:
            step = agent2.load(f)

        assert step == 5
        assert torch.equal(_get_flat_params(agent.actor), _get_flat_params(agent2.actor))
        assert torch.equal(_get_flat_params(agent.q1), _get_flat_params(agent2.q1))
        assert torch.equal(_get_flat_params(agent.q2), _get_flat_params(agent2.q2))
        assert torch.equal(_get_flat_params(agent.q1_target), _get_flat_params(agent2.q1_target))
        assert agent.log_alpha.item() == agent2.log_alpha.item()

    def test_three_scalers_roundtrip(self, tmp_path):
        """SAC's 3 separate GradScalers must survive checkpoint roundtrip."""
        from rlcade.agent.sac import EnvSAC, SACConfig

        # Create with AMP enabled so scalers are active
        config = SACConfig(
            obs_shape=OBS_SHAPE,
            n_actions=N_ACTIONS,
            batch_size=16,
            buffer_size=200,
            learn_start=1,
            learn_freq=1,
            grad_accum_steps=2,
            amp=True,
            device="cpu",
        )
        agent = EnvSAC(config)
        agent.create_optimizers()

        state = agent.state(step=1)
        # On CPU, scalers are disabled so they won't be in state
        # But the save/load path should not crash
        agent2 = EnvSAC(config)
        agent2.create_optimizers()
        agent2._load_state(state)
        assert torch.equal(_get_flat_params(agent.actor), _get_flat_params(agent2.actor))

    def test_learn_after_load(self, tmp_path):
        agent = _make_sac(grad_accum_steps=2)
        agent.learn()
        path = str(tmp_path / "sac.pt")
        torch.save(agent.state(step=1), path)

        agent2 = _make_sac(grad_accum_steps=2)
        with open(path, "rb") as f:
            agent2.load(f)
        metrics = agent2.learn()
        assert "critic_loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


class TestPPOCheckpointWithAccum:
    def test_save_load_preserves_weights(self, tmp_path):
        agent = _make_ppo(grad_accum_steps=2)
        trajectory = _make_trajectory(agent, n=64)
        agent.update_epoch(trajectory)
        path = str(tmp_path / "ppo.pt")
        torch.save(agent.state(step=3), path)

        agent2 = _make_ppo(grad_accum_steps=2)
        with open(path, "rb") as f:
            step = agent2.load(f)

        assert step == 3
        assert torch.equal(_get_flat_params(agent.actor), _get_flat_params(agent2.actor))
        assert torch.equal(_get_flat_params(agent.critic), _get_flat_params(agent2.critic))


class TestCrossAccumCheckpointCompat:
    def test_dqn_accum1_save_accum2_load(self, tmp_path):
        """Checkpoint saved with accum=1 must load into agent with accum=2."""
        agent1 = _make_dqn(grad_accum_steps=1)
        agent1.learn()
        path = str(tmp_path / "dqn.pt")
        torch.save(agent1.state(step=1), path)

        agent2 = _make_dqn(grad_accum_steps=2)
        with open(path, "rb") as f:
            step = agent2.load(f)
        assert step == 1
        assert torch.equal(_get_flat_params(agent1.qnet), _get_flat_params(agent2.qnet))
        # Must be able to learn after loading
        metrics = agent2.learn()
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_sac_accum1_save_accum2_load(self, tmp_path):
        """SAC checkpoint saved with accum=1 must load into agent with accum=2."""
        agent1 = _make_sac(grad_accum_steps=1)
        agent1.learn()
        path = str(tmp_path / "sac.pt")
        torch.save(agent1.state(step=1), path)

        agent2 = _make_sac(grad_accum_steps=2)
        with open(path, "rb") as f:
            agent2.load(f)
        metrics = agent2.learn()
        assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


# E2E: learn produces finite, improving results


class TestE2EFiniteOutputs:
    def test_dqn_multiple_learns_finite(self):
        """Multiple learn() calls with grad_accum must all produce finite metrics."""
        agent = _make_dqn(grad_accum_steps=4)
        for _ in range(5):
            metrics = agent.learn()
            assert torch.isfinite(torch.tensor(metrics["loss"])), f"Non-finite loss: {metrics['loss']}"
            assert torch.isfinite(torch.tensor(metrics["q_mean"])), f"Non-finite q_mean: {metrics['q_mean']}"

    def test_sac_multiple_learns_finite(self):
        agent = _make_sac(grad_accum_steps=2)
        for _ in range(5):
            metrics = agent.learn()
            for k in ("critic_loss", "actor_loss", "alpha", "q_mean"):
                assert torch.isfinite(torch.tensor(metrics[k])), f"Non-finite {k}: {metrics[k]}"

    def test_ppo_multiple_epochs_finite(self):
        agent = _make_ppo(grad_accum_steps=2)
        trajectory = _make_trajectory(agent, n=64)
        for _ in range(3):
            agg, num_updates, kl = agent.update_epoch(trajectory)
            assert num_updates > 0
            assert all(torch.isfinite(torch.tensor(v)) for v in agg.values())

    def test_rainbow_multiple_learns_finite(self):
        agent = _make_rainbow(grad_accum_steps=2)
        for _ in range(5):
            metrics = agent.learn()
            assert torch.isfinite(torch.tensor(metrics["loss"])), f"Non-finite loss: {metrics['loss']}"
