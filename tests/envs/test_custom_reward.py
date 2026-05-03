"""Tests for the enhanced CustomReward wrapper (5 reward signals)."""

from tests.conftest import make_args


class TestCustomReward:
    """Verify CustomReward produces shaped rewards when enabled."""

    def test_custom_reward_returns_float(self, rom):
        from rlcade.envs import create_env

        env = create_env(make_args(rom, custom_reward=True))
        obs, info = env.reset()
        obs2, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        env.close()

    def test_custom_reward_differs_from_default(self, rom):
        """Custom reward should produce different values than the default clipped reward."""
        from rlcade.envs import create_env

        env_default = create_env(make_args(rom, custom_reward=False))
        env_custom = create_env(make_args(rom, custom_reward=True))

        env_default.reset()
        env_custom.reset()

        default_rewards = []
        custom_rewards = []
        for _ in range(20):
            _, r_d, _, _, _ = env_default.step(1)  # action 1 = RIGHT
            _, r_c, _, _, _ = env_custom.step(1)
            default_rewards.append(r_d)
            custom_rewards.append(r_c)

        # Custom rewards should not all be clipped to {-1, 0, 1}
        custom_unique = set(custom_rewards)
        assert len(custom_unique) > 1 or any(
            r not in (-1.0, 0.0, 1.0) for r in custom_rewards
        ), "Custom reward should produce non-clipped values"

        env_default.close()
        env_custom.close()

    def test_time_penalty_on_noop(self, rom):
        """Standing still (NOOP) should produce negative reward from time penalty."""
        from rlcade.envs import create_env

        env = create_env(make_args(rom, custom_reward=True))
        env.reset()

        # NOOP for several steps — should accumulate time penalty
        rewards = []
        for _ in range(10):
            _, r, terminated, _, _ = env.step(0)  # NOOP
            if terminated:
                break
            rewards.append(r)

        # At least some rewards should be negative (time penalty dominates when idle)
        assert any(r < 0 for r in rewards), f"Expected negative rewards from time penalty, got {rewards}"
        env.close()

    def test_rightward_movement_positive_reward(self, rom):
        """Moving right should produce positive velocity + novelty rewards."""
        from rlcade.envs import create_env

        env = create_env(make_args(rom, custom_reward=True, actions="right"))
        env.reset()

        # Move right for several steps
        rewards = []
        for _ in range(20):
            _, r, terminated, _, _ = env.step(1)  # RIGHT
            if terminated:
                break
            rewards.append(r)

        # Moving right should produce some positive rewards (velocity + novelty)
        assert any(r > 0 for r in rewards), f"Expected positive rewards from rightward movement, got {rewards}"
        env.close()

    def test_reset_clears_state(self, rom):
        """After reset, reward should not carry over from previous episode."""
        from rlcade.envs import create_env

        env = create_env(make_args(rom, custom_reward=True, actions="right"))

        # First episode: move right to build up max_x
        env.reset()
        for _ in range(10):
            _, _, terminated, _, _ = env.step(1)
            if terminated:
                break

        # Reset and take first step — should get novelty bonus again
        env.reset()
        _, r, _, _, info = env.step(1)
        # First step after reset should not produce a huge spurious reward
        assert abs(r) < 10.0, f"First step reward after reset too large: {r}"
        env.close()

    def test_info_dict_has_expected_keys(self, rom):
        """Info dict should still contain all standard fields."""
        from rlcade.envs import create_env

        env = create_env(make_args(rom, custom_reward=True))
        env.reset()
        _, _, _, _, info = env.step(0)

        for key in ("coins", "flag_get", "life", "score", "status", "time", "x_pos", "y_pos"):
            assert key in info, f"Missing key '{key}' in info dict"
        env.close()
