import gymnasium as gym
import numpy as np

from tests.conftest import make_args, step_until_terminated

MAX_RANDOM_STEPS = 50000
FRAME_STACK = 4
OBS_SIZE = 84


# Helpers


def make_vec_env_config(rom, world, stage, *, actions="complex"):
    """Build a single vec-env config dict with sensible test defaults."""
    from rlcade.envs.smb import ACTIONS

    return dict(
        rom=rom,
        actions=list(ACTIONS[actions]),
        world=world,
        stage=stage,
        skip=4,
        episodic_life=True,
        custom_reward=False,
        clip_rewards=True,
        frame_stack=FRAME_STACK,
    )


def step_vec_until_auto_reset(env, num_envs, num_actions, seed=0):
    """Step with random actions until every env has auto-reset at least once.

    Returns a list of info dicts collected at the first auto-reset of each env.
    """
    rng = np.random.RandomState(seed)
    reset_infos = [None] * num_envs

    for _ in range(MAX_RANDOM_STEPS):
        actions = rng.randint(num_actions, size=num_envs).tolist()
        _, _, terminated, _, infos = env.step(actions)

        for env_idx in np.flatnonzero(terminated):
            if reset_infos[env_idx] is None:
                reset_infos[env_idx] = infos[env_idx]

        if all(info is not None for info in reset_infos):
            return reset_infos

    missing = sum(info is None for info in reset_infos)
    raise RuntimeError(f"{missing}/{num_envs} envs never terminated within {MAX_RANDOM_STEPS} steps")


def assert_world_stage(info, expected_world, expected_stage, label=""):
    """Assert info dict reports the expected world and stage."""
    prefix = f"{label}: " if label else ""
    assert info["world"] == expected_world, f"{prefix}expected world={expected_world}, got {info['world']}"
    assert info["stage"] == expected_stage, f"{prefix}expected stage={expected_stage}, got {info['stage']}"


# Tests


class TestSuperMarioBrosEnv:
    def test_single_env(self, rom):
        from rlcade.envs import create_env

        env = create_env(make_args(rom))
        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert env.action_space.n == 12
        env.close()

    def test_simple_actions(self, rom):
        from rlcade.envs import create_env

        env = create_env(make_args(rom, actions="simple"))
        assert env.action_space.n == 7
        env.close()

    def test_right_only_actions(self, rom):
        from rlcade.envs import create_env

        env = create_env(make_args(rom, actions="right"))
        assert env.action_space.n == 6
        env.close()

    def test_reset_and_step(self, rom):
        from rlcade.envs import create_env

        env = create_env(make_args(rom))
        obs, info = env.reset()
        assert obs.shape == (FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert obs.dtype == np.float32

        obs2, reward, terminated, truncated, info = env.step(0)
        assert obs2.shape == (FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()


class TestVecEnvParity:
    """Verify in-process VecEnv and AsyncVectorEnv produce identical results."""

    def test_vec_env_matches_async(self, rom):
        from rlcade.envs import create_vector_env
        from tests.conftest import make_vec_args

        args = make_vec_args(rom)
        vec_env = create_vector_env(args, use_gym=False)
        async_env = create_vector_env(args, use_gym=True)

        num_envs = vec_env.num_envs
        num_actions = vec_env.action_space.n

        rng = np.random.RandomState(42)

        vec_obs, _ = vec_env.reset()
        async_obs, _ = async_env.reset()

        np.testing.assert_array_almost_equal(vec_obs, async_obs, decimal=5)

        for _ in range(16):
            actions = rng.randint(0, num_actions, size=num_envs)
            vec_obs, vec_rewards, vec_terminated, vec_truncated, _ = vec_env.step(actions)
            async_obs, async_rewards, async_terminated, async_truncated, _ = async_env.step(actions)

            np.testing.assert_array_almost_equal(vec_rewards, async_rewards, decimal=5)
            np.testing.assert_array_equal(vec_terminated, async_terminated)

            # After a done, auto-reset obs may differ in ordering,
            # so only compare obs for non-terminated envs
            alive = ~vec_terminated
            if alive.any():
                np.testing.assert_array_almost_equal(vec_obs[alive], async_obs[alive], decimal=5)

        vec_env.close()
        async_env.close()


class TestWorldStageReset:
    """Verify reset returns to the correct world/stage after episode termination.

    Regression test for stale RAM bug: the NES CPU reset doesn't clear RAM,
    so game state from the previous episode could override write_stage values.
    """

    def test_single_stage_reset_preserves_world_stage(self, rom):
        """After termination, reset should return to the same world/stage."""
        from rlcade.envs import create_env

        env = create_env(make_args(rom, world=2, stage=2))
        _, info = env.reset()
        assert_world_stage(info, 2, 2)

        rng = np.random.RandomState(123)
        step_until_terminated(env, rng)

        _, info = env.reset()
        assert_world_stage(info, 2, 2, label="after reset")
        env.close()

    def test_vec_env_auto_reset_preserves_world_stage(self, rom):
        """Vec env auto-reset should return to the correct world/stage."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        config = make_vec_env_config(rom, world=2, stage=2)
        env = SuperMarioBrosVecEnv([config])
        env.reset()

        reset_infos = step_vec_until_auto_reset(env, num_envs=1, num_actions=len(config["actions"]), seed=456)
        assert_world_stage(reset_infos[0], 2, 2, label="after auto-reset")
        env.close()

    def test_multi_env_auto_reset_preserves_different_configs(self, rom):
        """Envs with different world/stage configs each preserve their own assignment after auto-reset."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        world_stage_pairs = [(1, 1), (2, 2), (3, 1)]
        configs = [make_vec_env_config(rom, world=w, stage=s) for w, s in world_stage_pairs]
        env = SuperMarioBrosVecEnv(configs)
        env.reset()

        reset_infos = step_vec_until_auto_reset(
            env, num_envs=len(configs), num_actions=len(configs[0]["actions"]), seed=789
        )
        for idx, (expected_world, expected_stage) in enumerate(world_stage_pairs):
            assert_world_stage(reset_infos[idx], expected_world, expected_stage, label=f"env {idx}")
        env.close()

    def test_identical_vec_envs_preserve_world_stage(self, rom):
        """32 identical envs targeting (4, 2) must ALL reset to 4-2, never 1-1.

        Regression: stale RAM caused write_stage to silently fall back to 1-1.
        """
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        target_world, target_stage, num_envs = 4, 2, 32
        config = make_vec_env_config(rom, world=target_world, stage=target_stage)
        configs = [config] * num_envs
        env = SuperMarioBrosVecEnv(configs)
        _, initial_infos = env.reset()

        for idx in range(num_envs):
            assert_world_stage(initial_infos[idx], target_world, target_stage, label=f"env {idx} initial")

        reset_infos = step_vec_until_auto_reset(env, num_envs, num_actions=len(config["actions"]), seed=999)
        for idx in range(num_envs):
            assert_world_stage(reset_infos[idx], target_world, target_stage, label=f"env {idx} auto-reset")
        env.close()

    def test_vec_reset_produces_valid_observations(self, rom):
        """Observations have correct shape, dtype, range [0,1], and become non-zero after stepping."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        num_envs = 2
        config = make_vec_env_config(rom, world=1, stage=1, actions="simple")
        env = SuperMarioBrosVecEnv([config] * num_envs)
        obs, _ = env.reset()

        expected_shape = (num_envs, FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert obs.shape == expected_shape
        assert obs.dtype == np.float32
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

        # SMB initial frame is black; step a few times to get gameplay pixels
        noop_actions = [0] * num_envs
        for _ in range(5):
            obs, _, _, _, _ = env.step(noop_actions)
        assert obs.sum() > 0, "Observation still all zeros after stepping"
        env.close()

    def test_auto_reset_obs_differs_from_final_observation(self, rom):
        """After termination, the returned obs (new episode) differs from final_observation (terminal frame)."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        config = make_vec_env_config(rom, world=1, stage=1)
        env = SuperMarioBrosVecEnv([config])
        env.reset()

        rng = np.random.RandomState(321)
        num_actions = len(config["actions"])
        for _ in range(MAX_RANDOM_STEPS):
            obs, _, terminated, _, infos = env.step([rng.randint(num_actions)])
            if not terminated[0]:
                continue
            assert "final_observation" in infos[0], "Auto-reset should provide final_observation"
            final_obs = np.array(infos[0]["final_observation"], dtype=np.float32)
            post_reset_obs = obs[0].flatten()
            assert not np.array_equal(post_reset_obs, final_obs), "Post-reset obs should differ from terminal frame"
            break
        env.close()


class TestVecEnvSlice:
    def test_slice_num_envs(self, rom):
        """Slicing produces a vec env with the correct number of envs."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        configs = [make_vec_env_config(rom, world=w, stage=1) for w in range(1, 5)]
        env = SuperMarioBrosVecEnv(configs)
        sliced = env[0:2]
        assert sliced.num_envs == 2
        assert env.num_envs == 4

    def test_slice_step_and_reset(self, rom):
        """Sliced env can step and reset with correct shapes."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        configs = [make_vec_env_config(rom, world=w, stage=1) for w in range(1, 5)]
        env = SuperMarioBrosVecEnv(configs)
        sliced = env[1:3]
        obs, _ = sliced.reset()
        assert obs.shape == (2, FRAME_STACK, OBS_SIZE, OBS_SIZE)

        actions = [0, 0]
        obs, rewards, terminated, truncated, infos = sliced.step(actions)
        assert obs.shape == (2, FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)

    def test_slice_preserves_world_stage(self, rom):
        """Sliced envs retain their original world/stage assignment."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        pairs = [(1, 1), (2, 1), (3, 1), (4, 1)]
        configs = [make_vec_env_config(rom, world=w, stage=s) for w, s in pairs]
        env = SuperMarioBrosVecEnv(configs)
        sliced = env[2:4]
        _, infos = sliced.reset()
        assert_world_stage(infos[0], 3, 1, label="sliced env 0")
        assert_world_stage(infos[1], 4, 1, label="sliced env 1")

    def test_multiple_slices_independent(self, rom):
        """Two slices of the same env can step independently."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        configs = [make_vec_env_config(rom, world=w, stage=1) for w in range(1, 5)]
        env = SuperMarioBrosVecEnv(configs)
        a = env[0:2]
        b = env[2:4]

        a.reset()
        b.reset()

        obs_a, _, _, _, _ = a.step([0, 0])
        obs_b, _, _, _, _ = b.step([1, 1])
        assert obs_a.shape == (2, FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert obs_b.shape == (2, FRAME_STACK, OBS_SIZE, OBS_SIZE)

    def test_int_indexing(self, rom):
        """Integer indexing returns a single-env vec env."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        configs = [make_vec_env_config(rom, world=w, stage=1) for w in range(1, 4)]
        env = SuperMarioBrosVecEnv(configs)
        single = env[1]
        assert single.num_envs == 1
        obs, infos = single.reset()
        assert obs.shape == (1, FRAME_STACK, OBS_SIZE, OBS_SIZE)
        assert_world_stage(infos[0], 2, 1, label="int index")

    def test_negative_indexing(self, rom):
        """Negative indexing works like Python lists."""
        from rlcade.envs.smb import SuperMarioBrosVecEnv

        configs = [make_vec_env_config(rom, world=w, stage=1) for w in range(1, 4)]
        env = SuperMarioBrosVecEnv(configs)
        single = env[-1]
        assert single.num_envs == 1
        _, infos = single.reset()
        assert_world_stage(infos[0], 3, 1, label="negative index")
