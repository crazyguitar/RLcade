import pytest

from rlcade.plugins.curriculum import CurriculumPlugin
from tests.conftest import make_args


class TestCurriculumPlugin:
    def test_initial_stages(self, rom):
        args = make_args(rom)
        stages = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
        plugin = CurriculumPlugin(args, stages=stages, initial_stages=4)
        assert len(plugin.active_stages) == 4
        assert plugin.active_stages == [(1, 1), (1, 2), (1, 3), (1, 4)]

    def test_expand_creates_new_env(self, rom):
        """After exceeding threshold, curriculum should expand and rebuild env."""
        from rlcade.training import create_trainer
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="ppo", max_iterations=10, checkpoint_path="")

        stages = [(1, 1), (1, 2), (1, 3), (1, 4)]
        plugin = CurriculumPlugin(
            args,
            stages=stages,
            initial_stages=2,
            expand_threshold=0.0,  # always expand
            expand_count=1,
            window=1,
        )

        trainer = create_trainer("ppo", args, plugins=[plugin])
        trainer.setup()

        # Initial env has 2 stages (on_setup called _rebuild_env)
        assert trainer.env.num_envs == 2

        # Simulate one episode completing with any score
        trainer.metrics.record_episodes([1.0])
        plugin.on_step_end(trainer, 1, {})

        # Should have expanded to 3 stages
        assert plugin.current_count == 3
        assert trainer.env.num_envs == 3

        trainer.env.close()

    def test_no_expand_below_threshold(self, rom):
        from rlcade.envs import create_vector_env
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="ppo")
        env = create_vector_env(args)

        stages = [(1, 1), (1, 2), (1, 3), (1, 4)]
        plugin = CurriculumPlugin(
            args,
            stages=stages,
            initial_stages=2,
            expand_threshold=9999.0,
            window=3,
        )

        # Fake a trainer-like object with metrics
        class _Trainer:
            pass

        t = _Trainer()
        t.env = env
        t.metrics = type("M", (), {"episode_scores": [10.0, 20.0, 30.0]})()
        t.num_envs = env.num_envs

        plugin.on_step_end(t, 1, {})
        assert plugin.current_count == 2  # no expansion
        env.close()

    def test_cap_at_max_stages(self, rom):
        args = make_args(rom)
        stages = [(1, 1), (1, 2)]
        plugin = CurriculumPlugin(
            args,
            stages=stages,
            initial_stages=2,
            expand_threshold=0.0,
            expand_count=4,
            window=1,
        )
        plugin._rebuild_env = lambda _trainer: None  # skip env rebuild for this unit check

        class _Trainer:
            pass

        t = _Trainer()
        t.metrics = type("M", (), {"episode_scores": [999.0]})()

        plugin.on_step_end(t, 1, {})
        assert plugin.current_count == 2  # already at max

    def test_full_training_loop_with_curriculum(self, rom):
        """Run a short training loop with curriculum to verify no crashes."""
        from rlcade.training import create_trainer
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="ppo", max_iterations=5, checkpoint_path="")

        plugin = CurriculumPlugin(
            args,
            stages=[(1, 1), (1, 2), (1, 3), (1, 4)],
            initial_stages=2,
            expand_threshold=0.0,
            expand_count=1,
            window=1,
        )

        trainer = create_trainer("ppo", args, plugins=[plugin])

        # on_setup is called by trainer.train() -> setup() which rebuilds the env
        # Verify initial env was rebuilt with 2 stages after train starts

        trainer.train()
        # No crash is the main assertion — expansion depends on episodes completing
        trainer.env.close()
