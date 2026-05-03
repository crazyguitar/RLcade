from tests.conftest import make_off_policy_trainer


class TestDQNTrainer:
    def test_short_training_loop(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "dqn")
        trainer.train()
        assert len(trainer.metrics.episode_scores) >= 0
        env.close()

    def test_training_with_eval(self, rom):
        trainer, env, eval_env = make_off_policy_trainer(rom, "dqn", eval_interval=25)
        trainer.train()
        assert len(trainer.metrics.eval_scores) > 0
        env.close()
        eval_env.close()

    def test_vec_training(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "dqn", vec=True)
        trainer.train()
        assert trainer.num_envs > 1
        env.close()

    def test_swap_no_crash(self, rom):
        """After swap, the next step() should not crash."""
        from rlcade.envs import create_vector_env
        from tests.conftest import make_vec_args

        trainer, env, _ = make_off_policy_trainer(rom, "dqn", vec=True)
        trainer.setup()

        for i in range(1, 6):
            trainer.step(i)

        new_env = create_vector_env(make_vec_args(rom, agent="dqn"))
        trainer.swap(new_env)
        trainer.step(6)
        assert trainer.obs is not None
        trainer.env.close()
