from tests.conftest import make_off_policy_trainer


class TestSACTrainer:
    def test_short_training_loop(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "sac")
        trainer.train()
        assert len(trainer.metrics.episode_scores) >= 0
        env.close()

    def test_training_with_eval(self, rom):
        trainer, env, eval_env = make_off_policy_trainer(rom, "sac", eval_interval=25)
        trainer.train()
        assert len(trainer.metrics.eval_scores) > 0
        env.close()
        eval_env.close()

    def test_vec_training(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "sac", vec=True)
        trainer.train()
        assert trainer.num_envs > 1
        env.close()
