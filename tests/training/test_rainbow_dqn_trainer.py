from tests.conftest import make_off_policy_trainer


class TestRainbowDQNTrainer:
    def test_short_training_loop(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "rainbow_dqn", qnet="rainbow_qnet")
        trainer.train()
        assert len(trainer.metrics.episode_scores) >= 0
        env.close()

    def test_training_with_eval(self, rom):
        trainer, env, eval_env = make_off_policy_trainer(rom, "rainbow_dqn", qnet="rainbow_qnet", eval_interval=25)
        trainer.train()
        assert len(trainer.metrics.eval_scores) > 0
        env.close()
        eval_env.close()

    def test_beta_annealing(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "rainbow_dqn", qnet="rainbow_qnet")
        initial_beta = trainer.agent.beta
        trainer.train()
        assert trainer.agent.beta > initial_beta
        env.close()

    def test_vec_training(self, rom):
        trainer, env, _ = make_off_policy_trainer(rom, "rainbow_dqn", qnet="rainbow_qnet", vec=True)
        trainer.train()
        assert trainer.num_envs > 1
        env.close()
