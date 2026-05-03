import os
import tempfile

from rlcade.plugins.checkpoint import CheckpointPlugin
from tests.conftest import make_args, make_vec_args


def _ckpt_plugins(ckpt_path, num_steps=16):
    return [CheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=num_steps)]


class TestPPOTrainer:
    @staticmethod
    def _ckpt_path():
        d = tempfile.mkdtemp()
        return os.path.join(d, "test_ckpt.pt")

    def test_short_training_loop(self, rom):
        from rlcade.training import create_trainer

        ckpt_path = self._ckpt_path()
        args = make_args(rom, checkpoint_path=ckpt_path)
        trainer = create_trainer("ppo", args, plugins=_ckpt_plugins(ckpt_path))
        try:
            trainer.train()
            assert os.path.exists(ckpt_path)
            assert trainer.metrics.total_steps > 0
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)
            trainer.env.close()

    def test_training_with_eval(self, rom):
        from rlcade.training import create_trainer

        ckpt_path = self._ckpt_path()
        args = make_args(rom, checkpoint_path=ckpt_path, eval_interval=1, eval_episodes=1)
        trainer = create_trainer("ppo", args, plugins=_ckpt_plugins(ckpt_path))
        try:
            trainer.train()
            assert len(trainer.metrics.eval_scores) > 0
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)
            trainer.env.close()

    def test_training_with_lr_schedule(self, rom):
        from rlcade.training import create_trainer

        ckpt_path = self._ckpt_path()
        args = make_args(rom, checkpoint_path=ckpt_path, lr_schedule=True)
        trainer = create_trainer("ppo", args, plugins=_ckpt_plugins(ckpt_path))
        try:
            trainer.train()
            lr = trainer.agent.ppo.optimizer.param_groups[0]["lr"]
            assert lr < 2.5e-4
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)
            trainer.env.close()

    def test_custom_reward(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, custom_reward=True)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        rollout, _ = agent.collect_rollout(env, num_steps=8)
        assert rollout["rewards"].shape[0] == 8
        env.close()

    def test_vec_training_with_eval(self, rom):
        from rlcade.training import create_trainer

        ckpt_path = self._ckpt_path()
        args = make_vec_args(rom, checkpoint_path=ckpt_path, eval_interval=1, eval_episodes=1)
        trainer = create_trainer("ppo", args, plugins=_ckpt_plugins(ckpt_path))
        try:
            trainer.train()
            assert len(trainer.metrics.eval_scores) > 0
            assert trainer.metrics.total_steps > 0
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)
            trainer.env.close()
            if trainer.eval_env:
                trainer.eval_env.close()


class TestPPOSwap:
    def test_swap_resets_obs(self, rom):
        """After swap, obs should be None so collect_rollout does a fresh reset."""
        from rlcade.envs import create_vector_env
        from rlcade.training import create_trainer

        args = make_vec_args(rom, agent="ppo", max_iterations=5, checkpoint_path="")
        trainer = create_trainer("ppo", args)
        trainer.setup()

        # Run one step to populate obs
        trainer.step(1)
        assert trainer.obs is not None

        # Swap env
        args2 = make_vec_args(rom, agent="ppo")
        new_env = create_vector_env(args2)
        trainer.swap(new_env)
        assert trainer.obs is None

        # Next step should work (collect_rollout handles None obs)
        trainer.step(2)
        assert trainer.obs is not None
        trainer.env.close()
