"""Base Trainer — shared infrastructure for all training loops."""

import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from rlcade.agent import create_agent
from rlcade.agent.base import wrap_agent
from rlcade.envs import create_vector_env, get_env_info
from rlcade.plugins import TrainerPlugin
from rlcade.utils import resolve_device, Metrics, ProgressBar
from rlcade.logger import get_logger

logger = get_logger(__name__)


class Distributed:
    """Distributed training mixin — auto-inits when launched with torchrun."""

    def __init__(self, *, backend="nccl"):
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1
        self._backend = None
        self._distributed = False

        if "RANK" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = dist.get_world_size()
            self._backend = backend
            self._distributed = True
            logger.info(
                "Rank %d/%d local_rank=%d (backend=%s)",
                self._rank,
                self._world_size,
                self._local_rank,
                backend,
            )

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def backend(self):
        return self._backend

    @property
    def distributed(self):
        return self._distributed

    @property
    def rank0(self):
        return self._rank == 0


class Trainer(ABC, Distributed):
    def __init__(self, args, *, plugins: list[TrainerPlugin] | None = None):
        Distributed.__init__(self, backend=args.backend)
        args.device = resolve_device(args.device)
        if self.distributed and args.device.startswith("cuda"):
            args.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)
            # Only set CPU affinity in distributed mode where each process is
            # bound to a specific local_rank GPU.  In local (non-distributed)
            # runs the user may not set CUDA_VISIBLE_DEVICES and could use
            # multiple GPUs, so we skip affinity to avoid restricting them.
            self._set_affinity()

        dist_kw = dict(rank=self.rank, world_size=self.world_size) if self.distributed else {}
        self.env = create_vector_env(args, label="train", **dist_kw)
        self.eval_env = create_vector_env(args, label="eval") if args.eval_interval and self.rank0 else None
        args.obs_shape, args.n_actions = get_env_info(self.env)

        self.agent = create_agent(args.agent, args, self.env)
        self.agent._impl.pin_memory = getattr(args, "pin_memory", True)
        self._distributed_strategy = args.distributed
        self._eager = getattr(args, "eager", False)

        logger.info(
            "Agent: %s | Obs: %s | Actions: %d | Device: %s",
            args.agent,
            args.obs_shape,
            args.n_actions,
            args.device,
        )

        self.max_iterations = args.max_iterations
        self.target_score = args.target_score
        self.eval_interval = args.eval_interval
        self.eval_episodes = args.eval_episodes
        self.log_interval = getattr(args, "log_interval", 1)
        self.metrics = Metrics()
        self.plugins = plugins or []

    @property
    def config(self) -> dict:
        return {
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
        }

    def _notify(self, method: str, *args) -> None:
        for p in self.plugins:
            getattr(p, method)(self, *args)

    def train(self):
        """Run the full training loop.

        Subclasses implement setup() and step() — never override train().
        The loop guarantees symmetric on_step_start/on_step_end for every iteration.
        """
        logger.info("Trainer config: %s", self.config)
        self.setup()
        pbar = ProgressBar(self.max_iterations, initial=self.start_iteration, disable=not self.rank0)
        last_logged = self.start_iteration

        try:
            for iteration in range(self.start_iteration + 1, self.max_iterations + 1):
                self._notify("on_step_start", iteration)
                self.step(iteration)
                self.reduce_score()
                summary = self.maybe_log(iteration)
                self._notify("on_step_end", iteration, summary)

                if summary:
                    pbar.update(summary, n=iteration - last_logged)
                    last_logged = iteration
                self.maybe_evaluate(iteration)
                if self.done:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            pbar.close()
            self._notify("on_done")

    def setup(self) -> None:
        """Called once before the training loop. Wraps agent, creates optimizers, runs plugin setup."""
        # Wrap with DDP/FSDP2 first so the distributed wrapper is innermost.
        # CUDAGraphWrapper.__getattr__ then peels through torch.compile + DDP
        # to reach the underlying module for non-forward calls (reset_noise, dist).
        self.agent = wrap_agent(self.agent, self._distributed_strategy, self.distributed)
        self.agent.compile(eager=self._eager, strategy=self._distributed_strategy)
        self.agent.create_optimizers()
        self._start_iteration = 0
        self._notify("on_setup")

    @property
    def start_iteration(self) -> int:
        return self._start_iteration

    @abstractmethod
    def step(self, iteration: int) -> None:
        """Run one training step. Updates self.metrics in place."""

    def maybe_log(self, iteration: int) -> dict[str, float] | None:
        """Build a display summary from metrics. Returns None to skip pbar."""
        if iteration % self.log_interval != 0:
            return None
        return self.metrics.summary()

    def maybe_evaluate(self, iteration: int):
        if self.eval_interval and self.eval_env and iteration % self.eval_interval == 0:
            scores = self.agent.evaluate(self.eval_env, num_episodes=self.eval_episodes)
            self.metrics.record_eval(scores)
            self._notify("on_eval", iteration, scores)

    def reduce_score(self) -> None:
        """All-reduce the score metric across ranks. No-op when not distributed."""
        score = self.metrics.mean_score()
        if not self.distributed:
            return
        t = torch.tensor([score], device=self.agent.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        score = t.item()
        self.metrics.score = score

    @property
    def done(self) -> bool:
        if self.target_score is None or len(self.metrics.episode_scores) < 10:
            return False
        return self.metrics.score >= self.target_score

    def _set_affinity(self) -> None:
        """Bind this process to CPUs on the same NUMA node as its GPU."""
        try:
            from rlcade.utils.affinity import set_gpu_affinity

            set_gpu_affinity(self.local_rank)
        except Exception:
            logger.debug("GPU affinity not available, skipping", exc_info=True)

    def swap(self, new_env) -> None:
        """Replace the current env. Subclasses override to reset their state."""
        old = self.env
        self.env = new_env
        if old is not None:
            old.close()
