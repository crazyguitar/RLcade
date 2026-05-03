"""Train entrypoint — importable from any process."""

from rlcade.training import create_trainer
from rlcade.logger import get_logger

logger = get_logger(__name__)


def _build_plugins(args):
    plugins = []

    # Checkpoint plugin -- always first so on_setup loads before other plugins run
    if getattr(args, "async_checkpoint", False):
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin as _Plugin
    else:
        from rlcade.plugins.checkpoint import CheckpointPlugin as _Plugin

    num_steps = getattr(args, "num_steps", None)
    plugins.append(
        _Plugin(
            checkpoint_path=args.checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            num_steps=num_steps,
        )
    )

    if args.tensorboard:
        from rlcade.plugins.tensorboard import TensorBoardPlugin

        plugins.append(TensorBoardPlugin(log_dir=args.tensorboard))

    if getattr(args, "nsys", False):
        from rlcade.plugins.nsys import NsysPlugin

        plugins.append(NsysPlugin(start_step=args.nsys_start, end_step=args.nsys_end))

    if getattr(args, "memory_profiler", False):
        from rlcade.plugins.memory_profiler import MemoryProfilerPlugin

        plugins.append(
            MemoryProfilerPlugin(
                start_step=args.memory_profiler_start,
                end_step=args.memory_profiler_end,
                output_file=args.memory_profiler_output,
                max_entries=args.memory_profiler_max_entries,
            )
        )

    if args.viztracer:
        from rlcade.plugins.viztracer import VizTracerPlugin

        viztracer_kwargs = {
            "output_file": args.viztracer,
            "tracer_entries": args.viztracer_tracer_entries,
            "max_stack_depth": args.viztracer_max_stack_depth,
            "include_files": args.viztracer_include_files,
            "exclude_files": args.viztracer_exclude_files,
            "ignore_c_function": args.viztracer_ignore_c_function,
            "ignore_frozen": args.viztracer_ignore_frozen,
            "log_func_retval": args.viztracer_log_func_retval,
            "log_func_args": args.viztracer_log_func_args,
            "log_print": args.viztracer_log_print,
            "log_gc": args.viztracer_log_gc,
            "log_sparse": args.viztracer_log_sparse,
            "log_async": args.viztracer_log_async,
            "log_torch": args.viztracer_log_torch,
            "min_duration": args.viztracer_min_duration,
            "minimize_memory": args.viztracer_minimize_memory,
            "process_name": args.viztracer_process_name,
        }
        plugins.append(
            VizTracerPlugin(
                start_step=args.viztracer_start,
                end_step=args.viztracer_end,
                **viztracer_kwargs,
            )
        )
    return plugins


def _build_curriculum(args, plugins):
    if getattr(args, "curriculum", False):
        from rlcade.plugins.curriculum import CurriculumPlugin

        curriculum = CurriculumPlugin(
            args,
            initial_stages=args.curriculum_initial,
            expand_threshold=args.curriculum_threshold,
            expand_count=args.curriculum_expand,
        )
        plugins.append(curriculum)


def train_fn(args):
    """Run training — called once per process by the launcher."""
    plugins = _build_plugins(args)
    _build_curriculum(args, plugins)
    trainer = create_trainer(args.agent, args, plugins=plugins)
    trainer.train()
