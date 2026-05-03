"""PyTorch CUDA memory profiling argument group."""

# fmt: off
def add_memory_profiler_args(parser):
    group = parser.add_argument_group("memory_profiler", "PyTorch CUDA memory profiling options")
    group.add_argument("--memory-profiler", action="store_true", help="Enable CUDA memory profiling")
    group.add_argument("--memory-profiler-start", type=int, default=1, help="Start recording at this step")
    group.add_argument("--memory-profiler-end", type=int, default=10, help="Stop recording at this step")
    group.add_argument("--memory-profiler-output", type=str, default="memory_snapshot.pickle", help="Output snapshot file path")
    group.add_argument("--memory-profiler-max-entries", type=int, default=100000, help="Max memory history entries to keep")
# fmt: on
