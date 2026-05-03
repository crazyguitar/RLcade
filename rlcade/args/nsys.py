"""Nsight Systems profiling argument group."""

# fmt: off
def add_nsys_args(parser):
    group = parser.add_argument_group("nsys", "Nsight Systems profiling options")
    group.add_argument("--nsys", action="store_true", help="Enable Nsight Systems profiling")
    group.add_argument("--nsys-start", type=int, default=1, help="Start profiling at this step")
    group.add_argument("--nsys-end", type=int, default=10, help="Stop profiling at this step")
# fmt: on
