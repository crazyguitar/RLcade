"""Launcher argument group."""

# fmt: off
def add_launcher_args(parser):
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "elastic", "mp", "ray"],
                        help="Launch backend: none (direct), elastic (torchrun), mp (multiprocessing), ray")
    # elastic / mp
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Processes per node (elastic/mp)")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes (elastic)")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master address (mp)")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port (mp/elastic)")
    # ray
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (ray://host:port)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Total GPUs to use (ray, auto-detected if omitted)")
# fmt: on
