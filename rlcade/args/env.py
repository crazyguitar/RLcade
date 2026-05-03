"""Environment argument group."""

# fmt: off
def add_env_args(parser):
    parser.add_argument("--env", type=str, default="rlcade/SuperMarioBros-v0", help="Gymnasium env ID")
    parser.add_argument("--rom", type=str, required=True, help="Path to .nes ROM file")
    parser.add_argument("--render-mode", type=str, default=None, help="Render mode (human, rgb_array)")
# fmt: on
