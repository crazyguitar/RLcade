"""Super Mario Bros argument group."""

# fmt: off
def add_smb_args(parser):
    parser.add_argument("--world", type=int, default=None, help="World number (1-8); omit for full game")
    parser.add_argument("--stage", type=int, default=None, help="Stage number (1-4); omit for full game")
    parser.add_argument("--actions", type=str, default="simple", help="Action space (right, simple, complex)")
    parser.add_argument("--custom-reward", action="store_true", help="Enable score-based reward shaping")
# fmt: on
