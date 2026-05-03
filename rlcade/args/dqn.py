"""DQN hyperparameter argument group."""

# fmt: off
def add_dqn_args(parser):
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=100000, help="Epsilon decay steps")
    parser.add_argument("--double", action="store_true", default=True, help="Use double DQN")
    # --tau, --buffer-size, --learn-start, --learn-freq, --qnet in off_policy args
# fmt: on
