"""SAC-Discrete hyperparameter argument group."""

# fmt: off
def add_sac_args(parser):
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--lr-alpha", type=float, default=3e-4, help="Entropy temperature learning rate")
    parser.add_argument("--init-alpha", type=float, default=0.2, help="Initial entropy temperature")
    parser.add_argument("--target-entropy-ratio", type=float, default=0.98, help="Target entropy as fraction of max entropy log(n_actions)")
    # --tau, --buffer-size, --learn-start, --learn-freq, --qnet in off_policy args
# fmt: on
