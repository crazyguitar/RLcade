"""Rainbow DQN hyperparameter argument group.

Shared args (--tau, --buffer-size, --learn-start, --learn-freq, --qnet)
are defined in dqn.py and reused by Rainbow DQN. This file only adds
Rainbow-specific parameters.
"""

# fmt: off
def add_rainbow_dqn_args(parser):
    parser.add_argument("--alpha", type=float, default=0.6, help="PER alpha")
    parser.add_argument("--beta-start", type=float, default=0.4, help="PER beta start")
    parser.add_argument("--beta-end", type=float, default=1.0, help="PER beta end")
    parser.add_argument("--prior-eps", type=float, default=1e-6, help="PER minimum priority")
    parser.add_argument("--num-atoms", type=int, default=51, help="C51 number of atoms")
    parser.add_argument("--v-min", type=float, default=-200.0, help="C51 support minimum")
    parser.add_argument("--v-max", type=float, default=200.0, help="C51 support maximum")
    parser.add_argument("--noise-std", type=float, default=0.5, help="NoisyNet std init")
    parser.add_argument("--n-step", type=int, default=3, help="N-step return")
# fmt: on
