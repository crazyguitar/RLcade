"""Shared arguments for off-policy agents (DQN, Rainbow DQN, SAC)."""

# fmt: off
def add_off_policy_args(parser):
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft target update rate")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--learn-start", type=int, default=10000, help="Steps before learning starts")
    parser.add_argument("--learn-freq", type=int, default=4, help="Learn every N steps")
# fmt: on
