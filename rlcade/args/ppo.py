"""PPO hyperparameter argument group."""

# fmt: off
def add_ppo_args(parser):
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout steps per iteration")
# fmt: on
