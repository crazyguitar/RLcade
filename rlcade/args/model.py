"""Agent/model argument group."""

# fmt: off
def add_model_args(parser):
    parser.add_argument("--agent", type=str, default="ppo", help="Agent type")
    parser.add_argument("--actor", type=str, default="actor", help="Actor network name")
    parser.add_argument("--critic", type=str, default="critic", help="Critic network name")
    parser.add_argument("--qnet", type=str, default="qnet", help="Q-network name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto, cpu, cuda, cuda:0, ...)")
    # Encoder selection
    parser.add_argument("--encoder", type=str, default="cnn", help="Encoder type (cnn, lstm, resnet)")
    # LSTM encoder options
    parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--lstm-layers", type=int, default=1, help="Number of LSTM layers")
    # ResNet encoder options
    parser.add_argument("--resnet-channels", type=str, default="16,32,32", help="Comma-separated channel sizes per stage")
    parser.add_argument("--resnet-out-dim", type=int, default=256, help="ResNet output feature dimension")
    # ICM (Intrinsic Curiosity Module)
    parser.add_argument("--icm", action="store_true", help="Enable ICM intrinsic curiosity")
    parser.add_argument("--icm-coef", type=float, default=0.01, help="ICM intrinsic reward scaling")
    parser.add_argument("--icm-feature-dim", type=int, default=256, help="ICM feature embedding dimension")
    # torch.compile
    parser.add_argument("--eager", action="store_true", help="Disable torch.compile and use eager mode")
# fmt: on
