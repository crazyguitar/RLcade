"""Training argument group."""

# fmt: off
def add_training_args(parser):
    parser.add_argument("--max-iterations", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint-path", type=str, default="ppo_smb.pt", help="Checkpoint file path")
    parser.add_argument("--async-checkpoint", action="store_true", help="Offload checkpoint writes to a background thread")
    parser.add_argument("--safetensors-path", type=str, default="ppo_smb.safetensors", help="Safetensors export file path (written once at training end; empty to disable)")
    parser.add_argument("--target-score", type=float, default=None, help="Early stop target score")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N iterations (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--lr-schedule", action="store_true", help="Enable linear LR decay")
    parser.add_argument("--tensorboard", type=str, default=None, help="TensorBoard log directory (enables TB logging)")
    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (progressive stage unlock)")
    parser.add_argument("--curriculum-initial", type=int, default=4, help="Initial number of stages")
    parser.add_argument("--curriculum-threshold", type=float, default=500.0, help="Mean score to unlock next stages")
    parser.add_argument("--curriculum-expand", type=int, default=4, help="Stages to add per expansion")
    # Distributed training
    parser.add_argument("--distributed", type=str, default=None, choices=["ddp", "fsdp2"], help="Distributed strategy")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Distributed backend")
    # Mixed precision & gradient accumulation
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Split each batch into N micro-batches to reduce peak GPU memory while keeping effective batch size unchanged")
    # Host/device transfer optimization (enabled by default)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable pinned CPU buffers for H2D/D2H transfers")
# fmt: on
