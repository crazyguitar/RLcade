#!/bin/bash
# DQN + IMPALA ResNet (Double DQN with dueling architecture)
# Usage: bash examples/dqn/dqn_resnet.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
#
# ResNet encoder options (env vars):
#   RESNET_CHANNELS  comma-separated channel sizes per stage (default: 16,32,32)
#   RESNET_OUT_DIM   output feature dimension (default: 256)
set -x

COMMON_ARGS=(
  --agent dqn
  --env "${ENV:-rlcade/SuperMarioBros-v0}"
  --rom "${ROM:-games/super-mario-bros.nes}"
  --actions "${ACTIONS:-complex}"
  --encoder resnet
  --resnet-channels "${RESNET_CHANNELS:-16,32,32}"
  --resnet-out-dim "${RESNET_OUT_DIM:-256}"
  --checkpoint "${CHECKPOINT:-checkpoints/dqn_resnet_smb.pt}"
)

TRAIN=false
LAUNCHER=elastic
NPROC=1
NNODES=1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --train)
    TRAIN=true
    shift
    ;;
  --launcher)
    LAUNCHER="$2"
    shift 2
    ;;
  --nproc-per-node)
    NPROC="$2"
    shift 2
    ;;
  --nnodes)
    NNODES="$2"
    shift 2
    ;;
  *)
    ARGS+=("$1")
    shift
    ;;
  esac
done

if $TRAIN; then
  mkdir -p checkpoints
  python -m rlcade.training \
    --launcher "$LAUNCHER" \
    --nproc-per-node "$NPROC" \
    --nnodes "$NNODES" \
    "${COMMON_ARGS[@]}" \
    --distributed ddp \
    --lr "${LR:-1e-4}" \
    --gamma "${GAMMA:-0.99}" \
    --tau "${TAU:-1e-3}" \
    --epsilon-start "${EPSILON_START:-1.0}" \
    --epsilon-end "${EPSILON_END:-0.01}" \
    --epsilon-decay "${EPSILON_DECAY:-50000000}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --buffer-size "${BUFFER_SIZE:-100000}" \
    --learn-start "${LEARN_START:-10000}" \
    --learn-freq "${LEARN_FREQ:-4}" \
    --double \
    --custom-reward \
    --max-iterations "${MAX_ITERATIONS:-1000000}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL:-10000}" \
    --checkpoint-path "${CHECKPOINT:-checkpoints/dqn_resnet_smb.pt}" \
    --safetensors-path "${SAFETENSORS:-checkpoints/dqn_resnet_smb.safetensors}" \
    --eval-interval "${EVAL_INTERVAL:-10000}" \
    --eval-episodes "${EVAL_EPISODES:-5}" \
    --log-interval "${LOG_INTERVAL:-1000}" \
    --tensorboard "${TB_LOG:-tensorboard/dqn_resnet_smb}" \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
