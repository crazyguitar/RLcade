#!/bin/bash
# SAC-Discrete + IMPALA ResNet
#
# Off-policy entropy-regularized actor-critic with automatic temperature tuning.
# Uses dual Q-networks and a categorical policy over discrete actions.
#
# Usage: bash examples/sac/sac_resnet.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
#
# ResNet encoder options (env vars):
#   RESNET_CHANNELS  comma-separated channel sizes per stage (default: 16,32,32)
#   RESNET_OUT_DIM   output feature dimension (default: 256)
set -x

COMMON_ARGS=(
  --agent sac
  --env "${ENV:-rlcade/SuperMarioBros-v0}"
  --rom "${ROM:-games/super-mario-bros.nes}"
  --actions "${ACTIONS:-complex}"
  --encoder resnet
  --resnet-channels "${RESNET_CHANNELS:-16,32,32}"
  --resnet-out-dim "${RESNET_OUT_DIM:-256}"
  --checkpoint "${CHECKPOINT:-checkpoints/sac_resnet_smb.pt}"
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
    --lr-actor "${LR_ACTOR:-3e-4}" \
    --lr-critic "${LR_CRITIC:-3e-4}" \
    --lr-alpha "${LR_ALPHA:-3e-4}" \
    --init-alpha "${INIT_ALPHA:-0.2}" \
    --gamma "${GAMMA:-0.99}" \
    --tau "${TAU:-5e-3}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --buffer-size "${BUFFER_SIZE:-500000}" \
    --learn-start "${LEARN_START:-10000}" \
    --learn-freq "${LEARN_FREQ:-4}" \
    --custom-reward \
    --max-iterations "${MAX_ITERATIONS:-1000000}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL:-10000}" \
    --checkpoint-path "${CHECKPOINT:-checkpoints/sac_resnet_smb.pt}" \
    --safetensors-path "${SAFETENSORS:-checkpoints/sac_resnet_smb.safetensors}" \
    --async-checkpoint \
    --eval-interval "${EVAL_INTERVAL:-10000}" \
    --eval-episodes "${EVAL_EPISODES:-5}" \
    --log-interval "${LOG_INTERVAL:-1000}" \
    --tensorboard "${TB_LOG:-tensorboard/sac_resnet_smb}" \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
