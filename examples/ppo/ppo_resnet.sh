#!/bin/bash
# PPO + IMPALA ResNet encoder
# Usage: bash examples/ppo/ppo_resnet.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
#
# ResNet encoder options (env vars):
#   RESNET_CHANNELS  comma-separated channel sizes per stage (default: 16,32,32)
#   RESNET_OUT_DIM   output feature dimension (default: 256)
#
# AMP + Gradient Accumulation examples:
#   bash examples/ppo/ppo_resnet.sh --train --amp
#   bash examples/ppo/ppo_resnet.sh --train --grad-accum-steps 4
#   bash examples/ppo/ppo_resnet.sh --train --amp --grad-accum-steps 4
set -x

COMMON_ARGS=(
  --agent ppo
  --env "${ENV:-rlcade/SuperMarioBros-v0}"
  --rom "${ROM:-games/super-mario-bros.nes}"
  --actions "${ACTIONS:-complex}"
  --encoder resnet
  --resnet-channels "${RESNET_CHANNELS:-16,32,32}"
  --resnet-out-dim "${RESNET_OUT_DIM:-256}"
  --checkpoint "${CHECKPOINT:-checkpoints/ppo_resnet_smb.pt}"
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
    --lr "${LR:-2.5e-4}" \
    --gamma "${GAMMA:-0.99}" \
    --gae-lambda "${GAE_LAMBDA:-0.95}" \
    --clip-coef "${CLIP_COEF:-0.2}" \
    --ent-coef "${ENT_COEF:-0.01}" \
    --vf-coef "${VF_COEF:-0.5}" \
    --max-grad-norm "${MAX_GRAD_NORM:-0.5}" \
    --custom-reward \
    --lr-schedule \
    --num-steps "${NUM_STEPS:-2048}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --update-epochs "${UPDATE_EPOCHS:-4}" \
    --max-iterations "${MAX_ITERATIONS:-10000}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL:-100}" \
    --checkpoint-path "${CHECKPOINT:-checkpoints/ppo_resnet_smb.pt}" \
    --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-episodes "${EVAL_EPISODES:-5}" \
    --tensorboard "${TB_LOG:-tensorboard/ppo_resnet_smb}" \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
