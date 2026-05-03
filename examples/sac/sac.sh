#!/bin/bash
# SAC-Discrete + CNN
#
# Off-policy entropy-regularized actor-critic with automatic temperature tuning.
# Uses dual Q-networks and a categorical policy over discrete actions.
#
# Paper: Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings"
#        https://arxiv.org/abs/1910.07207
#
# Usage: bash examples/sac/sac.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
set -x

COMMON_ARGS=(
  --agent sac
  --env rlcade/SuperMarioBros-v0
  --rom games/super-mario-bros.nes
  --actions complex
  --encoder cnn
  --checkpoint checkpoints/sac_smb.pt
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
    --lr-actor 3e-4 \
    --lr-critic 3e-4 \
    --lr-alpha 3e-4 \
    --init-alpha 0.2 \
    --gamma 0.99 \
    --tau 5e-3 \
    --batch-size 128 \
    --buffer-size 500000 \
    --learn-start 10000 \
    --learn-freq 4 \
    --custom-reward \
    --max-iterations 1000000 \
    --checkpoint-interval 10000 \
    --checkpoint-path checkpoints/sac_smb.pt \
    --eval-interval 10000 \
    --eval-episodes 5 \
    --log-interval 1000 \
    --tensorboard tensorboard/sac_smb \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
