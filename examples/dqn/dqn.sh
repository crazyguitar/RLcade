#!/bin/bash
# DQN + CNN (Double DQN with dueling architecture)
# Usage: bash examples/dqn/dqn.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
set -x

COMMON_ARGS=(
  --agent dqn
  --env rlcade/SuperMarioBros-v0
  --rom games/super-mario-bros.nes
  --actions complex
  --encoder cnn
  --checkpoint checkpoints/dqn_smb.pt
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
    --lr 1e-4 \
    --gamma 0.99 \
    --tau 1e-3 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 50000000 \
    --batch-size 64 \
    --buffer-size 100000 \
    --learn-start 10000 \
    --learn-freq 4 \
    --double \
    --custom-reward \
    --max-iterations 1000000 \
    --checkpoint-interval 10000 \
    --checkpoint-path checkpoints/dqn_smb.pt \
    --eval-interval 10000 \
    --eval-episodes 5 \
    --log-interval 1000 \
    --tensorboard tensorboard/dqn_smb \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
