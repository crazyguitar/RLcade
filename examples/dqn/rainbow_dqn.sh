#!/bin/bash
# Rainbow DQN (C51 + PER + NoisyNet + Dueling + Double + N-step)
# Usage: bash examples/dqn/rainbow_dqn.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
set -x

COMMON_ARGS=(
  --agent rainbow_dqn
  --env rlcade/SuperMarioBros-v0
  --rom games/super-mario-bros.nes
  --actions complex
  --qnet rainbow_qnet
  --encoder cnn
  --checkpoint checkpoints/rainbow_dqn_smb.pt
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
    --lr 6.25e-5 \
    --gamma 0.99 \
    --tau 1e-3 \
    --batch-size 64 \
    --buffer-size 100000 \
    --learn-start 10000 \
    --learn-freq 4 \
    --alpha 0.6 \
    --beta-start 0.4 \
    --beta-end 1.0 \
    --num-atoms 51 \
    --v-min -200 \
    --v-max 200 \
    --noise-std 0.5 \
    --n-step 3 \
    --custom-reward \
    --max-iterations 1000000 \
    --checkpoint-interval 10000 \
    --checkpoint-path checkpoints/rainbow_dqn_smb.pt \
    --eval-interval 10000 \
    --eval-episodes 5 \
    --log-interval 1000 \
    --tensorboard tensorboard/rainbow_dqn_smb \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
