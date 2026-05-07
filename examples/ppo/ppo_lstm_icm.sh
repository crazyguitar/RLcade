#!/bin/bash
# PPO + LSTM + ICM (intrinsic curiosity)
# Usage: bash examples/ppo/ppo_lstm_icm.sh [--train] [--launcher BACKEND] [--nproc-per-node N] [--nnodes N] [extra args...]
set -x

COMMON_ARGS=(
  --agent lstm_ppo
  --env rlcade/SuperMarioBros-v0
  --rom games/super-mario-bros.nes
  --actions complex
  --lstm-hidden 256
  --icm
  --icm-coef 0.01
  --checkpoint checkpoints/ppo_lstm_icm_smb.pt
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
    --lr 2.5e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.2 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --custom-reward \
    --lr-schedule \
    --num-steps 2048 \
    --batch-size 256 \
    --update-epochs 4 \
    --max-iterations 10000 \
    --checkpoint-interval 100 \
    --checkpoint-path checkpoints/ppo_lstm_icm_smb.pt \
    --safetensors-path checkpoints/ppo_lstm_icm_smb.safetensors \
    --eval-interval 100 \
    --eval-episodes 5 \
    --tensorboard tensorboard/ppo_lstm_icm_smb \
    "${ARGS[@]}"
else
  python -m rlcade.agent "${COMMON_ARGS[@]}" \
    --render-mode human \
    "${ARGS[@]}"
fi
