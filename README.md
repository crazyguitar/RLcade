# RLcade

A PyTorch-based reinforcement learning framework for playing games. RLcade provides multiple RL algorithms (PPO, DQN, Rainbow DQN, SAC), distributed training support, and a plugin-based architecture for profiling and curriculum learning.

## Key Features

- **RL Algorithms**: [PPO](rlcade/agent/ppo.py), [DQN](rlcade/agent/dqn.py) (Double + Dueling), [Rainbow DQN](rlcade/agent/dqn.py) (C51 + PER + NoisyNet + N-step), [SAC-Discrete](rlcade/agent/sac.py)
- **Encoders**: [CNN](rlcade/modules/encoders.py), [LSTM](rlcade/modules/encoders.py), [IMPALA ResNet](rlcade/modules/encoders.py) -- pluggable via `--encoder`
- **Exploration**: [ICM](rlcade/modules/icm.py) (Intrinsic Curiosity Module) for sparse-reward environments
- **Curriculum Learning**: [Progressive stage unlock](rlcade/plugins/curriculum.py) based on agent performance
- **Distributed Training**: [DDP and FSDP2](rlcade/agent/base.py) with multiple launch backends (elastic, mp, Ray)
- **Performance**:
  - `torch.compile` + [CUDA graph capture](rlcade/graph/__init__.py) (default on CUDA, `--eager` to disable)
  - [Mixed precision (AMP)](rlcade/utils/amp.py) + gradient accumulation
  - [Pinned-memory H2D staging](rlcade/utils/pin_memory.py) for async rollout transfers (default on, `--no-pin-memory` to disable)
  - [GPU affinity](rlcade/utils/affinity.py) — NUMA-aware CPU binding for multi-GPU training
- **Profiling**: [VizTracer](rlcade/plugins/viztracer.py), [Nsight Systems](rlcade/plugins/nsys.py), and [CUDA Memory Profiler](rlcade/plugins/memory_profiler.py) integration
- **Checkpointing**:
  - Local and [S3](rlcade/checkpoint/s3.py) filesystem backends
  - [Async writes](rlcade/plugins/async_checkpoint.py) (`--async-checkpoint`) — ThreadPoolExecutor-backed, non-blocking periodic saves using PyTorch's [StateDictStager](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/_state_dict_stager.py) for CPU offload
  - Cross-strategy compatible — local/DDP/FSDP2 checkpoints are interchangeable
- **NES Emulator**: Rust core with [PyO3 bindings](crates/), supporting 5 mappers (NROM, MMC1, UxROM, CNROM, MMC3)

## Supported Mappers

| Mapper | Name  | Example Games |
|--------|-------|---------------|
| 0      | NROM  | Super Mario Bros, Donkey Kong |
| 1      | SxROM (MMC1) | The Legend of Zelda, Metroid |
| 2      | UxROM | Mega Man, Castlevania |
| 3      | CNROM | Paperboy, Gradius |
| 4      | TxROM (MMC3) | Super Mario Bros 2/3, Kirby's Adventure |

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (1.85+)
- SDL2
- Python 3.10+
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

### Install SDL2

**macOS:**

```sh
brew install sdl2
```

**Ubuntu/Debian:**

```sh
sudo apt install libsdl2-dev
```

### Install

```sh
python3 -m venv .venv
source .venv/bin/activate
make install
```

### Play with a trained agent

```sh
python -m rlcade.agent --rom <path/to/rom.nes> --checkpoint <path/to/checkpoint>
```

### Run the NES emulator standalone

```sh
cargo build --release

# Default: loads games/super-mario-bros.nes
cargo run --release

# Specify a ROM file
cargo run --release -- path/to/rom.nes
```

## Training

All training scripts follow the same pattern: `bash <script> --train` to train, omit `--train` to play with a checkpoint. Every script supports distributed training via `--launcher`, `--nproc-per-node`, and `--nnodes`, and extra args are forwarded to the trainer.

```sh
# Common flags
--train                  # Run training (omit to run inference with human rendering)
--device cuda            # Use GPU (defaults to auto: GPU if available, else CPU)
--launcher BACKEND       # Launch backend: none, elastic, mp, ray (default: elastic)
--nproc-per-node N       # Number of GPUs per node (default: 1)
--nnodes N               # Number of nodes (default: 1)
--distributed STRATEGY   # Distributed strategy: ddp, fsdp2 (default: none)
```

### Launchers

| Backend | When to use |
|---------|-------------|
| `none` | Single-process, or when an external scheduler (Slurm) pre-sets `RANK`/`WORLD_SIZE` env vars |
| `elastic` | Single or multi-node GPU training (equivalent to `torchrun --standalone`) |
| `mp` | Single-node multi-GPU via `torch.multiprocessing.spawn` |
| `ray` | Multi-node GPU training on a Ray cluster (GPU-only, auto-detects topology; `pip install 'ray[default]>=2.9'`) |

```sh
# Single GPU (launcher=none, no distributed)
bash examples/ppo/ppo.sh --train --launcher none

# 8 GPUs, single node, elastic launch + DDP
bash examples/ppo/ppo.sh --train --launcher elastic --nproc-per-node 8

# 8 GPUs, single node, mp launch + DDP
bash examples/ppo/ppo.sh --train --launcher mp --nproc-per-node 8

# Multi-node (2 nodes x 8 GPUs) via elastic
bash examples/ppo/ppo.sh --train --launcher elastic --nproc-per-node 8 --nnodes 2

# Ray cluster (auto-detect GPUs)
bash examples/ppo/ppo.sh --train --launcher ray --ray-address ray://head:6379

# FSDP2 (sharded training)
bash examples/ppo/ppo.sh --train --launcher elastic --nproc-per-node 8 --distributed fsdp2
```

### PPO

```sh
# Train PPO + CNN baseline
bash examples/ppo/ppo.sh --train

# Multi-GPU (8x H100)
bash examples/ppo/ppo.sh --train --nproc-per-node 8

# PPO + LSTM encoder
bash examples/ppo/ppo_lstm.sh --train

# PPO + LSTM + ICM (intrinsic curiosity)
bash examples/ppo/ppo_lstm_icm.sh --train

# PPO + LSTM + ICM + Curriculum learning
bash examples/ppo/ppo_lstm_icm_curriculum.sh --train

# PPO + IMPALA ResNet encoder
bash examples/ppo/ppo_resnet.sh --train
```

### DQN

```sh
# Double DQN + Dueling
bash examples/dqn/dqn.sh --train

# Rainbow DQN (C51 + PER + NoisyNet + Dueling + Double + N-step)
bash examples/dqn/rainbow_dqn.sh --train

# DQN + IMPALA ResNet encoder
bash examples/dqn/dqn_resnet.sh --train

# DQN + Curriculum learning
bash examples/dqn/dqn_curriculum.sh --train

# Rainbow DQN + Curriculum learning
bash examples/dqn/rainbow_dqn_curriculum.sh --train
```

### SAC

```sh
# SAC-Discrete (dual Q-networks + auto temperature tuning)
bash examples/sac/sac.sh --train

# SAC + IMPALA ResNet encoder
bash examples/sac/sac_resnet.sh --train
```

### Play with a trained checkpoint

Omit `--train` to run inference with human rendering:

```sh
bash examples/ppo/ppo.sh
bash examples/dqn/rainbow_dqn.sh
bash examples/sac/sac.sh
```

### Script summary

| Script | Agent | Features |
|--------|-------|----------|
| `examples/ppo/ppo.sh` | PPO | CNN encoder |
| `examples/ppo/ppo_lstm.sh` | PPO | LSTM encoder |
| `examples/ppo/ppo_lstm_icm.sh` | PPO | LSTM + ICM curiosity |
| `examples/ppo/ppo_lstm_icm_curriculum.sh` | PPO | LSTM + ICM + curriculum |
| `examples/ppo/ppo_resnet.sh` | PPO | IMPALA ResNet encoder |
| `examples/dqn/dqn.sh` | DQN | Double + Dueling |
| `examples/dqn/dqn_resnet.sh` | DQN | Double + Dueling + IMPALA ResNet |
| `examples/dqn/rainbow_dqn.sh` | Rainbow DQN | C51 + PER + NoisyNet + Dueling + Double + N-step |
| `examples/dqn/dqn_curriculum.sh` | DQN | Double + Dueling + curriculum |
| `examples/dqn/rainbow_dqn_curriculum.sh` | Rainbow DQN | Full Rainbow + curriculum |
| `examples/sac/sac.sh` | SAC | Discrete SAC + auto alpha |
| `examples/sac/sac_resnet.sh` | SAC | Discrete SAC + IMPALA ResNet |

## Benchmarks

Benchmark trainer throughput for all agents (PPO, DQN, Rainbow DQN):

```sh
# Run all benchmarks (env + all agents, 8 iterations each)
python -m bench --rom <rom>

# Benchmark a specific agent
python -m bench --rom <rom> --bench ppo
python -m bench --rom <rom> --bench dqn
python -m bench --rom <rom> --bench rainbow_dqn

# Env-only benchmark (single + vectorized step throughput)
python -m bench --rom <rom> --bench env

# Options
python -m bench --rom <rom> --bench ppo --device cuda --num-steps 256 --iterations 16
```

## Profiling

Use [VizTracer](https://github.com/gaogaotiantian/viztracer) to profile training runs. Traces are saved to the `profile/` directory.

### Profiling via benchmarks

```sh
# Profile all agents (outputs profile/trace_ppo.json, profile/trace_dqn.json, etc.)
python -m bench --rom <rom> --viztracer trace

# Profile a single agent
python -m bench --rom <rom> --bench ppo --viztracer trace
```

### Profiling via training

```sh
# Profile steps 50-60 of a training run
python -m rlcade.training \
    --viztracer profile/training.json \
    --viztracer-start 50 \
    --viztracer-end 60

# With additional VizTracer options
python -m rlcade.training --viztracer profile/training.json \
    --viztracer-start 10 \
    --viztracer-end 20 \
    --viztracer-max-stack-depth 15 \
    --viztracer-ignore-c-function \
    --viztracer-log-func-args
```

### Nsight Systems (GPU profiling)

Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) to profile CUDA kernels, memory transfers, and NVTX-annotated training steps.

```sh
# Profile steps 5-15 of a training run
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  -o nsys_trace --force-overwrite=true \
  python -m rlcade.training \
    --nsys \
    --nsys-start 5 \
    --nsys-end 15 \
    --agent ppo \
    --env rlcade/SuperMarioBros-v0 \
    --rom games/super-mario-bros.nes \
    --device cuda

# View the trace
nsys-ui nsys_trace.nsys-rep
```

Each training step appears as an NVTX range (`step_5`, `step_6`, ...) in the timeline, making it easy to correlate CUDA kernel activity with specific iterations.

### CUDA Memory Profiling

Use PyTorch's [CUDA Memory Profiler](https://pytorch.org/docs/main/torch_cuda_memory.html) to capture memory allocation history and diagnose leaks or fragmentation.

```sh
# Record memory history for steps 5-15
python -m rlcade.training \
    --memory-profiler \
    --memory-profiler-start 5 \
    --memory-profiler-end 15 \
    --agent ppo \
    --env rlcade/SuperMarioBros-v0 \
    --rom games/super-mario-bros.nes \
    --device cuda

# Custom output path and max history entries
python -m rlcade.training \
    --memory-profiler \
    --memory-profiler-start 10 \
    --memory-profiler-end 20 \
    --memory-profiler-output profile/memory.pkl \
    --memory-profiler-max-entries 200000
```

Upload the snapshot file to [pytorch.org/memory_viz](https://pytorch.org/memory_viz) for an interactive visualization of memory allocations, including stack traces and allocation timelines.

### Viewing traces

```sh
vizviewer profile/trace_ppo.json
```

This opens an interactive timeline in the browser showing call stacks, durations, and PyTorch operations. Rust code (NES emulator via PyO3) appears as opaque C-extension blocks -- you can see how long `env.step()` takes but not the internal Rust call stack. For Rust-level profiling, use `cargo flamegraph` or `samply`.

## Tests

```sh
# ROM file is not included in the repo due to licensing
# Tests are skipped when --rom is not provided
make test

# Run with a ROM file
python -m pytest tests/ -v --rom "/path/to/super-mario-bros.nes"
```

## Controls

### Player 1

| NES Button | Keyboard    |
|------------|-------------|
| D-pad      | Arrow keys  |
| A          | J           |
| B          | K           |
| Start      | M           |
| Select     | N           |

### Player 2

| NES Button | Keyboard    |
|------------|-------------|
| D-pad      | WASD        |
| A          | G           |
| B          | F           |
| Start      | Y           |
| Select     | T           |

| Action     | Key         |
|------------|-------------|
| Quit       | Escape      |

## References

- [NESDev Wiki](https://www.nesdev.org/wiki/)
- [6502 CPU Reference](https://www.nesdev.org/obelisk-6502-guide/)
- [PPU Rendering](https://www.nesdev.org/wiki/PPU_rendering)
- [Mapper List](https://www.nesdev.org/wiki/Mapper)
