# FastMind LM

FastMind LM is a hyper-optimized, high-performance Language Model architecture written in Rust. It utilizes a state-of-the-art hybrid RWKV-SSM (Mamba-style) recurrent architecture capable of rapid training and incredibly resilient long-context performance without the quadratic bottleneck of traditional Self-Attention.

## Features

- **Hybrid RWKV-SSM Architecture**: A cutting-edge blend of RWKV's Time Mixing (Exponential Moving Average) and SSM's Selective Scan (Mamba) ensuring linear complexity $O(N)$.
- **Blazing Fast Training**: Achieves near-optimal speed on standard hardware.
  - Multi-threaded AVX2 / `matrixmultiply` SGEMM integrations.
  - Batched operations across both time-mixing and memory scanning mechanisms parallelized natively via Rayon.
  - O(1) allocation during the autograd forward pass, keeping memory footprint incredibly minimal.
- **Custom BPE Tokenizer**: Fully parallelized $O(\log n)$ memory BPE tokenizer built from scratch capable of processing gigabyte corpora natively in seconds.
- **Dynamic Training Dashboard**: Real-time terminal stats including tokens/s throughput, batch loss trends, perplexity tracking, validation metrics, and live generated text.

## Tech Stack

- **Rust** (Standard + Rayon for natively parallel execution)
- Custom computational graph auto-differentiation engine.
- Zero dependencies on standard PyTorch / Cuda, fully stand-alone ML system executing pure BLAS optimizations via `matrixmultiply`.

## Getting Started

### Prerequisites

- Build tools & [Rust / Cargo](https://rustup.rs/)
- For maximum speed, a modern CPU with AVX2/FMA features is recommended.

### Train Corpus Preparation

Put your dataset inside `data/train.txt` and `data/val.txt`.

### Step 1: Tokenize

Create the native vocabulary before training:

```sh
cargo run --release -- tokenize --train-corpus data/train.txt --vocab-size 16384 --output tokenizer_16k.bpe
```

### Step 2: Training

Execute training using the `fast` configuration:

```sh
cargo run --release -- train --config configs/fast.toml --data data/train.txt
```

### Step 3: Benchmarks

Check your baseline performance relative to theoretical CPU limits:

```sh
cargo run --release -- benchmark
```

## Structure

- `configs/` - Model definitions and training hyperparameters (TOML files).
- `data/` - Corpus and binary training datasets (`.bin` generated automatically to eliminate tokenization overhead during loading).
- `ckpts/` - Training checkpoints generated dynamically, rotated locally.
- `src/` - Entire FastMind codebase including Tensors, Datasets, and Model topology.

## License

MIT License
