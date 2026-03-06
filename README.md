# Hikki (Hybrid RWKV-SSM Architecture)

Hikki is a hyper-optimized language model implementation in Rust, combining the best of RWKV (Linear Attention) and SSM (Mamba-style) architectures. Designed for extreme performance on consumer CPUs and NVIDIA GPUs.

## Key Features

- **Hybrid Architecture**: Melds RWKV Time-Mixing with selective scanning (SSM) for O(N) complexity.
- **High Performance**:
  - **CPU**: Optimized with `matrixmultiply` SGEMM, AVX2/FMA intrinsics, and Rayon multi-threading.
  - **GPU**: CUDA support with cuBLAS for high-speed matrix operations and custom kernels.
- **Efficiency**: O(1) memory allocation in the autograd graph and zero-copy dataset loading.
- **Custom BPE**: Includes a high-speed parallel BPE tokenizer.

## Installation & Build

### Prerequisites

- [Rust & Cargo](https://rustup.rs/)
- (Optional) CUDA Toolkit for GPU support.

### Build

Default CPU-optimized build:

```bash
cargo build --release
```

Build with CUDA support:

```bash
cargo build --release --features cuda
```

## Usage

### 1. Tokenization

Train the tokenizer on your corpus:

```bash
cargo run --release -- tokenize --train-corpus data/train.txt --vocab-size 16384 --output tokenizer_16k.bpe
```

### 2. Training

Run the training process with the optimized "hikki" configuration:

```bash
cargo run --release -- train --config configs/hikki.toml --data data/train.txt --tokenizer tokenizer_16k.bpe
```

### 3. Benchmarking

Verify your hardware performance:

```bash
cargo run --release -- benchmark
```

## Directory Structure

- `src/`: Core implementation (tensor ops, model architecture, trainer).
- `configs/`: TOML configuration files for different model sizes.

## License

MIT
