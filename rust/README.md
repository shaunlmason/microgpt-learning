# microgpt-rust

The most atomic way to train and inference a GPT in pure Rust. Translated from [@karpathy's microgpt.py](https://github.com/karpathy/microgpt).

## Requirements

- Rust 1.70+ (install via [rustup](https://rustup.rs/))

## Build

```bash
# Debug build (faster compilation, slower runtime)
cargo build

# Release build (slower compilation, optimized runtime)
cargo build --release
```

## Run Training

The training binary downloads the names dataset and trains a character-level GPT to generate names.

```bash
# Run training (release mode recommended)
cargo run --release

# Or run the binary directly after building
./target/release/microgpt
```

Expected output:

```bash
num docs: 32033
vocab size: 27
num params: 4064
step    1 /  500 | loss 3.3630
step    2 /  500 | loss 3.2917
...
step  500 /  500 | loss 2.0859

--- inference ---
sample  1: kalia
sample  2: ameli
...
```

## Run Benchmarks

The benchmark binary outputs JSON for cross-language comparison:

```bash
# Run benchmarks
cargo run --release --bin benchmark

# Output is JSON to stdout
cargo run --release --bin benchmark 2>/dev/null | jq .
```

## Project Structure

```bash
rust/
├── Cargo.toml          # Package configuration
├── README.md           # This file
└── src/
    ├── lib.rs          # Core library (MersenneTwister, Value, GPT)
    ├── main.rs         # Training binary
    └── benchmark.rs    # Benchmark binary
```

## Architecture

The implementation follows the same structure as the Python original:

| Component | Description |
| ----------- | ------------- |
| `MersenneTwister` | PRNG matching Python's `random` module exactly |
| `Value` | Autograd scalar with forward/backward pass |
| `linear`, `softmax`, `rmsnorm` | Neural network primitives |
| `gpt` | GPT-2 style forward pass (single token, with KV cache) |
| `StateDict` | Model weights storage |

## Configuration

Default hyperparameters (matching other implementations):

```rust
vocab_size: 27      // 26 letters + BOS token
n_embd: 16          // embedding dimension
n_head: 4           // attention heads
n_layer: 1          // transformer layers
block_size: 8       // context length
learning_rate: 1e-2 // with cosine decay
```

## Performance

Rust provides excellent performance, especially for compute-bound operations:

| Benchmark | Rust | Python | Speedup |
| ----------- | ------ | -------- | --------- |
| random_1m | ~3ms | ~32ms | ~10x |
| gauss_100k | ~1ms | ~19ms | ~15x |
| value_backward_1k | ~0.5ms | ~4ms | ~8x |

Run `./benchmarks/run.sh` from the project root to compare all implementations.
