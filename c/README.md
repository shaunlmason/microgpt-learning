# microgpt-c

The most atomic way to train and inference a GPT in pure C. Translated from [@karpathy's microgpt.py](https://github.com/karpathy/microgpt).

## Build

```bash
gcc -O3 -o microgpt microgpt.c -lm
```

## Run Tests

```bash
# Build and run test suite
gcc -O3 -o test_microgpt test_microgpt.c -lm
./test_microgpt
```

Tests verify:
- Mersenne Twister PRNG matches Python's random module
- Value operations (add, mul, pow)
- Neural network primitives (softmax, rmsnorm, linear)

## Run Benchmarks

```bash
# Run benchmark with JSON output
./microgpt
```

Or from the project root:

```bash
./benchmarks/run.sh c
```

## Implementation

The C implementation includes:

- **Mersenne Twister PRNG**: Matches Python's random module exactly
- **Value class**: Autograd with pre-allocated value pool
- **GPT model**: Full implementation with:
  - Multi-head attention with KV cache
  - RMSNorm layer normalization
  - MLP with ReLU² activation
  - Token and position embeddings
- **Training**: Forward pass with cross-entropy loss and Adam optimizer step

## Performance

C dominates all benchmarks:

| Benchmark | C | Rust | Python | Speedup vs Python |
|-----------|------:|------:|--------:|------------------:|
| random_1m | ~5ms | ~3ms | ~31ms | ~6x |
| gauss_100k | ~2ms | ~1ms | ~19ms | ~10x |
| value_forward_10k | ~0.6ms | ~3ms | ~10ms | ~17x |
| value_backward_1k | ~3ms | ~0.6ms | ~4ms | ~1.3x |
| gpt_forward_10 | ~0.1ms | ~5ms | ~21ms | **~210x** |
| training_step_1 | ~0.07ms | ~5ms | ~31ms | **~440x** |

The C version achieves exceptional performance through:
- Pre-allocated memory pools (no malloc/free during benchmarks)
- Stack-allocated arrays where possible
- Compiler optimizations (O3)
- Minimal overhead from abstractions

## Architecture

The GPT model in C follows the same architecture as Python/Rust/TypeScript:

```
Input: token_id, pos_id
↓
Token Embedding + Position Embedding
↓
RMSNorm
↓
For each layer:
  ├─ Multi-Head Attention
  │   ├─ Compute Q, K, V
  │   ├─ Cache K, V
  │   ├─ Attention(Q, K_cached, V_cached)
  │   └─ Output projection
  ├─ Residual connection
  └─ MLP (ReLU²)
      ├─ FC1 (n_embd → 4*n_embd)
      ├─ ReLU² activation
      └─ FC2 (4*n_embd → n_embd)
↓
Output: logits over vocabulary
```

## Requirements

- GCC or Clang
- Standard C library (math.h, stdio.h, stdlib.h, string.h, time.h, sys/time.h)

## Test Suite

The test suite (`test_microgpt.c`) verifies correctness against Python reference values:

- ✅ PRNG random() matches Python exactly
- ✅ Softmax produces correct probabilities
- ✅ RMSNorm produces correct normalized values
- ✅ Linear layer computes correct dot products
