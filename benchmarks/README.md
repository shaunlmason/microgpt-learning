# microgpt Benchmark Suite

A language-agnostic benchmarking framework for comparing microgpt implementations across different programming languages.

## Quick Start

```bash
# Run all benchmarks
./benchmarks/run.sh

# Run specific language only
./benchmarks/run.sh python
./benchmarks/run.sh typescript
./benchmarks/run.sh bun
./benchmarks/run.sh rust

# List available benchmarks
./benchmarks/run.sh --list

# View results
cat benchmarks/results/comparison.md
```

## Requirements

- **jq**: Required for JSON parsing in the comparison script

  ```bash
  # macOS
  brew install jq

  # Ubuntu/Debian
  apt-get install jq
  ```

- **Language-specific**: Each language needs its runtime installed (Python 3, Node.js, etc.)

## Adding a New Language

To add benchmarks for a new language (e.g., Go, Zig):

### 1. Create the benchmark script

Create a benchmark script in your language's directory that outputs JSON to **stdout** in this exact format:

```json
{
  "language": "rust",
  "version": "1.75.0",
  "timestamp": "2026-02-13T10:30:00Z",
  "benchmarks": [
    {
      "name": "random_1m",
      "description": "1M random() calls",
      "iterations": 1,
      "time_ms": 45.23,
      "ops_per_sec": 22109.44
    },
    {
      "name": "gauss_100k",
      "description": "100K gauss() calls",
      "iterations": 1,
      "time_ms": 89.12,
      "ops_per_sec": 1122.08
    },
    {
      "name": "value_forward_10k",
      "description": "10K chained Value operations",
      "iterations": 1,
      "time_ms": 12.34,
      "ops_per_sec": 810372.77
    },
    {
      "name": "value_backward_1k",
      "description": "1K backward passes",
      "iterations": 1,
      "time_ms": 23.45,
      "ops_per_sec": 42643.92
    },
    {
      "name": "gpt_forward_10",
      "description": "10 GPT forward passes",
      "iterations": 1,
      "time_ms": 567.89,
      "ops_per_sec": 17.61
    },
    {
      "name": "training_step_1",
      "description": "1 complete training step",
      "iterations": 1,
      "time_ms": 1234.56,
      "ops_per_sec": 0.81
    }
  ]
}
```

### 2. Register the benchmark

Edit `benchmarks/run.sh` and add your language to the `BENCHMARKS` array:

```bash
declare -a BENCHMARKS=(
    "python|python3 benchmark_json.py|py"
    "typescript|npx tsx benchmark.ts|ts-node"
    "rust|cargo run --release --bin benchmark|rs"     # <-- Add your language
)
```

Format: `"name|command|working_directory"`

- **name**: Language identifier (used for output file naming)
- **command**: Command to run the benchmark
- **working_directory**: Directory relative to project root where the command runs

### 3. Run and verify

```bash
./benchmarks/run.sh your-language
cat benchmarks/results/your-language.json
```

## Benchmark Specifications

All implementations must run these **standardized benchmarks** with consistent parameters:

| Benchmark | Name | Description | Parameters |
| ----------- | ------ | ------------- | ------------ |
| 1 | `random_1m` | Random number generation | 1,000,000 calls to `random()` |
| 2 | `gauss_100k` | Gaussian random generation | 100,000 calls to `gauss(0, 1)` |
| 3 | `value_forward_10k` | Autograd forward pass | 10,000 iterations: `v = ((v + 1) * 2)^2.relu()` |
| 4 | `value_backward_1k` | Autograd backward pass | 1,000 iterations: `z = (x*y + x^2).log(); z.backward()` |
| 5 | `gpt_forward_10` | GPT forward inference | 10 forward passes with config below |
| 6 | `training_step_1` | Complete training step | Forward + backward + Adam update |

### GPT Configuration

Use these parameters for GPT benchmarks:

```bash
SEED = 42
N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 8
VOCAB_SIZE = 27
```

### Best Practices

1. **Warmup**: Run each benchmark once before timing to warm up JIT/caches
2. **Seed**: Use seed `42` for reproducibility where applicable
3. **Timing**: Use high-resolution timers (`time.perf_counter()`, `performance.now()`)
4. **Output**: Print only the JSON result to stdout; use stderr for any progress messages

## Output Files

Results are saved to `benchmarks/results/`:

```bash
benchmarks/results/
├── python.json       # Python benchmark results
├── typescript.json   # TypeScript/Node.js benchmark results
├── bun.json          # TypeScript/Bun benchmark results
├── rust.json         # Rust benchmark results
├── comparison.md     # Generated comparison table
└── .gitkeep
```

The `comparison.md` file is auto-generated when you run `./benchmarks/run.sh`.

## Example Output

```markdown
# Benchmark Comparison

| Benchmark | python | typescript | Ratio |
|-----------|-------:|-----------:|------:|
| random_1m | 45.23ms | 23.12ms | 1.96x (typescript) |
| gauss_100k | 89.12ms | 45.67ms | 1.95x (typescript) |
| value_forward_10k | 123.45ms | 98.76ms | 1.25x (typescript) |
| value_backward_1k | 234.56ms | 187.65ms | 1.25x (typescript) |
| gpt_forward_10 | 567.89ms | 432.10ms | 1.31x (typescript) |
| training_step_1 | 1234.56ms | 987.65ms | 1.25x (typescript) |
```

## Troubleshooting

### "jq: command not found"

Install jq for JSON parsing:

```bash
brew install jq  # macOS
apt-get install jq  # Ubuntu/Debian
```

### Invalid JSON output

Ensure your benchmark script:

- Outputs **only** JSON to stdout
- Uses stderr for any progress/debug messages
- Includes all required fields in the schema

### Benchmark not found

Check that:

- The working directory exists
- The command is correct
- Any build steps completed (e.g., `cargo build --release`)
