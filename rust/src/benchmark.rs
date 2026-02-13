//! Benchmark script for microgpt - JSON output format.
//!
//! Outputs structured JSON to stdout for cross-language comparison.
//! All benchmarks include a warmup phase before timing.

use microgpt::{
    create_state_dict, get_params, gpt, softmax, Config, KVCache, MersenneTwister, Value,
};
use serde::Serialize;
use std::time::Instant;

// Configuration
const SEED: u32 = 42;
const N_EMBD: usize = 16;
const N_HEAD: usize = 4;
const N_LAYER: usize = 1;
const BLOCK_SIZE: usize = 8;
const VOCAB_SIZE: usize = 27;
const HEAD_DIM: usize = N_EMBD / N_HEAD;

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    description: String,
    iterations: u32,
    time_ms: f64,
    ops_per_sec: f64,
}

#[derive(Serialize)]
struct BenchmarkOutput {
    language: String,
    version: String,
    timestamp: String,
    benchmarks: Vec<BenchmarkResult>,
}

fn benchmark<F>(
    name: &str,
    description: &str,
    iterations: u32,
    warmup_iterations: u32,
    mut f: F,
) -> BenchmarkResult
where
    F: FnMut(),
{
    // Warmup phase
    for _ in 0..warmup_iterations {
        f();
    }

    // Timed phase
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkResult {
        name: name.to_string(),
        description: description.to_string(),
        iterations,
        time_ms: (elapsed_ms * 100.0).round() / 100.0,
        ops_per_sec: if elapsed_ms > 0.0 {
            ((iterations as f64 / (elapsed_ms / 1000.0)) * 100.0).round() / 100.0
        } else {
            0.0
        },
    }
}

fn get_config() -> Config {
    Config {
        vocab_size: VOCAB_SIZE,
        n_embd: N_EMBD,
        n_head: N_HEAD,
        n_layer: N_LAYER,
        block_size: BLOCK_SIZE,
        head_dim: HEAD_DIM,
    }
}

fn get_rust_version() -> String {
    // Get rustc version
    match std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            version
                .split_whitespace()
                .nth(1)
                .unwrap_or("unknown")
                .to_string()
        }
        Err(_) => "unknown".to_string(),
    }
}

fn get_timestamp() -> String {
    // Simple ISO 8601 timestamp
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();

    // Convert to approximate ISO format (good enough for benchmarks)
    let days_since_epoch = secs / 86400;
    let years = 1970 + days_since_epoch / 365;
    let remaining_days = days_since_epoch % 365;
    let month = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hour = (secs % 86400) / 3600;
    let minute = (secs % 3600) / 60;
    let second = secs % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, month, day, hour, minute, second
    )
}

fn main() {
    let mut results = BenchmarkOutput {
        language: "rust".to_string(),
        version: get_rust_version(),
        timestamp: get_timestamp(),
        benchmarks: vec![],
    };

    // Benchmark 1: MersenneTwister.random() - 1M calls
    results
        .benchmarks
        .push(benchmark("random_1m", "1M random() calls", 1, 1, || {
            let mut rng = MersenneTwister::new(SEED);
            for _ in 0..1_000_000 {
                rng.random();
            }
        }));

    // Benchmark 2: MersenneTwister.gauss() - 100K calls
    results
        .benchmarks
        .push(benchmark("gauss_100k", "100K gauss() calls", 1, 1, || {
            let mut rng = MersenneTwister::new(SEED);
            for _ in 0..100_000 {
                rng.gauss(0.0, 1.0);
            }
        }));

    // Benchmark 3: Value forward - 10K chained operations
    results.benchmarks.push(benchmark(
        "value_forward_10k",
        "10K chained Value operations",
        1,
        1,
        || {
            for i in 0..10_000 {
                let v = Value::new(i as f64);
                let _ = v.add_scalar(1.0).mul_scalar(2.0).pow(2.0).relu();
            }
        },
    ));

    // Benchmark 4: Value backward - 1K passes
    results.benchmarks.push(benchmark(
        "value_backward_1k",
        "1K backward passes",
        1,
        1,
        || {
            for _ in 0..1_000 {
                let x = Value::new(2.0);
                let y = Value::new(3.0);
                let z = x.mul(&y).add(&x.pow(2.0)).log();
                z.backward();
            }
        },
    ));

    // Benchmark 5: GPT forward - 10 passes
    let config = get_config();
    results.benchmarks.push(benchmark(
        "gpt_forward_10",
        "10 GPT forward passes",
        1,
        1,
        || {
            let mut rng = MersenneTwister::new(SEED);
            let state_dict = create_state_dict(&mut rng, &config);
            for _ in 0..10 {
                let mut kv_cache = KVCache::new(N_LAYER);
                gpt(0, 0, &mut kv_cache, &state_dict, &config);
            }
        },
    ));

    // Benchmark 6: Training step - 1 complete step
    results.benchmarks.push(benchmark(
        "training_step_1",
        "1 complete training step",
        1,
        1,
        || {
            let mut rng = MersenneTwister::new(SEED);
            let state_dict = create_state_dict(&mut rng, &config);
            let params = get_params(&state_dict);

            let learning_rate = 1e-2;
            let beta1 = 0.9;
            let beta2 = 0.95;
            let eps_adam = 1e-8;
            let mut m: Vec<f64> = vec![0.0; params.len()];
            let mut v: Vec<f64> = vec![0.0; params.len()];

            // Simple test document "test"
            let doc = "test";
            let bos = 26usize;
            let uchars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
            let mut tokens: Vec<usize> = vec![bos];
            for ch in doc.chars() {
                tokens.push(uchars.iter().position(|&c| c == ch).unwrap());
            }
            tokens.push(bos);
            let n = BLOCK_SIZE.min(tokens.len() - 1);

            let mut kv_cache = KVCache::new(N_LAYER);
            let mut losses = Vec::new();

            for pos_id in 0..n {
                let token_id = tokens[pos_id];
                let target_id = tokens[pos_id + 1];
                let logits = gpt(token_id, pos_id, &mut kv_cache, &state_dict, &config);
                let probs = softmax(&logits);
                let loss_t = probs[target_id].log().neg();
                losses.push(loss_t);
            }

            let mut loss = Value::new(0.0);
            for l in &losses {
                loss = loss.add(l);
            }
            loss = loss.div_scalar(n as f64);
            loss.backward();

            // Adam optimizer step
            let lr_t = learning_rate * 0.5 * (1.0 + 0.0_f64.cos()); // cos(0) = 1
            for (i, p) in params.iter().enumerate() {
                let grad = p.grad();
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
                let m_hat = m[i] / (1.0 - beta1);
                let v_hat = v[i] / (1.0 - beta2);
                p.set_data(p.data() - lr_t * m_hat / (v_hat.sqrt() + eps_adam));
                p.set_grad(0.0);
            }
        },
    ));

    // Output JSON to stdout
    let json = serde_json::to_string_pretty(&results).unwrap();
    println!("{}", json);
}
