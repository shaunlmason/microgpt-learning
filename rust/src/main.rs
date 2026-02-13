//! The most atomic way to train and inference a GPT in pure Rust.
//! This is the main training binary.

use microgpt::{create_state_dict, get_params, gpt, softmax, Config, KVCache, MersenneTwister};
use std::f64::consts::PI;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    let mut rng = MersenneTwister::new(42);

    // Load or download input dataset
    let input_path = "input.txt";
    if !Path::new(input_path).exists() {
        println!("Downloading input.txt...");
        let names_url =
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";

        // Simple HTTP download (requires network)
        match std::process::Command::new("curl")
            .args(["-o", input_path, names_url])
            .output()
        {
            Ok(_) => println!("Downloaded input.txt"),
            Err(e) => {
                eprintln!("Failed to download: {}. Please download manually.", e);
                std::process::exit(1);
            }
        }
    }

    // Parse documents
    let file = fs::File::open(input_path).expect("Failed to open input.txt");
    let reader = BufReader::new(file);
    let mut docs: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    // Tokenizer
    let mut uchars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect();
    uchars.sort();
    uchars.dedup();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    // Configuration
    let config = Config {
        vocab_size,
        n_embd: 16,
        n_head: 4,
        n_layer: 1,
        block_size: 8,
        head_dim: 4, // n_embd / n_head
    };

    // Initialize state dict
    let state_dict = create_state_dict(&mut rng, &config);
    let params = get_params(&state_dict);
    println!("num params: {}", params.len());

    // Adam optimizer buffers
    let learning_rate = 1e-2;
    let beta1 = 0.9;
    let beta2 = 0.95;
    let eps_adam = 1e-8;
    let mut m: Vec<f64> = vec![0.0; params.len()];
    let mut v: Vec<f64> = vec![0.0; params.len()];

    // Training loop
    let num_steps = 500;
    for step in 0..num_steps {
        // Take single document, tokenize it
        let doc = &docs[step % docs.len()];
        let mut tokens: Vec<usize> = vec![bos];
        for ch in doc.chars() {
            tokens.push(uchars.iter().position(|&c| c == ch).unwrap());
        }
        tokens.push(bos);
        let n = config.block_size.min(tokens.len() - 1);

        // Forward pass
        let mut kv_cache = KVCache::new(config.n_layer);
        let mut losses = Vec::new();

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(token_id, pos_id, &mut kv_cache, &state_dict, &config);
            let probs = softmax(&logits);
            let loss_t = probs[target_id].log().neg();
            losses.push(loss_t);
        }

        // Average loss
        let mut loss = microgpt::Value::new(0.0);
        for l in &losses {
            loss = loss.add(l);
        }
        loss = loss.div_scalar(n as f64);

        // Backward pass
        loss.backward();

        // Adam optimizer update
        let lr_t = learning_rate * 0.5 * (1.0 + (PI * step as f64 / num_steps as f64).cos());
        for (i, p) in params.iter().enumerate() {
            let grad = p.grad();
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
            let m_hat = m[i] / (1.0 - beta1.powi((step + 1) as i32));
            let v_hat = v[i] / (1.0 - beta2.powi((step + 1) as i32));
            p.set_data(p.data() - lr_t * m_hat / (v_hat.sqrt() + eps_adam));
            p.set_grad(0.0);
        }

        println!(
            "step {:4} / {:4} | loss {:.4}",
            step + 1,
            num_steps,
            loss.data()
        );
    }

    // Inference
    let temperature = 0.5;
    println!("\n--- inference ---");
    for sample_idx in 0..20 {
        let mut kv_cache = KVCache::new(config.n_layer);
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..config.block_size {
            let logits = gpt(token_id, pos_id, &mut kv_cache, &state_dict, &config);
            let scaled_logits: Vec<microgpt::Value> =
                logits.iter().map(|l| l.div_scalar(temperature)).collect();
            let probs = softmax(&scaled_logits);
            let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            let population: Vec<usize> = (0..vocab_size).collect();
            token_id = rng.choices(&population, &weights);
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}
