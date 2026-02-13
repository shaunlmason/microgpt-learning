//! The most atomic way to train and inference a GPT in pure Rust.
//! This file is the complete algorithm.
//! Everything else is just efficiency.
//! Translated from @karpathy's microgpt.py

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

// ============================================================================
// Mersenne Twister PRNG (MT19937)
// Implements the same algorithm as Python's random module
// ============================================================================

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

pub struct MersenneTwister {
    mt: [u32; N],
    mti: usize,
    gauss_next: Option<f64>,
}

impl MersenneTwister {
    pub fn new(seed: u32) -> Self {
        let mut rng = Self {
            mt: [0; N],
            mti: N + 1,
            gauss_next: None,
        };
        rng.init_by_array(&[seed]);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..N {
            let s = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (s.wrapping_mul(1812433253)).wrapping_add(i as u32);
        }
        self.mti = N;
    }

    // Python uses init_by_array even for simple integer seeds
    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19650218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(init_key.len());

        while k > 0 {
            let s = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ s.wrapping_mul(1664525))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= init_key.len() {
                j = 0;
            }
            k -= 1;
        }

        k = N - 1;
        while k > 0 {
            let s = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ s.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }

        self.mt[0] = 0x80000000; // MSB is 1; assuring non-zero initial array
    }

    fn genrand_int32(&mut self) -> u32 {
        let mag01 = [0u32, MATRIX_A];

        if self.mti >= N {
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];

            self.mti = 0;
        }

        let mut y = self.mt[self.mti];
        self.mti += 1;

        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;

        y
    }

    /// Returns a random float in [0, 1) - matches Python's random.random()
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_int32() >> 5) as f64;
        let b = (self.genrand_int32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// Returns k random bits - matches Python's random.getrandbits()
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k <= 32 {
            self.genrand_int32() >> (32 - k)
        } else {
            panic!("k must be <= 32 for this implementation");
        }
    }

    /// Returns a random int in [0, n) - matches Python's random.randrange()
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            panic!("n must be positive");
        }
        let k = 32 - n.leading_zeros(); // bit length
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// Gaussian distribution - matches Python's random.gauss()
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        if let Some(next) = self.gauss_next.take() {
            return mu + sigma * next;
        }

        loop {
            let u1 = 2.0 * self.random() - 1.0;
            let u2 = 2.0 * self.random() - 1.0;
            let r = u1 * u1 + u2 * u2;

            if r < 1.0 && r != 0.0 {
                let mult = (-2.0 * r.ln() / r).sqrt();
                self.gauss_next = Some(u2 * mult);
                return mu + sigma * u1 * mult;
            }
        }
    }

    /// Fisher-Yates shuffle - matches Python's random.shuffle()
    pub fn shuffle<T>(&mut self, array: &mut [T]) {
        for i in (1..array.len()).rev() {
            let j = self.randbelow((i + 1) as u32) as usize;
            array.swap(i, j);
        }
    }

    /// Weighted random choice - matches Python's random.choices() with k=1
    pub fn choices<T: Clone>(&mut self, population: &[T], weights: &[f64]) -> T {
        let total: f64 = weights.iter().sum();
        let r = self.random() * total;

        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                return population[i].clone();
            }
        }
        population.last().unwrap().clone()
    }
}

// ============================================================================
// Value Class (Autograd)
// ============================================================================

#[derive(Clone)]
struct ValueInner {
    data: f64,
    grad: f64,
    children: Vec<Rc<RefCell<ValueInner>>>,
    local_grads: Vec<f64>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children: vec![],
            local_grads: vec![],
        })))
    }

    fn new_with_children(data: f64, children: Vec<Value>, local_grads: Vec<f64>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children: children.iter().map(|v| v.0.clone()).collect(),
            local_grads,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn add(&self, other: &Value) -> Value {
        Value::new_with_children(
            self.data() + other.data(),
            vec![self.clone(), other.clone()],
            vec![1.0, 1.0],
        )
    }

    pub fn add_scalar(&self, other: f64) -> Value {
        let other_val = Value::new(other);
        self.add(&other_val)
    }

    pub fn mul(&self, other: &Value) -> Value {
        Value::new_with_children(
            self.data() * other.data(),
            vec![self.clone(), other.clone()],
            vec![other.data(), self.data()],
        )
    }

    pub fn mul_scalar(&self, other: f64) -> Value {
        let other_val = Value::new(other);
        self.mul(&other_val)
    }

    pub fn pow(&self, n: f64) -> Value {
        let data = self.data();
        Value::new_with_children(
            data.powf(n),
            vec![self.clone()],
            vec![n * data.powf(n - 1.0)],
        )
    }

    pub fn log(&self) -> Value {
        let data = self.data();
        Value::new_with_children(data.ln(), vec![self.clone()], vec![1.0 / data])
    }

    pub fn exp(&self) -> Value {
        let exp_val = self.data().exp();
        Value::new_with_children(exp_val, vec![self.clone()], vec![exp_val])
    }

    pub fn relu(&self) -> Value {
        let data = self.data();
        Value::new_with_children(
            data.max(0.0),
            vec![self.clone()],
            vec![if data > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    pub fn neg(&self) -> Value {
        self.mul_scalar(-1.0)
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.pow(-1.0))
    }

    pub fn div_scalar(&self, other: f64) -> Value {
        self.mul_scalar(1.0 / other)
    }

    pub fn backward(&self) {
        // Topological sort
        let mut topo: Vec<Rc<RefCell<ValueInner>>> = vec![];
        let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();

        fn build_topo(
            v: &Rc<RefCell<ValueInner>>,
            topo: &mut Vec<Rc<RefCell<ValueInner>>>,
            visited: &mut HashSet<*const RefCell<ValueInner>>,
        ) {
            let ptr = Rc::as_ptr(v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v.borrow().children {
                    build_topo(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        build_topo(&self.0, &mut topo, &mut visited);

        // Backward pass
        self.0.borrow_mut().grad = 1.0;
        for v in topo.iter().rev() {
            let v_borrowed = v.borrow();
            let grad = v_borrowed.grad;
            for (child, &local_grad) in v_borrowed
                .children
                .iter()
                .zip(v_borrowed.local_grads.iter())
            {
                child.borrow_mut().grad += local_grad * grad;
            }
        }
    }
}

// ============================================================================
// Neural Network Helpers
// ============================================================================

pub fn linear(x: &[Value], w: &[Vec<Value>]) -> Vec<Value> {
    w.iter()
        .map(|wo| {
            wo.iter()
                .zip(x.iter())
                .fold(Value::new(0.0), |sum, (wi, xi)| sum.add(&wi.mul(xi)))
        })
        .collect()
}

pub fn softmax(logits: &[Value]) -> Vec<Value> {
    let max_val = logits
        .iter()
        .map(|v| v.data())
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value> = logits
        .iter()
        .map(|v| v.add_scalar(-max_val).exp())
        .collect();
    let total = exps.iter().fold(Value::new(0.0), |sum, e| sum.add(e));
    exps.iter().map(|e| e.div(&total)).collect()
}

pub fn rmsnorm(x: &[Value]) -> Vec<Value> {
    let ms = x
        .iter()
        .fold(Value::new(0.0), |sum, xi| sum.add(&xi.mul(xi)))
        .div_scalar(x.len() as f64);
    let scale = ms.add_scalar(1e-5).pow(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

// ============================================================================
// Parameter Initialization
// ============================================================================

pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
    pub head_dim: usize,
}

pub fn matrix(rng: &mut MersenneTwister, nout: usize, nin: usize, std: f64) -> Vec<Vec<Value>> {
    (0..nout)
        .map(|_| (0..nin).map(|_| Value::new(rng.gauss(0.0, std))).collect())
        .collect()
}

pub struct StateDict {
    pub wte: Vec<Vec<Value>>,
    pub wpe: Vec<Vec<Value>>,
    pub lm_head: Vec<Vec<Value>>,
    pub layers: Vec<LayerWeights>,
}

pub struct LayerWeights {
    pub attn_wq: Vec<Vec<Value>>,
    pub attn_wk: Vec<Vec<Value>>,
    pub attn_wv: Vec<Vec<Value>>,
    pub attn_wo: Vec<Vec<Value>>,
    pub mlp_fc1: Vec<Vec<Value>>,
    pub mlp_fc2: Vec<Vec<Value>>,
}

pub fn create_state_dict(rng: &mut MersenneTwister, config: &Config) -> StateDict {
    let mut layers = Vec::new();
    for _ in 0..config.n_layer {
        layers.push(LayerWeights {
            attn_wq: matrix(rng, config.n_embd, config.n_embd, 0.02),
            attn_wk: matrix(rng, config.n_embd, config.n_embd, 0.02),
            attn_wv: matrix(rng, config.n_embd, config.n_embd, 0.02),
            attn_wo: matrix(rng, config.n_embd, config.n_embd, 0.0),
            mlp_fc1: matrix(rng, 4 * config.n_embd, config.n_embd, 0.02),
            mlp_fc2: matrix(rng, config.n_embd, 4 * config.n_embd, 0.0),
        });
    }

    StateDict {
        wte: matrix(rng, config.vocab_size, config.n_embd, 0.02),
        wpe: matrix(rng, config.block_size, config.n_embd, 0.02),
        lm_head: matrix(rng, config.vocab_size, config.n_embd, 0.02),
        layers,
    }
}

pub fn get_params(state_dict: &StateDict) -> Vec<Value> {
    let mut params = Vec::new();

    for row in &state_dict.wte {
        params.extend(row.iter().cloned());
    }
    for row in &state_dict.wpe {
        params.extend(row.iter().cloned());
    }
    for row in &state_dict.lm_head {
        params.extend(row.iter().cloned());
    }

    for layer in &state_dict.layers {
        for row in &layer.attn_wq {
            params.extend(row.iter().cloned());
        }
        for row in &layer.attn_wk {
            params.extend(row.iter().cloned());
        }
        for row in &layer.attn_wv {
            params.extend(row.iter().cloned());
        }
        for row in &layer.attn_wo {
            params.extend(row.iter().cloned());
        }
        for row in &layer.mlp_fc1 {
            params.extend(row.iter().cloned());
        }
        for row in &layer.mlp_fc2 {
            params.extend(row.iter().cloned());
        }
    }

    params
}

// ============================================================================
// GPT Forward Pass
// ============================================================================

pub struct KVCache {
    pub keys: Vec<Vec<Vec<Value>>>,   // [layer][position][dim]
    pub values: Vec<Vec<Vec<Value>>>, // [layer][position][dim]
}

impl KVCache {
    pub fn new(n_layer: usize) -> Self {
        KVCache {
            keys: (0..n_layer).map(|_| Vec::new()).collect(),
            values: (0..n_layer).map(|_| Vec::new()).collect(),
        }
    }
}

pub fn gpt(
    token_id: usize,
    pos_id: usize,
    kv_cache: &mut KVCache,
    state_dict: &StateDict,
    config: &Config,
) -> Vec<Value> {
    // Token and position embeddings
    let tok_emb = &state_dict.wte[token_id];
    let pos_emb = &state_dict.wpe[pos_id];
    let mut x: Vec<Value> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| t.add(p))
        .collect();
    x = rmsnorm(&x);

    for li in 0..config.n_layer {
        let layer = &state_dict.layers[li];

        // 1) Multi-head attention block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &layer.attn_wq);
        let k = linear(&x, &layer.attn_wk);
        let v = linear(&x, &layer.attn_wv);
        kv_cache.keys[li].push(k);
        kv_cache.values[li].push(v);

        let mut x_attn: Vec<Value> = Vec::new();
        for h in 0..config.n_head {
            let hs = h * config.head_dim;
            let q_h: Vec<Value> = q[hs..hs + config.head_dim].to_vec();
            let k_h: Vec<&[Value]> = kv_cache.keys[li]
                .iter()
                .map(|ki| &ki[hs..hs + config.head_dim])
                .collect();
            let v_h: Vec<&[Value]> = kv_cache.values[li]
                .iter()
                .map(|vi| &vi[hs..hs + config.head_dim])
                .collect();

            // Attention logits
            let attn_logits: Vec<Value> = k_h
                .iter()
                .map(|k_t| {
                    let dot = q_h
                        .iter()
                        .zip(k_t.iter())
                        .fold(Value::new(0.0), |sum, (q_j, k_j)| sum.add(&q_j.mul(k_j)));
                    dot.div_scalar((config.head_dim as f64).sqrt())
                })
                .collect();

            let attn_weights = softmax(&attn_logits);

            // Weighted sum of values
            for j in 0..config.head_dim {
                let sum = attn_weights
                    .iter()
                    .zip(v_h.iter())
                    .fold(Value::new(0.0), |sum, (w, v_t)| sum.add(&w.mul(&v_t[j])));
                x_attn.push(sum);
            }
        }

        x = linear(&x_attn, &layer.attn_wo);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &layer.mlp_fc1);
        x = x.iter().map(|xi| xi.relu().pow(2.0)).collect();
        x = linear(&x, &layer.mlp_fc2);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();
    }

    linear(&x, &state_dict.lm_head)
}
