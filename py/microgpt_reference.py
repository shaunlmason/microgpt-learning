#!/usr/bin/env python3
"""
Generate reference values from microgpt.py for TypeScript cross-validation.
Outputs: reference_values.json

This script does NOT modify the original microgpt.py - it captures values
by running the same algorithms with controlled inputs.
"""

import json
import math
import random
import os

# ========== PRNG Reference Values ==========
# Capture random values before any other code runs


def capture_prng_reference():
    """Capture PRNG sequences from Python's random module."""
    ref = {}

    # random.random() sequence
    random.seed(42)
    ref["prng_random_100"] = [random.random() for _ in range(100)]

    # random.gauss() sequence
    random.seed(42)
    ref["prng_gauss_50"] = [random.gauss(0, 1) for _ in range(50)]

    # random.shuffle() result
    random.seed(42)
    test_list = list(range(100))
    random.shuffle(test_list)
    ref["prng_shuffle_100"] = test_list

    # random.choices() with weights
    random.seed(42)
    ref["prng_choices_100"] = [
        random.choices([0, 1, 2, 3], weights=[1, 2, 3, 4])[0] for _ in range(100)
    ]

    return ref


# ========== Value Class Reference ==========
# We need to define Value class here to capture reference values
# This is a copy of the Value class from microgpt.py


class Value:
    """Stores a single scalar value and its gradient, as a node in a computation graph."""

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def capture_value_reference():
    """Capture Value class forward/backward reference values."""
    ref = {}

    # Basic operations
    ref["value_add"] = {"a": 2.0, "b": 3.0, "result": Value(2.0) + Value(3.0)}
    ref["value_add"]["result"] = ref["value_add"]["result"].data

    ref["value_mul"] = {"a": 2.0, "b": 3.0, "result": (Value(2.0) * Value(3.0)).data}
    ref["value_sub"] = {"a": 5.0, "b": 3.0, "result": (Value(5.0) - Value(3.0)).data}
    ref["value_div"] = {"a": 6.0, "b": 2.0, "result": (Value(6.0) / Value(2.0)).data}
    ref["value_pow"] = {"a": 2.0, "n": 3.0, "result": (Value(2.0) ** 3).data}
    ref["value_neg"] = {"a": 5.0, "result": (-Value(5.0)).data}
    ref["value_log"] = {"a": math.e, "result": Value(math.e).log().data}
    ref["value_exp"] = {"a": 1.0, "result": Value(1.0).exp().data}
    ref["value_relu_pos"] = {"a": 5.0, "result": Value(5.0).relu().data}
    ref["value_relu_neg"] = {"a": -5.0, "result": Value(-5.0).relu().data}

    # Complex expression: ((x + y) * z) ** 2
    x, y, z = Value(2.0), Value(3.0), Value(4.0)
    expr = ((x + y) * z) ** 2
    expr.backward()
    ref["value_complex_expr"] = {
        "x": 2.0,
        "y": 3.0,
        "z": 4.0,
        "result": expr.data,
        "x_grad": x.grad,
        "y_grad": y.grad,
        "z_grad": z.grad,
    }

    # Fan-out test: x used twice in x * x
    x = Value(3.0)
    expr = x * x
    expr.backward()
    ref["value_fanout"] = {
        "x": 3.0,
        "result": expr.data,
        "x_grad": x.grad,
    }

    # Chain rule test: (x * y) ** 2
    x, y = Value(2.0), Value(3.0)
    expr = (x * y) ** 2
    expr.backward()
    ref["value_chain_rule"] = {
        "x": 2.0,
        "y": 3.0,
        "result": expr.data,
        "x_grad": x.grad,
        "y_grad": y.grad,
    }

    # More complex: log(exp(x) + y) * z
    x, y, z = Value(1.0), Value(2.0), Value(3.0)
    expr = (x.exp() + y).log() * z
    expr.backward()
    ref["value_log_exp_chain"] = {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "result": expr.data,
        "x_grad": x.grad,
        "y_grad": y.grad,
        "z_grad": z.grad,
    }

    return ref


# ========== Helper Functions Reference ==========


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def capture_helper_reference():
    """Capture helper function reference values."""
    ref = {}

    # Linear
    x_vec = [Value(1.0), Value(2.0), Value(3.0)]
    w_mat = [[Value(1.0), Value(0.0), Value(0.0)], [Value(0.0), Value(1.0), Value(0.0)]]
    out = linear(x_vec, w_mat)
    ref["linear_identity"] = {"x": [1.0, 2.0, 3.0], "result": [v.data for v in out]}

    # Linear with non-trivial weights
    x_vec = [Value(1.0), Value(2.0), Value(3.0)]
    w_mat = [[Value(1.0), Value(2.0), Value(3.0)], [Value(4.0), Value(5.0), Value(6.0)]]
    out = linear(x_vec, w_mat)
    ref["linear_weighted"] = {
        "x": [1.0, 2.0, 3.0],
        "w": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "result": [v.data for v in out],
    }

    # Softmax
    logits = [Value(1.0), Value(2.0), Value(3.0)]
    probs = softmax(logits)
    ref["softmax_123"] = {"logits": [1.0, 2.0, 3.0], "result": [p.data for p in probs]}

    # Softmax with large values (numerical stability)
    logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
    probs = softmax(logits)
    ref["softmax_large"] = {
        "logits": [1000.0, 1001.0, 1002.0],
        "result": [p.data for p in probs],
    }

    # RMSNorm
    x_vec = [Value(1.0), Value(2.0), Value(3.0)]
    normed = rmsnorm(x_vec)
    ref["rmsnorm_123"] = {"x": [1.0, 2.0, 3.0], "result": [v.data for v in normed]}

    # RMSNorm with same values
    x_vec = [Value(5.0), Value(5.0), Value(5.0)]
    normed = rmsnorm(x_vec)
    ref["rmsnorm_same"] = {"x": [5.0, 5.0, 5.0], "result": [v.data for v in normed]}

    return ref


# ========== Parameter Initialization Reference ==========


def capture_matrix_reference():
    """Capture matrix initialization reference values."""
    ref = {}

    random.seed(42)
    # Generate a 3x4 matrix with default std=0.02
    matrix_values = [[random.gauss(0, 0.02) for _ in range(4)] for _ in range(3)]
    ref["matrix_3x4_std002"] = matrix_values

    random.seed(42)
    # Generate a 2x2 matrix with std=0 (all zeros effectively, but gauss still called)
    matrix_values = [[random.gauss(0, 0) for _ in range(2)] for _ in range(2)]
    ref["matrix_2x2_std0"] = matrix_values

    return ref


# ========== GPT Forward Pass Reference ==========


def capture_gpt_reference():
    """Capture GPT forward pass reference values with fixed initialization."""
    ref = {}

    # Set up minimal config
    n_embd = 16
    n_head = 4
    n_layer = 1
    block_size = 8
    head_dim = n_embd // n_head
    vocab_size = 27  # For names dataset (26 letters + BOS)

    random.seed(42)

    # Initialize state_dict (same as microgpt.py)
    def matrix(nout, nin, std=0.02):
        return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd, std=0)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd, std=0)

    params = [p for mat in state_dict.values() for row in mat for p in row]

    # Capture initial params
    ref["initial_params_20"] = [p.data for p in params[:20]]
    ref["num_params"] = len(params)

    # Define gpt function (same as microgpt.py)
    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict["wte"][token_id]
        pos_emb = state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs : hs + head_dim]
                k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() ** 2 for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict["lm_head"])
        return logits

    # Forward pass with token_id=0, pos_id=0
    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    logits = gpt(0, 0, keys_cache, values_cache)
    ref["gpt_logits_t0_p0"] = [l.data for l in logits]

    # Forward pass with token_id=5, pos_id=3
    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    # First fill cache for positions 0, 1, 2
    for p in range(3):
        gpt(p, p, keys_cache, values_cache)
    logits = gpt(5, 3, keys_cache, values_cache)
    ref["gpt_logits_t5_p3"] = [l.data for l in logits]

    return ref


# ========== Training Reference ==========


def capture_training_reference():
    """Capture training loop reference values."""
    ref = {}

    # Check if input.txt exists, download if not
    if not os.path.exists("input.txt"):
        import urllib.request

        names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
        urllib.request.urlretrieve(names_url, "input.txt")

    random.seed(42)

    # Load and prepare data (same as microgpt.py)
    docs = [
        l.strip() for l in open("input.txt").read().strip().split("\n") if l.strip()
    ]
    random.shuffle(docs)

    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1

    ref["vocab_size"] = vocab_size
    ref["uchars"] = uchars
    ref["num_docs"] = len(docs)
    ref["first_doc_after_shuffle"] = docs[0]

    # Config
    n_embd = 16
    n_head = 4
    n_layer = 1
    block_size = 8
    head_dim = n_embd // n_head

    # Initialize
    def matrix(nout, nin, std=0.02):
        return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd, std=0)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd, std=0)

    params = [p for mat in state_dict.values() for row in mat for p in row]

    # gpt function (same as above)
    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict["wte"][token_id]
        pos_emb = state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs : hs + head_dim]
                k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() ** 2 for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict["lm_head"])
        return logits

    # Adam optimizer
    learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    num_steps = 5
    losses_captured = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys_cache, values_cache = (
            [[] for _ in range(n_layer)],
            [[] for _ in range(n_layer)],
        )
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys_cache, values_cache)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        losses_captured.append(loss.data)

        # Capture step 1 details
        if step == 0:
            ref["step1_loss"] = loss.data
            ref["step1_logits_first"] = [l.data for l in logits]  # Last position logits

        loss.backward()

        if step == 0:
            ref["step1_gradients_20"] = [p.grad for p in params[:20]]

        lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

    ref["losses_5_steps"] = losses_captured
    ref["step5_params_20"] = [p.data for p in params[:20]]

    return ref


# ========== Main ==========


def main():
    print("Generating reference values for TypeScript cross-validation...")

    reference = {}

    print("  - PRNG reference...")
    reference.update(capture_prng_reference())

    print("  - Value class reference...")
    reference.update(capture_value_reference())

    print("  - Helper functions reference...")
    reference.update(capture_helper_reference())

    print("  - Matrix initialization reference...")
    reference.update(capture_matrix_reference())

    print("  - GPT forward pass reference...")
    reference.update(capture_gpt_reference())

    print("  - Training loop reference...")
    reference.update(capture_training_reference())

    with open("reference_values.json", "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nReference values written to reference_values.json")
    print(f"Total keys: {len(reference)}")


if __name__ == "__main__":
    main()
