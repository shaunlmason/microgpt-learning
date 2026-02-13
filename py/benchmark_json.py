#!/usr/bin/env python3
"""
Benchmark script for microgpt.py - JSON output format.

Outputs structured JSON to stdout for cross-language comparison.
All benchmarks include a warmup phase before timing.

NOTE: This file is self-contained and does not import from microgpt.py
because that file runs training at module load time.
"""

import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone

# Configuration
SEED = 42
N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 8
VOCAB_SIZE = 27
HEAD_DIM = N_EMBD // N_HEAD


# ============================================================================
# Value class (autograd) - copied from microgpt.py
# ============================================================================
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


# ============================================================================
# Neural network helpers - copied from microgpt.py
# ============================================================================
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


def matrix(nout, nin, std=0.02):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


# ============================================================================
# GPT model - state dict and forward pass
# ============================================================================
def create_state_dict():
    """Initialize the GPT state dictionary."""
    state_dict = {
        "wte": matrix(VOCAB_SIZE, N_EMBD),
        "wpe": matrix(BLOCK_SIZE, N_EMBD),
        "lm_head": matrix(VOCAB_SIZE, N_EMBD),
    }
    for i in range(N_LAYER):
        state_dict[f"layer{i}.attn_wq"] = matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wk"] = matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wv"] = matrix(N_EMBD, N_EMBD)
        state_dict[f"layer{i}.attn_wo"] = matrix(N_EMBD, N_EMBD, std=0)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * N_EMBD, N_EMBD)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(N_EMBD, 4 * N_EMBD, std=0)
    return state_dict


# Global state dict for benchmarks (initialized lazily)
_state_dict = None


def get_state_dict():
    global _state_dict
    if _state_dict is None:
        random.seed(SEED)
        _state_dict = create_state_dict()
    return _state_dict


def gpt(token_id, pos_id, keys, values, state_dict):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(N_LAYER):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs : hs + HEAD_DIM]
            k_h = [ki[hs : hs + HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs : hs + HEAD_DIM] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


# ============================================================================
# Benchmark utilities
# ============================================================================
def benchmark(
    name: str, description: str, iterations: int, fn, warmup_iterations: int = 1
):
    """Run a benchmark with warmup and return results."""
    # Warmup phase
    for _ in range(warmup_iterations):
        fn()

    # Timed phase
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "name": name,
        "description": description,
        "iterations": iterations,
        "time_ms": round(elapsed_ms, 2),
        "ops_per_sec": round(iterations / (elapsed_ms / 1000), 2)
        if elapsed_ms > 0
        else 0,
    }


def main():
    results = {
        "language": "python",
        "version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": [],
    }

    # Benchmark 1: random.random() - 1M calls
    def random_1m():
        random.seed(SEED)
        for _ in range(1_000_000):
            random.random()

    results["benchmarks"].append(
        benchmark("random_1m", "1M random() calls", 1, random_1m, warmup_iterations=1)
    )

    # Benchmark 2: random.gauss() - 100K calls
    def gauss_100k():
        random.seed(SEED)
        for _ in range(100_000):
            random.gauss(0, 1)

    results["benchmarks"].append(
        benchmark(
            "gauss_100k", "100K gauss() calls", 1, gauss_100k, warmup_iterations=1
        )
    )

    # Benchmark 3: Value forward - 10K chained operations
    def value_forward():
        for i in range(10_000):
            v = Value(i)
            v = v + 1
            v = v * 2
            v = v**2
            v = v.relu()

    results["benchmarks"].append(
        benchmark(
            "value_forward_10k",
            "10K chained Value operations",
            1,
            value_forward,
            warmup_iterations=1,
        )
    )

    # Benchmark 4: Value backward - 1K passes
    def value_backward():
        for _ in range(1_000):
            x = Value(2)
            y = Value(3)
            z = (x * y + x**2).log()
            z.backward()

    results["benchmarks"].append(
        benchmark(
            "value_backward_1k",
            "1K backward passes",
            1,
            value_backward,
            warmup_iterations=1,
        )
    )

    # Benchmark 5: GPT forward - 10 passes
    def gpt_forward():
        random.seed(SEED)
        state_dict = create_state_dict()
        for _ in range(10):
            keys = [[] for _ in range(N_LAYER)]
            values = [[] for _ in range(N_LAYER)]
            gpt(0, 0, keys, values, state_dict)

    results["benchmarks"].append(
        benchmark(
            "gpt_forward_10",
            "10 GPT forward passes",
            1,
            gpt_forward,
            warmup_iterations=1,
        )
    )

    # Benchmark 6: Training step - 1 complete step
    def training_step():
        random.seed(SEED)
        state_dict = create_state_dict()
        params = [p for mat in state_dict.values() for row in mat for p in row]

        doc = "test"
        tokens = [26] + [ord(c) % 26 for c in doc] + [26]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state_dict)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

    results["benchmarks"].append(
        benchmark(
            "training_step_1",
            "1 complete training step",
            1,
            training_step,
            warmup_iterations=1,
        )
    )

    # Output JSON to stdout
    json.dump(results, sys.stdout, indent=2)
    print()  # Trailing newline


if __name__ == "__main__":
    main()
