#!/usr/bin/env python3
"""Benchmark script for microgpt.py"""

import time
import random

from microgpt_reference import linear, softmax, rmsnorm, Value
from microgpt import gpt, matrix

# Seed for reproducibility
random.seed(42)

print("Python microgpt.py Benchmarks")
print("=" * 50)

# Benchmark 1: random.random() calls
print("\n1. random.random(): 1M calls")
start = time.perf_counter()
for _ in range(1_000_000):
    random.random()
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

# Benchmark 2: random.gauss() calls
print("\n2. random.gauss(): 100K calls")
random.seed(42)
start = time.perf_counter()
for _ in range(100_000):
    random.gauss(0, 1)
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

# Import the Value class and functions from microgpt.py
# Read just the Value class and helper functions (before the input dataset section)
with open("./py/microgpt.py", "r") as f:
    content = f.read()

# Extract just the parts we need (Value class and helper functions)
# Split at the dataset loading section
parts = content.split("# Let there be an input")
header_and_value = parts[0]

# Execute the code to get Value class and helper functions
exec(header_and_value, globals())

# Define gpt function (it's defined after the state_dict initialization in microgpt.py)
# We need to extract it
code_parts = content.split("def gpt(token_id, pos_id, keys, values):")
gpt_code = (
    "def gpt(token_id, pos_id, keys, values):"
    + code_parts[1].split("\n# Let there be Adam")[0]
)
exec(gpt_code, globals())

# Benchmark 3: Value forward operations
print("\n3. Value forward: 10K chained operations")
random.seed(42)
start = time.perf_counter()
for i in range(10_000):
    v = Value(i)
    v = v + 1
    v = v * 2
    v = v**2
    v = v.relu()
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

# Benchmark 4: Value backward passes
print("\n4. Value backward: 1K backward passes")
random.seed(42)
start = time.perf_counter()
for _ in range(1_000):
    x = Value(2)
    y = Value(3)
    z = (x * y + x**2).log()
    z.backward()
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

# Benchmark 5: GPT forward passes
print("\n5. GPT forward: 10 forward passes")
random.seed(42)
n_embd = 16
n_head = 4
n_layer = 1
block_size = 8
head_dim = n_embd // n_head
vocab_size = 27

# Initialize state dict
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

start = time.perf_counter()
for _ in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    gpt(0, 0, keys, values)
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

# Benchmark 6: Full training step
print("\n6. Training step: 1 complete step")
random.seed(42)

# Minimal training step
doc = "test"
tokens = [26] + [ord(c) % 26 for c in doc] + [26]  # BOS + chars + BOS
n = min(block_size, len(tokens) - 1)

keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []

start = time.perf_counter()
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)

loss = (1 / n) * sum(losses)
loss.backward()
elapsed = (time.perf_counter() - start) * 1000
print(f"   Result: {elapsed:.2f}ms")

print("\n" + "=" * 50)
print("Benchmarks complete!")