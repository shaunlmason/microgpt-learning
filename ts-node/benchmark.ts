#!/usr/bin/env npx tsx
/**
 * Benchmark script for microgpt.ts - JSON output format.
 *
 * Outputs structured JSON to stdout for cross-language comparison.
 * All benchmarks include a warmup phase before timing.
 */

import {
  Config,
  MersenneTwister,
  Value,
  createStateDict,
  getParams,
  gpt,
  softmax,
} from "./microgpt.js";

// Configuration
const SEED = 42;
const N_EMBD = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 8;
const VOCAB_SIZE = 27;

interface BenchmarkResult {
  name: string;
  description: string;
  iterations: number;
  time_ms: number;
  ops_per_sec: number;
}

interface BenchmarkOutput {
  language: string;
  version: string;
  timestamp: string;
  benchmarks: BenchmarkResult[];
}

function benchmark(
  name: string,
  description: string,
  iterations: number,
  fn: () => void,
  warmupIterations: number = 1
): BenchmarkResult {
  // Warmup phase
  for (let i = 0; i < warmupIterations; i++) {
    fn();
  }

  // Timed phase
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const elapsedMs = performance.now() - start;

  return {
    name,
    description,
    iterations,
    time_ms: Math.round(elapsedMs * 100) / 100,
    ops_per_sec:
      elapsedMs > 0 ? Math.round((iterations / (elapsedMs / 1000)) * 100) / 100 : 0,
  };
}

function getNodeVersion(): string {
  try {
    return process.version.replace(/^v/, "");
  } catch {
    return "unknown";
  }
}

function getConfig(): Config {
  return {
    vocabSize: VOCAB_SIZE,
    nEmbd: N_EMBD,
    nHead: N_HEAD,
    nLayer: N_LAYER,
    blockSize: BLOCK_SIZE,
    headDim: N_EMBD / N_HEAD,
  };
}

function main(): void {
  const results: BenchmarkOutput = {
    language: "typescript",
    version: getNodeVersion(),
    timestamp: new Date().toISOString(),
    benchmarks: [],
  };

  // Benchmark 1: MersenneTwister.random() - 1M calls
  results.benchmarks.push(
    benchmark(
      "random_1m",
      "1M random() calls",
      1,
      () => {
        const rng = new MersenneTwister(SEED);
        for (let i = 0; i < 1_000_000; i++) {
          rng.random();
        }
      },
      1
    )
  );

  // Benchmark 2: MersenneTwister.gauss() - 100K calls
  results.benchmarks.push(
    benchmark(
      "gauss_100k",
      "100K gauss() calls",
      1,
      () => {
        const rng = new MersenneTwister(SEED);
        for (let i = 0; i < 100_000; i++) {
          rng.gauss(0, 1);
        }
      },
      1
    )
  );

  // Benchmark 3: Value forward - 10K chained operations
  results.benchmarks.push(
    benchmark(
      "value_forward_10k",
      "10K chained Value operations",
      1,
      () => {
        for (let i = 0; i < 10_000; i++) {
          new Value(i).add(1).mul(2).pow(2).relu();
        }
      },
      1
    )
  );

  // Benchmark 4: Value backward - 1K passes
  results.benchmarks.push(
    benchmark(
      "value_backward_1k",
      "1K backward passes",
      1,
      () => {
        for (let i = 0; i < 1_000; i++) {
          const x = new Value(2);
          const y = new Value(3);
          const z = x.mul(y).add(x.pow(2)).log();
          z.backward();
        }
      },
      1
    )
  );

  // Benchmark 5: GPT forward - 10 passes
  const config = getConfig();
  results.benchmarks.push(
    benchmark(
      "gpt_forward_10",
      "10 GPT forward passes",
      1,
      () => {
        const rng = new MersenneTwister(SEED);
        const sd = createStateDict(rng, config);
        for (let i = 0; i < 10; i++) {
          const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
          const values: Value[][][] = Array.from({ length: N_LAYER }, () => []);
          gpt(0, 0, keys, values, sd, config);
        }
      },
      1
    )
  );

  // Benchmark 6: Training step - 1 complete step
  results.benchmarks.push(
    benchmark(
      "training_step_1",
      "1 complete training step",
      1,
      () => {
        const rng = new MersenneTwister(SEED);
        const sd = createStateDict(rng, config);
        const params = getParams(sd);

        const learningRate = 1e-2;
        const beta1 = 0.9;
        const beta2 = 0.95;
        const epsAdam = 1e-8;
        const m: number[] = new Array(params.length).fill(0);
        const v: number[] = new Array(params.length).fill(0);

        // Simple test document
        const doc = "test";
        const BOS = 26;
        const uchars = "abcdefghijklmnopqrstuvwxyz".split("");
        const tokens = [BOS, ...doc.split("").map((ch) => uchars.indexOf(ch)), BOS];
        const n = Math.min(BLOCK_SIZE, tokens.length - 1);

        const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
        const values: Value[][][] = Array.from({ length: N_LAYER }, () => []);
        const losses: Value[] = [];

        for (let posId = 0; posId < n; posId++) {
          const tokenId = tokens[posId];
          const targetId = tokens[posId + 1];
          const logits = gpt(tokenId, posId, keys, values, sd, config);
          const probs = softmax(logits);
          const lossT = probs[targetId].log().neg();
          losses.push(lossT);
        }

        const loss = losses.reduce((sum, l) => sum.add(l), new Value(0)).div(n);
        loss.backward();

        // Adam optimizer step
        const lrT = learningRate * 0.5 * (1 + Math.cos(0));
        for (let i = 0; i < params.length; i++) {
          const p = params[i];
          m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
          v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
          const mHat = m[i] / (1 - beta1);
          const vHat = v[i] / (1 - beta2);
          p.data -= (lrT * mHat) / (Math.sqrt(vHat) + epsAdam);
          p.grad = 0;
        }
      },
      1
    )
  );

  // Output JSON to stdout
  console.log(JSON.stringify(results, null, 2));
}

main();
