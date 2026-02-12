/**
 * The most atomic way to train and inference a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * Translated from @karpathy's microgpt.py
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";

// ========== Mersenne Twister PRNG (MT19937) ==========
// Implements the same algorithm as Python's random module

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

// Helper for 32-bit unsigned multiplication (JavaScript numbers are 64-bit floats)
function umul32(a: number, b: number): number {
  const ah = (a >>> 16) & 0xffff;
  const al = a & 0xffff;
  const bh = (b >>> 16) & 0xffff;
  const bl = b & 0xffff;
  const low = al * bl;
  const mid = al * bh + ah * bl;
  return ((low + ((mid & 0xffff) << 16)) >>> 0);
}

export class MersenneTwister {
  private mt: number[] = new Array(N);
  private mti: number = N + 1;

  constructor(seed: number) {
    this.initByArray([seed]);
  }

  private initGenrand(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < N; i++) {
      const s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (umul32(s, 1812433253) + i) >>> 0;
    }
    this.mti = N;
  }

  // Python uses init_by_array even for simple integer seeds
  private initByArray(initKey: number[]): void {
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(N, initKey.length);

    for (; k > 0; k--) {
      const s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ umul32(s, 1664525)) + initKey[j] + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
      if (j >= initKey.length) {
        j = 0;
      }
    }

    for (k = N - 1; k > 0; k--) {
      const s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ umul32(s, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
    }

    this.mt[0] = 0x80000000; // MSB is 1; assuring non-zero initial array
  }

  genrandInt32(): number {
    let y: number;
    const mag01 = [0, MATRIX_A];

    if (this.mti >= N) {
      let kk: number;

      for (kk = 0; kk < N - M; kk++) {
        y = (this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK);
        this.mt[kk] = this.mt[kk + M] ^ (y >>> 1) ^ mag01[y & 1];
      }
      for (; kk < N - 1; kk++) {
        y = (this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK);
        this.mt[kk] = this.mt[kk + (M - N)] ^ (y >>> 1) ^ mag01[y & 1];
      }
      y = (this.mt[N - 1] & UPPER_MASK) | (this.mt[0] & LOWER_MASK);
      this.mt[N - 1] = this.mt[M - 1] ^ (y >>> 1) ^ mag01[y & 1];

      this.mti = 0;
    }

    y = this.mt[this.mti++];

    // Tempering
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;

    return y >>> 0;
  }

  // Returns a random float in [0, 1) - matches Python's random.random()
  random(): number {
    const a = this.genrandInt32() >>> 5;
    const b = this.genrandInt32() >>> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  // Returns k random bits - matches Python's random.getrandbits()
  getrandbits(k: number): number {
    if (k <= 0) {
      throw new Error("number of bits must be positive");
    }
    
    // For k <= 32, use one 32-bit value
    if (k <= 32) {
      return this.genrandInt32() >>> (32 - k);
    }
    
    // For larger k, combine multiple 32-bit values
    const numWords = Math.floor((k + 31) / 32);
    let result = 0;
    for (let i = 0; i < numWords; i++) {
      result = (result << 32) | this.genrandInt32();
    }
    // Trim to k bits
    return result >>> (numWords * 32 - k);
  }

  // Returns a random int in [0, n) - matches Python's random.randrange()
  randbelow(n: number): number {
    if (n <= 0) {
      throw new Error("n must be positive");
    }
    
    const k = n.toString(2).length; // bit length
    let r: number;
    
    // Rejection sampling to get uniform distribution
    do {
      r = this.getrandbits(k);
    } while (r >= n);
    
    return r;
  }

  // Gaussian distribution - matches Python's random.gauss()
  // Python uses the Box-Muller transform with caching
  private gaussNext: number | null = null;

  gauss(mu: number = 0, sigma: number = 1): number {
    // Python's random.gauss uses Box-Muller with caching
    // It generates pairs and caches the second value
    
    if (this.gaussNext !== null) {
      const result = this.gaussNext;
      this.gaussNext = null;
      return mu + sigma * result;
    }
    
    let u1: number, u2: number, r: number;
    
    do {
      u1 = this.random();
      u2 = this.random();
      // Convert to [-1, 1] range
      u1 = 2 * u1 - 1;
      u2 = 2 * u2 - 1;
      r = u1 * u1 + u2 * u2;
    } while (r >= 1 || r === 0);
    
    const mult = Math.sqrt(-2.0 * Math.log(r) / r);
    this.gaussNext = u2 * mult;
    return mu + sigma * u1 * mult;
  }

  // Fisher-Yates shuffle - matches Python's random.shuffle()
  shuffle<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      // Python uses randbelow which generates int in [0, i]
      const j = this.randbelow(i + 1);
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  // Weighted random choice - matches Python's random.choices() with k=1
  choices<T>(population: T[], weights: number[]): T {
    // Compute cumulative weights
    const cumWeights: number[] = [];
    let total = 0;
    for (const w of weights) {
      total += w;
      cumWeights.push(total);
    }
    
    // Pick a random point
    const r = this.random() * total;
    
    // Binary search would be faster, but linear matches Python's behavior for small arrays
    for (let i = 0; i < cumWeights.length; i++) {
      if (r < cumWeights[i]) {
        return population[i];
      }
    }
    return population[population.length - 1];
  }
}

// ========== Value Class (Autograd) ==========

export class Value {
  data: number;
  grad: number = 0;
  private _children: Value[];
  private _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this._children = children;
    this._localGrads = localGrads;
  }

  private static wrap(v: Value | number): Value {
    return v instanceof Value ? v : new Value(v);
  }

  add(other: Value | number): Value {
    const o = Value.wrap(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = Value.wrap(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(n: number): Value {
    return new Value(
      Math.pow(this.data, n),
      [this],
      [n * Math.pow(this.data, n - 1)]
    );
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const expVal = Math.exp(this.data);
    return new Value(expVal, [this], [expVal]);
  }

  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0]
    );
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(Value.wrap(other).neg());
  }

  div(other: Value | number): Value {
    return this.mul(Value.wrap(other).pow(-1));
  }

  backward(): void {
    // Topological sort
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };

    buildTopo(this);

    // Backward pass
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// ========== Neural Network Helpers ==========

export function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((wo) =>
    wo.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), new Value(0))
  );
}

export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((sum, e) => sum.add(e), new Value(0));
  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[]): Value[] {
  const ms = x.reduce((sum, xi) => sum.add(xi.mul(xi)), new Value(0)).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// ========== Parameter Initialization ==========

export interface Config {
  vocabSize: number;
  nEmbd: number;
  nHead: number;
  nLayer: number;
  blockSize: number;
  headDim: number;
}

export function matrix(
  rng: MersenneTwister,
  nout: number,
  nin: number,
  std: number = 0.02
): Value[][] {
  const result: Value[][] = [];
  for (let i = 0; i < nout; i++) {
    const row: Value[] = [];
    for (let j = 0; j < nin; j++) {
      row.push(new Value(rng.gauss(0, std)));
    }
    result.push(row);
  }
  return result;
}

export function createStateDict(
  rng: MersenneTwister,
  config: Config
): Record<string, Value[][]> {
  const stateDict: Record<string, Value[][]> = {
    wte: matrix(rng, config.vocabSize, config.nEmbd),
    wpe: matrix(rng, config.blockSize, config.nEmbd),
    lm_head: matrix(rng, config.vocabSize, config.nEmbd),
  };

  for (let i = 0; i < config.nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(rng, config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(rng, config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wv`] = matrix(rng, config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.attn_wo`] = matrix(rng, config.nEmbd, config.nEmbd, 0);
    stateDict[`layer${i}.mlp_fc1`] = matrix(rng, 4 * config.nEmbd, config.nEmbd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(rng, config.nEmbd, 4 * config.nEmbd, 0);
  }

  return stateDict;
}

export function getParams(stateDict: Record<string, Value[][]>): Value[] {
  const params: Value[] = [];
  for (const mat of Object.values(stateDict)) {
    for (const row of mat) {
      for (const p of row) {
        params.push(p);
      }
    }
  }
  return params;
}

// ========== GPT Forward Pass ==========

export function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  stateDict: Record<string, Value[][]>,
  config: Config
): Value[] {
  // Token and position embeddings
  const tokEmb = stateDict["wte"][tokenId];
  const posEmb = stateDict["wpe"][posId];
  let x = tokEmb.map((t, i) => t.add(posEmb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < config.nLayer; li++) {
    // 1) Multi-head attention block
    const xResidual = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn: Value[] = [];
    for (let h = 0; h < config.nHead; h++) {
      const hs = h * config.headDim;
      const qH = q.slice(hs, hs + config.headDim);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + config.headDim));
      const vH = values[li].map((vi) => vi.slice(hs, hs + config.headDim));

      // Attention logits
      const attnLogits: Value[] = [];
      for (let t = 0; t < kH.length; t++) {
        let dot = new Value(0);
        for (let j = 0; j < config.headDim; j++) {
          dot = dot.add(qH[j].mul(kH[t][j]));
        }
        attnLogits.push(dot.div(Math.sqrt(config.headDim)));
      }

      const attnWeights = softmax(attnLogits);

      // Weighted sum of values
      const headOut: Value[] = [];
      for (let j = 0; j < config.headDim; j++) {
        let sum = new Value(0);
        for (let t = 0; t < vH.length; t++) {
          sum = sum.add(attnWeights[t].mul(vH[t][j]));
        }
        headOut.push(sum);
      }
      xAttn.push(...headOut);
    }

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual[i]));

    // 2) MLP block
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    x = x.map((xi) => xi.relu().pow(2));
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xResidual2[i]));
  }

  const logits = linear(x, stateDict["lm_head"]);
  return logits;
}

// ========== Main ==========

export async function main(): Promise<void> {
  const rng = new MersenneTwister(42);

  // Load or download input dataset
  const inputPath = "input.txt";
  if (!existsSync(inputPath)) {
    console.log("Downloading input.txt...");
    const namesUrl =
      "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
    const response = await fetch(namesUrl);
    const text = await response.text();
    writeFileSync(inputPath, text);
  }

  // Parse documents
  let docs = readFileSync(inputPath, "utf-8")
    .trim()
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  rng.shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  // Tokenizer
  const uchars = [...new Set(docs.join(""))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  console.log(`vocab size: ${vocabSize}`);

  // Config
  const nEmbd = 16;
  const nHead = 4;
  const nLayer = 1;
  const blockSize = 8;
  const headDim = nEmbd / nHead;

  const config: Config = {
    vocabSize,
    nEmbd,
    nHead,
    nLayer,
    blockSize,
    headDim,
  };

  // Initialize parameters
  const stateDict = createStateDict(rng, config);
  const params = getParams(stateDict);
  console.log(`num params: ${params.length}`);

  // Adam optimizer buffers
  const learningRate = 1e-2;
  const beta1 = 0.9;
  const beta2 = 0.95;
  const epsAdam = 1e-8;
  const m: number[] = new Array(params.length).fill(0);
  const v: number[] = new Array(params.length).fill(0);

  // Training loop
  const numSteps = 500;
  for (let step = 0; step < numSteps; step++) {
    // Get document and tokenize
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split("").map((ch) => uchars.indexOf(ch)), BOS];
    const n = Math.min(blockSize, tokens.length - 1);

    // Forward pass
    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const values: Value[][][] = Array.from({ length: nLayer }, () => []);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(tokenId, posId, keys, values, stateDict, config);
      const probs = softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);
    }

    const loss = losses
      .reduce((sum, l) => sum.add(l), new Value(0))
      .div(n);

    // Backward pass
    loss.backward();

    // Adam update
    const lrT = learningRate * 0.5 * (1 + Math.cos(Math.PI * step / numSteps));
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
      const mHat = m[i] / (1 - Math.pow(beta1, step + 1));
      const vHat = v[i] / (1 - Math.pow(beta2, step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      p.grad = 0;
    }

    console.log(
      `step ${String(step + 1).padStart(4)} / ${String(numSteps).padStart(4)} | loss ${loss.data.toFixed(4)}`
    );
  }

  // Inference
  const temperature = 0.5;
  console.log("\n--- inference ---");
  for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const values: Value[][][] = Array.from({ length: nLayer }, () => []);
    let tokenId = BOS;
    const sample: string[] = [];

    for (let posId = 0; posId < blockSize; posId++) {
      const logits = gpt(tokenId, posId, keys, values, stateDict, config);
      const scaledLogits = logits.map((l) => l.div(temperature));
      const probs = softmax(scaledLogits);
      const probValues = probs.map((p) => p.data);
      tokenId = rng.choices(
        Array.from({ length: vocabSize }, (_, i) => i),
        probValues
      );
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }

    console.log(`sample ${String(sampleIdx + 1).padStart(2)}: ${sample.join("")}`);
  }
}

// Entry point
const isMainModule = process.argv[1]?.endsWith("microgpt.ts") || 
                     process.argv[1]?.endsWith("microgpt.js");
if (isMainModule) {
  main().catch(console.error);
}
