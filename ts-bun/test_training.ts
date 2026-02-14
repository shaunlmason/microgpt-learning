import { MersenneTwister } from "./microgpt";
import { readFileSync, existsSync, writeFileSync } from "fs";
import { Value, softmax, gpt, createStateDict, getParams, Config } from "./microgpt";

const ref = JSON.parse(readFileSync("../reference_values.json", "utf-8"));

const rng = new MersenneTwister(42);

let docs = readFileSync("input.txt", "utf-8")
  .trim()
  .split("\n")
  .map((l: string) => l.trim())
  .filter((l: string) => l.length > 0);
rng.shuffle(docs);

const uchars = [...new Set(docs.join(""))].sort();
const BOS = uchars.length;
const vocabSize = uchars.length + 1;

console.log("vocabSize:", vocabSize, "ref:", ref.vocab_size);
console.log("num_docs:", docs.length, "ref:", ref.num_docs);
console.log("first_doc:", docs[0], "ref:", ref.first_doc_after_shuffle);

const config: Config = {
  vocabSize,
  nEmbd: 16,
  nHead: 4,
  nLayer: 1,
  blockSize: 8,
  headDim: 4,
};

const sd = createStateDict(rng, config);
const params = getParams(sd);

const learningRate = 1e-2;
const beta1 = 0.9;
const beta2 = 0.95;
const epsAdam = 1e-8;
const m: number[] = new Array(params.length).fill(0);
const v: number[] = new Array(params.length).fill(0);

const losses: number[] = [];
const numSteps = 5;
let step1Logits: number[] = [];
let step1Grads: number[] = [];

for (let step = 0; step < numSteps; step++) {
  const doc = docs[step % docs.length];
  const tokens = [
    BOS,
    ...doc.split("").map((ch) => uchars.indexOf(ch)),
    BOS,
  ];
  const n = Math.min(8, tokens.length - 1);

  const keys: Value[][][] = [[]];
  const values: Value[][][] = [[]];
  const stepLosses: Value[] = [];
  let lastLogits: Value[] = [];

  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId];
    const targetId = tokens[posId + 1];
    lastLogits = gpt(tokenId, posId, keys, values, sd, config);
    const probs = softmax(lastLogits);
    const lossT = probs[targetId].log().neg();
    stepLosses.push(lossT);
  }

  const loss = stepLosses
    .reduce((sum, l) => sum.add(l), new Value(0))
    .div(n);
  losses.push(loss.data);

  if (step === 0) {
    step1Logits = lastLogits.map((v) => v.data);
    loss.backward();
    step1Grads = params.slice(0, 20).map((p) => p.grad);
  } else {
    loss.backward();
  }

  const lrT = learningRate * 0.5 * (1 + Math.cos((Math.PI * step) / numSteps));
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
    const mHat = m[i] / (1 - Math.pow(beta1, step + 1));
    const vHat = v[i] / (1 - Math.pow(beta2, step + 1));
    p.data -= (lrT * mHat) / (Math.sqrt(vHat) + epsAdam);
    p.grad = 0;
  }
}

console.log("\nActual step1_loss:", losses[0]);
console.log("Expected step1_loss:", ref.step1_loss);
console.log("Diff:", Math.abs(losses[0] - ref.step1_loss));

console.log("\nActual losses_5_steps:", losses);
console.log("Expected losses_5_steps:", ref.losses_5_steps);

console.log("\nActual step1_logits_first[:5]:", step1Logits.slice(0, 5));
console.log("Expected step1_logits_first[:5]:", ref.step1_logits_first.slice(0, 5));

console.log("\nActual step1_gradients_20[:5]:", step1Grads.slice(0, 5));
console.log("Expected step1_gradients_20[:5]:", ref.step1_gradients_20.slice(0, 5));
