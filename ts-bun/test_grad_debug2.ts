import { MersenneTwister } from "./microgpt";
import { readFileSync, existsSync } from "fs";
import { Value, softmax, gpt, createStateDict, getParams, Config } from "./microgpt";

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

// Just test step 0
const doc = docs[0];
const tokens = [
  BOS,
  ...doc.split("").map((ch) => uchars.indexOf(ch)),
  BOS,
];
const n = Math.min(8, tokens.length - 1);

const keys: Value[][][] = [[]];
const values: Value[][][] = [[]];
const stepLosses: Value[] = [];

for (let posId = 0; posId < n; posId++) {
  const tokenId = tokens[posId];
  const targetId = tokens[posId + 1];
  const logits = gpt(tokenId, posId, keys, values, sd, config);
  const probs = softmax(logits);
  const lossT = probs[targetId].log().neg();
  stepLosses.push(lossT);
}

const loss = stepLosses
  .reduce((sum, l) => sum.add(l), new Value(0))
  .div(n);

loss.backward();

// Show first 20 params with their grads
console.log("First 20 params (data, grad):");
for (let i = 0; i < 20; i++) {
  console.log(`  ${i}: ${params[i].data.toFixed(6)}, ${params[i].grad.toFixed(10)}`);
}

// Show non-zero grads
const nonZeroIndices = params
  .map((p, i) => ({ idx: i, grad: p.grad }))
  .filter(x => Math.abs(x.grad) > 1e-10);
console.log(`\nFirst 10 non-zero grads:`);
for (let i = 0; i < Math.min(10, nonZeroIndices.length); i++) {
  const x = nonZeroIndices[i];
  console.log(`  param[${x.idx}]: grad=${x.grad.toFixed(10)}`);
}
