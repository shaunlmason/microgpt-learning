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

console.log("Processing document:", doc);
console.log("Tokens:", tokens.slice(0, n+1));

const keys: Value[][][] = [[]];
const values: Value[][][] = [[]];
const stepLosses: Value[] = [];
let lastLogits: Value[] = [];

for (let posId = 0; posId < n; posId++) {
  const tokenId = tokens[posId];
  const targetId = tokens[posId + 1];
  console.log(`\nPosition ${posId}: token=${tokenId} (${uchars[tokenId] || '<BOS>'}), target=${targetId} (${uchars[targetId] || '<BOS>'})`);
  
  lastLogits = gpt(tokenId, posId, keys, values, sd, config);
  const probs = softmax(lastLogits);
  const lossT = probs[targetId].log().neg();
  
  console.log(`  Logits sample: [${lastLogits.slice(0, 3).map(v => v.data.toFixed(4)).join(', ')}...]`);
  console.log(`  Probs sum: ${probs.reduce((a, b) => a + b.data, 0).toFixed(4)}`);
  console.log(`  Target prob: ${probs[targetId].data.toFixed(6)}`);
  console.log(`  Loss: ${lossT.data.toFixed(6)}`);
  
  stepLosses.push(lossT);
}

console.log("\n=== Before backward ===");
console.log("Sample param before:", params[0].data.toFixed(6), "grad:", params[0].grad);

const loss = stepLosses
  .reduce((sum, l) => sum.add(l), new Value(0))
  .div(n);

console.log("\nMean loss:", loss.data.toFixed(6));

loss.backward();

console.log("\n=== After backward ===");
console.log("Sample param after:", params[0].data.toFixed(6), "grad:", params[0].grad);

// Check a few more params
const nonZero = params.filter(p => Math.abs(p.grad) > 1e-10).length;
console.log(`Non-zero gradients: ${nonZero}/${params.length}`);
