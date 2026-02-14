import { MersenneTwister } from "./microgpt";
import { createStateDict, getParams, Config } from "./microgpt";

const rng = new MersenneTwister(42);

const config: Config = {
  vocabSize: 27,
  nEmbd: 16,
  nHead: 4,
  nLayer: 1,
  blockSize: 8,
  headDim: 4,
};

const sd = createStateDict(rng, config);
const params = getParams(sd);

console.log("Number of params:", params.length);
console.log("First 5 params:", params.slice(0, 5).map(p => p.data));
console.log("Params 10-15:", params.slice(10, 15).map(p => p.data));

// Check wte shape
console.log("\nwte shape:", sd.wte.length, "x", sd.wte[0].length);
console.log("wte[0][:5]:", sd.wte[0].slice(0, 5).map(v => v.data));
