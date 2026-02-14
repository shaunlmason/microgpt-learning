import { MersenneTwister } from "./microgpt";

const rng = new MersenneTwister(42);

console.log("First 10 random() values:");
for (let i = 0; i < 10; i++) {
  console.log(`  ${i}: ${rng.random()}`);
}

console.log("\nFirst 10 gauss() values:");
const rng2 = new MersenneTwister(42);
for (let i = 0; i < 10; i++) {
  console.log(`  ${i}: ${rng2.gauss(0, 1)}`);
}
