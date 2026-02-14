import { Value } from "./microgpt";

// Test simple gradient computation
const x = new Value(2.0);
const y = new Value(3.0);
const z = x.mul(y); // z = 6

console.log("Before backward:");
console.log("x.data:", x.data, "x.grad:", x.grad);
console.log("y.data:", y.data, "y.grad:", y.grad);
console.log("z.data:", z.data, "z.grad:", z.grad);

z.backward();

console.log("\nAfter backward:");
console.log("x.data:", x.data, "x.grad:", x.grad);
console.log("y.data:", y.data, "y.grad:", y.grad);
console.log("z.data:", z.data, "z.grad:", z.grad);

// Expected: x.grad = 3, y.grad = 2
console.log("\nExpected: x.grad = 3, y.grad = 2");
