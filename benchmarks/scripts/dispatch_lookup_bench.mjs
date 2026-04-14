/**
 * Microbenchmark: Python dict vs JS Record (plain object) vs JS Map
 * for operation-dispatch-style lookups.
 *
 * Run:  node benchmarks/scripts/dispatch_lookup_bench.mjs
 *       python3 benchmarks/scripts/dispatch_lookup_bench.py
 */

const ITERS = 10_000_000;
const WARMUP = 1_000_000;

// Build ~400 keys to mirror the real dispatch tables
const keys = Array.from({ length: 400 }, (_, i) => `op_${i}`);
const fns = keys.map((_, i) => () => i);

// --- Record (plain object) ---
const record = Object.create(null);
keys.forEach((k, i) => { record[k] = fns[i]; });

// --- Map ---
const map = new Map();
keys.forEach((k, i) => map.set(k, fns[i]));

// Pick a few keys to cycle through (simulates real workload hitting different ops)
const hotKeys = [keys[0], keys[50], keys[150], keys[300], keys[399]];
const hotLen = hotKeys.length;

function benchRecord() {
  let sum = 0;
  for (let i = 0; i < ITERS; i++) {
    const fn = record[hotKeys[i % hotLen]];
    sum += fn();
  }
  return sum;
}

function benchMap() {
  let sum = 0;
  for (let i = 0; i < ITERS; i++) {
    const fn = map.get(hotKeys[i % hotLen]);
    sum += fn();
  }
  return sum;
}

// --- If/else chain (baseline comparison) ---
function benchIfChain() {
  // Build a chain of 400 comparisons via a generated function
  let sum = 0;
  for (let i = 0; i < ITERS; i++) {
    const key = hotKeys[i % hotLen];
    // Simulate checking a few branches (best/mid/worst case averaged)
    let fn;
    for (let j = 0; j < keys.length; j++) {
      if (key === keys[j]) { fn = fns[j]; break; }
    }
    sum += fn();
  }
  return sum;
}

function run(label, fn) {
  // Warmup
  for (let w = 0; w < WARMUP; w++) {
    const key = hotKeys[w % hotLen];
    record[key]?.();
  }
  fn(); // one full warmup run

  const t0 = performance.now();
  const result = fn();
  const elapsed = performance.now() - t0;
  const opsPerSec = (ITERS / elapsed * 1000).toFixed(0);
  console.log(`${label.padEnd(20)} ${elapsed.toFixed(1).padStart(8)} ms   ${Number(opsPerSec).toLocaleString().padStart(15)} ops/sec   (checksum: ${result})`);
}

console.log(`\nJS dispatch lookup benchmark — ${ITERS.toLocaleString()} iterations, ${keys.length} keys\n`);
run('Record (object)', benchRecord);
run('Map', benchMap);
run('If/else chain', benchIfChain);
console.log();
