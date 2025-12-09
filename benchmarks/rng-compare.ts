/**
 * RNG Algorithm Comparison Benchmark
 *
 * Compares performance of three RNG algorithms:
 * - PCG32 (current implementation)
 * - MT19937 (Mersenne Twister - NumPy legacy)
 * - PCG64 (NumPy 2.0 recommended)
 */

// ============================================================================
// PCG32 Implementation (current)
// ============================================================================

const PCG_MULT = BigInt('6364136223846793005');
const MASK_32 = BigInt(0xffffffff);
const MASK_64 = BigInt('0xffffffffffffffff');

interface PCG32State {
  state: bigint;
  inc: bigint;
}

function createPCG32(seedValue: number): { next: () => number; nextFloat: () => number } {
  const s = BigInt(seedValue >>> 0);
  const rngState: PCG32State = {
    state: BigInt(0),
    inc: (s << BigInt(1)) | BigInt(1),
  };

  // Warm up
  const step = () => {
    const oldState = rngState.state;
    rngState.state = (oldState * PCG_MULT + rngState.inc) & MASK_64;
    const xorShifted = Number((((oldState >> BigInt(18)) ^ oldState) >> BigInt(27)) & MASK_32);
    const rot = Number(oldState >> BigInt(59));
    const negRot = -rot & 31;
    return ((xorShifted >>> rot) | (xorShifted << negRot)) >>> 0;
  };

  step();
  rngState.state = rngState.state + s;
  step();

  return {
    next: step,
    nextFloat: () => step() / 4294967296,
  };
}

// ============================================================================
// MT19937 Implementation (Mersenne Twister)
// ============================================================================

const MT_N = 624;
const MT_M = 397;
const MT_MATRIX_A = 0x9908b0df;
const MT_UPPER_MASK = 0x80000000;
const MT_LOWER_MASK = 0x7fffffff;

interface MT19937State {
  mt: Uint32Array;
  mti: number;
}

function createMT19937(seedValue: number): { next: () => number; nextFloat: () => number } {
  const state: MT19937State = {
    mt: new Uint32Array(MT_N),
    mti: MT_N + 1,
  };

  // Initialize generator state with a seed
  function initGenrand(seed: number): void {
    state.mt[0] = seed >>> 0;
    for (state.mti = 1; state.mti < MT_N; state.mti++) {
      const s = state.mt[state.mti - 1]! ^ (state.mt[state.mti - 1]! >>> 30);
      state.mt[state.mti] =
        ((((s & 0xffff0000) >>> 16) * 1812433253) << 16) +
        (s & 0x0000ffff) * 1812433253 +
        state.mti;
      state.mt[state.mti] >>>= 0; // Convert to unsigned 32-bit
    }
  }

  // Generate next random number
  function genrandInt32(): number {
    let y: number;
    const mag01 = [0, MT_MATRIX_A];

    if (state.mti >= MT_N) {
      let kk: number;

      for (kk = 0; kk < MT_N - MT_M; kk++) {
        y = (state.mt[kk]! & MT_UPPER_MASK) | (state.mt[kk + 1]! & MT_LOWER_MASK);
        state.mt[kk] = state.mt[kk + MT_M]! ^ (y >>> 1) ^ mag01[y & 1]!;
      }
      for (; kk < MT_N - 1; kk++) {
        y = (state.mt[kk]! & MT_UPPER_MASK) | (state.mt[kk + 1]! & MT_LOWER_MASK);
        state.mt[kk] = state.mt[kk + (MT_M - MT_N)]! ^ (y >>> 1) ^ mag01[y & 1]!;
      }
      y = (state.mt[MT_N - 1]! & MT_UPPER_MASK) | (state.mt[0]! & MT_LOWER_MASK);
      state.mt[MT_N - 1] = state.mt[MT_M - 1]! ^ (y >>> 1) ^ mag01[y & 1]!;

      state.mti = 0;
    }

    y = state.mt[state.mti++]!;

    // Tempering
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;

    return y >>> 0;
  }

  initGenrand(seedValue);

  return {
    next: genrandInt32,
    nextFloat: () => genrandInt32() / 4294967296,
  };
}

// ============================================================================
// PCG64 Implementation (DXSM variant - simpler and commonly used)
// ============================================================================

// PCG64 DXSM uses 128-bit state with simpler output function
// This is actually what NumPy 2.0 uses (PCG64 DXSM)
const PCG64_MULT = BigInt('0x2360ed051fc65da44385df649fccf645');
const PCG64_INC = BigInt('0x5851f42d4c957f2d14057b7ef767814f');
const MASK_128 = (BigInt(1) << BigInt(128)) - BigInt(1);

interface PCG64State {
  state: bigint; // 128-bit state
}

function createPCG64(seedValue: number): { next: () => number; nextFloat: () => number } {
  const s = BigInt(seedValue >>> 0);
  const state: PCG64State = {
    state: s | (s << BigInt(64)),
  };

  // DXSM output function: cheap-ish strong output function
  function output(): number {
    const oldState = state.state;
    // Advance state
    state.state = (oldState * PCG64_MULT + PCG64_INC) & MASK_128;

    // DXSM output: hi ^ lo, then multiply and shift
    const lo = oldState & MASK_64;
    const hi = (oldState >> BigInt(64)) & MASK_64;

    // XOR high and low, then further mix
    let out = hi ^ lo;
    out = (out * BigInt('0xda942042e4dd58b5')) & MASK_64;

    // Return high 32 bits as number
    return Number((out >> BigInt(32)) & MASK_32);
  }

  // Warm up
  output();
  output();

  return {
    next: output,
    nextFloat: () => output() / 4294967296,
  };
}

// ============================================================================
// Benchmark Runner
// ============================================================================

function benchmark(
  name: string,
  fn: () => void,
  iterations: number = 1000000,
  warmupIterations: number = 10000
): { name: string; opsPerSec: number; timeMs: number } {
  // Warm up
  for (let i = 0; i < warmupIterations; i++) {
    fn();
  }

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = performance.now();

  const timeMs = end - start;
  const opsPerSec = (iterations / timeMs) * 1000;

  return { name, opsPerSec, timeMs };
}

function formatOps(ops: number): string {
  if (ops >= 1e9) return `${(ops / 1e9).toFixed(2)}B`;
  if (ops >= 1e6) return `${(ops / 1e6).toFixed(2)}M`;
  if (ops >= 1e3) return `${(ops / 1e3).toFixed(2)}K`;
  return ops.toFixed(2);
}

function runBenchmarks(): void {
  console.log('='.repeat(70));
  console.log('RNG Algorithm Comparison Benchmark');
  console.log('='.repeat(70));
  console.log('');

  const pcg32 = createPCG32(42);
  const mt19937 = createMT19937(42);
  const pcg64 = createPCG64(42);

  const iterations = 10_000_000;

  console.log(`Generating ${formatOps(iterations)} random integers...\n`);

  // Integer generation benchmark
  const results = [
    benchmark('PCG32', () => pcg32.next(), iterations),
    benchmark('MT19937', () => mt19937.next(), iterations),
    benchmark('PCG64', () => pcg64.next(), iterations),
  ];

  console.log('Integer Generation (32-bit):');
  console.log('-'.repeat(50));
  console.log('Algorithm       | ops/sec        | Time (ms)');
  console.log('-'.repeat(50));

  // Sort by performance (highest ops/sec first)
  results.sort((a, b) => b.opsPerSec - a.opsPerSec);
  const fastest = results[0]!.opsPerSec;

  for (const r of results) {
    const relative = (r.opsPerSec / fastest) * 100;
    console.log(
      `${r.name.padEnd(15)} | ${formatOps(r.opsPerSec).padEnd(14)} | ${r.timeMs.toFixed(2).padEnd(10)} (${relative.toFixed(0)}%)`
    );
  }

  console.log('');

  // Float generation benchmark
  console.log(`\nGenerating ${formatOps(iterations)} random floats [0, 1)...\n`);

  const floatResults = [
    benchmark('PCG32', () => pcg32.nextFloat(), iterations),
    benchmark('MT19937', () => mt19937.nextFloat(), iterations),
    benchmark('PCG64', () => pcg64.nextFloat(), iterations),
  ];

  console.log('Float Generation [0, 1):');
  console.log('-'.repeat(50));
  console.log('Algorithm       | ops/sec        | Time (ms)');
  console.log('-'.repeat(50));

  floatResults.sort((a, b) => b.opsPerSec - a.opsPerSec);
  const fastestFloat = floatResults[0]!.opsPerSec;

  for (const r of floatResults) {
    const relative = (r.opsPerSec / fastestFloat) * 100;
    console.log(
      `${r.name.padEnd(15)} | ${formatOps(r.opsPerSec).padEnd(14)} | ${r.timeMs.toFixed(2).padEnd(10)} (${relative.toFixed(0)}%)`
    );
  }

  // Array fill benchmark
  console.log('');
  console.log(`\nFilling 1M element arrays...\n`);

  const arraySize = 1_000_000;
  const arrayIterations = 10;

  function fillArrayPCG32(): Float64Array {
    const arr = new Float64Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
      arr[i] = pcg32.nextFloat();
    }
    return arr;
  }

  function fillArrayMT19937(): Float64Array {
    const arr = new Float64Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
      arr[i] = mt19937.nextFloat();
    }
    return arr;
  }

  function fillArrayPCG64(): Float64Array {
    const arr = new Float64Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
      arr[i] = pcg64.nextFloat();
    }
    return arr;
  }

  const arrayResults = [
    benchmark('PCG32', fillArrayPCG32, arrayIterations, 2),
    benchmark('MT19937', fillArrayMT19937, arrayIterations, 2),
    benchmark('PCG64', fillArrayPCG64, arrayIterations, 2),
  ];

  console.log('Array Fill (1M Float64):');
  console.log('-'.repeat(50));
  console.log('Algorithm       | arrays/sec     | Time/array (ms)');
  console.log('-'.repeat(50));

  arrayResults.sort((a, b) => b.opsPerSec - a.opsPerSec);
  const fastestArray = arrayResults[0]!.opsPerSec;

  for (const r of arrayResults) {
    const relative = (r.opsPerSec / fastestArray) * 100;
    const timePerArray = r.timeMs / arrayIterations;
    console.log(
      `${r.name.padEnd(15)} | ${r.opsPerSec.toFixed(2).padEnd(14)} | ${timePerArray.toFixed(2).padEnd(10)} (${relative.toFixed(0)}%)`
    );
  }

  // Quality verification
  console.log('\n');
  console.log('='.repeat(70));
  console.log('Quality Verification (mean should be ~0.5 for uniform [0,1))');
  console.log('='.repeat(70));

  const sampleSize = 100000;

  for (const [name, rng] of [
    ['PCG32', createPCG32(12345)],
    ['MT19937', createMT19937(12345)],
    ['PCG64', createPCG64(12345)],
  ] as const) {
    let sum = 0;
    for (let i = 0; i < sampleSize; i++) {
      sum += rng.nextFloat();
    }
    const mean = sum / sampleSize;
    console.log(`${name.padEnd(10)}: mean = ${mean.toFixed(6)} (deviation: ${((mean - 0.5) * 100).toFixed(4)}%)`);
  }

  console.log('\n');
  console.log('='.repeat(70));
  console.log('Summary');
  console.log('='.repeat(70));
  console.log(`
Key findings:
- MT19937: Uses 32-bit integer operations, ~2.5KB state array
- PCG32:   Uses 64-bit BigInt operations, 16 bytes state
- PCG64:   Uses 128-bit BigInt emulation, more computation

Note: In native C/C++, PCG64 would be competitive with MT19937.
In JavaScript, BigInt operations are slower than native 32-bit integers,
which affects PCG32 and especially PCG64 performance.

Recommendations:
- For maximum JS performance: MT19937 (pure 32-bit ops)
- For current numpy-ts: PCG32 (good balance of quality and performance)
- For NumPy compatibility: PCG64 (matches NumPy 2.0 default_rng)
`);
}

// Run benchmarks
runBenchmarks();
