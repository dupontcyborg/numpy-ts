/**
 * Browser fixture: Symbol.dispose protocol through transpiled `using`
 *
 * Unlike dispose-using.ts (which self-executes and prints PASS), this fixture
 * exports a run() function that returns structured results. This allows browser
 * tests to dynamically import the bundled output, call run(), and assert against
 * the returned object.
 *
 * When bundled with a downlevel target (e.g. ES2020), the `using` keyword is
 * transpiled to a helper that looks up [Symbol.dispose] or
 * [Symbol.for("Symbol.dispose")]. On WebKit (Safari), our polyfill provides
 * the key; on Chromium/Firefox, the native symbol is used directly.
 *
 * Memory assertions follow the wasm-leaks pattern: warmup first, then measure
 * over N iterations so allocator overhead washes out and leak-per-iteration
 * must be exactly zero.
 */
import { arange, ones, wasmFreeBytes, zeros } from 'numpy-ts';

const SIZE = 10_000;
const ITERS = 10;

export interface DisposeTestResult {
  passed: boolean;
  leak: number;
  error?: string;
}

export function run(): DisposeTestResult {
  try {
    // Warmup: absorb one-time WASM heap initialization costs for each op.
    for (let i = 0; i < 3; i++) {
      zeros([SIZE]).dispose();
      ones([SIZE]).dispose();
      arange(SIZE).dispose();
    }

    // Absorb fixed allocator fragmentation from interleaving different sizes
    /* eslint-disable @typescript-eslint/no-unused-vars -- using vars exist for dispose side-effect */
    {
      using _a = zeros([SIZE]);
      using _b = ones([SIZE]);
      using _c = arange(SIZE);
    }
    /* eslint-enable @typescript-eslint/no-unused-vars */

    // --- Single array ---
    const beforeSingle = wasmFreeBytes();
    for (let i = 0; i < ITERS; i++) {
      using arr = zeros([SIZE]);
      if (arr.size !== SIZE) {
        return { passed: false, leak: 0, error: `single: unexpected size ${arr.size}` };
      }
      if (wasmFreeBytes() >= beforeSingle) {
        return { passed: false, leak: 0, error: 'single: memory not allocated during using block' };
      }
    }
    const singleLeak = beforeSingle - wasmFreeBytes();
    if (singleLeak > 0) {
      return {
        passed: false,
        leak: singleLeak,
        error: `single: leaked ${singleLeak} bytes over ${ITERS} iters`,
      };
    }

    // --- Multiple arrays ---
    const beforeMulti = wasmFreeBytes();
    for (let i = 0; i < ITERS; i++) {
      using a = zeros([SIZE]);
      using b = ones([SIZE]);
      using c = arange(SIZE);
      if (a.size !== SIZE || b.size !== SIZE || c.size !== SIZE) {
        return { passed: false, leak: 0, error: 'multi: unexpected size' };
      }
    }
    const multiLeak = beforeMulti - wasmFreeBytes();
    if (multiLeak > 0) {
      return {
        passed: false,
        leak: multiLeak,
        error: `multi: leaked ${multiLeak} bytes over ${ITERS} iters`,
      };
    }

    // --- Nested blocks ---
    const beforeNested = wasmFreeBytes();
    for (let i = 0; i < ITERS; i++) {
      using outer = zeros([SIZE]);
      const afterOuter = wasmFreeBytes();
      {
        using _inner = ones([SIZE]); // eslint-disable-line @typescript-eslint/no-unused-vars -- dispose side-effect
        if (wasmFreeBytes() >= afterOuter) {
          return { passed: false, leak: 0, error: 'nested: inner did not allocate' };
        }
      }
      // Inner should be freed, outer still alive
      if (outer.size !== SIZE) {
        return { passed: false, leak: 0, error: 'nested: outer corrupted after inner dispose' };
      }
    }
    const nestedLeak = beforeNested - wasmFreeBytes();
    if (nestedLeak > 0) {
      return {
        passed: false,
        leak: nestedLeak,
        error: `nested: leaked ${nestedLeak} bytes over ${ITERS} iters`,
      };
    }

    // --- Transpiler key ---
    const key = Symbol.dispose || Symbol.for('Symbol.dispose');
    const arr = zeros([1]);
    if (typeof (arr as any)[key] !== 'function') {
      return { passed: false, leak: 0, error: 'transpiler key: dispose method not found' };
    }
    (arr as any)[key]();

    return { passed: true, leak: 0 };
  } catch (e) {
    return { passed: false, leak: 0, error: String(e) };
  }
}
