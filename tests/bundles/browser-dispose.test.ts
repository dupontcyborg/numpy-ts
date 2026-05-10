/**
 * Browser dispose-protocol test — vanilla path (no bundler/transpiler)
 *
 * Tests native `using` support directly in the browser, importing the
 * browser bundle without any transpilation. This validates that runtimes
 * with native `using` support can use the dispose protocol out of the box.
 *
 * This file uses raw `using` syntax, so it CANNOT run on WebKit (Safari),
 * which doesn't parse `using`. It runs only on Chromium and Firefox via
 * the browser-dispose-vanilla vitest project.
 *
 * Memory assertions follow the wasm-leaks pattern: warmup first, then
 * measure over N iterations so allocator overhead washes out.
 *
 * Test matrix covered:
 *   vanilla (no transpiler) × Chromium, Firefox
 */

import { beforeAll, describe, expect, test } from 'vitest';

let np: any;
const ITERS = 10;
const SIZE = 10_000;

describe('vanilla dispose protocol (native using)', () => {
  beforeAll(async () => {
    np = await import('/dist/numpy-ts.browser.js');
    // Warmup: absorb one-time WASM heap initialization costs for each op.
    for (let i = 0; i < 3; i++) {
      np.zeros([SIZE]).dispose();
      np.ones([SIZE]).dispose();
      np.arange(SIZE).dispose();
    }
  });

  test('Symbol.dispose is defined (native)', () => {
    expect(typeof Symbol.dispose).toBe('symbol');
  });

  test('using: single array frees memory each iteration', () => {
    const before = np.wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using arr = np.zeros([SIZE]);
      expect(arr.size).toBe(SIZE);
      expect(np.wasmFreeBytes()).toBeLessThan(before);
    }

    const leak = before - np.wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });

  test('using: multiple arrays freed each iteration', () => {
    // Absorb fixed allocator fragmentation from interleaving sizes
    /* eslint-disable @typescript-eslint/no-unused-vars -- using vars exist for dispose side-effect */
    {
      using _a = np.zeros([SIZE]);
      using _b = np.ones([SIZE]);
      using _c = np.arange(SIZE);
    }
    /* eslint-enable @typescript-eslint/no-unused-vars */

    const before = np.wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using a = np.zeros([SIZE]);
      using b = np.ones([SIZE]);
      using c = np.arange(SIZE);
      expect(a.size).toBe(SIZE);
      expect(b.size).toBe(SIZE);
      expect(c.size).toBe(SIZE);
      expect(np.wasmFreeBytes()).toBeLessThan(before);
    }

    const leak = before - np.wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });

  test('using: nested blocks dispose inner before outer', () => {
    // Absorb fixed allocator fragmentation
    /* eslint-disable @typescript-eslint/no-unused-vars -- using vars exist for dispose side-effect */
    {
      using _a = np.zeros([SIZE]);
      using _b = np.ones([SIZE]);
    }
    /* eslint-enable @typescript-eslint/no-unused-vars */

    const before = np.wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using outer = np.zeros([SIZE]);
      const afterOuter = np.wasmFreeBytes();

      {
        using inner = np.ones([SIZE]);
        expect(np.wasmFreeBytes()).toBeLessThan(afterOuter);
        expect(inner.size).toBe(SIZE);
      }

      expect(np.wasmFreeBytes()).toBeLessThan(before);
      expect(outer.size).toBe(SIZE);
    }

    const leak = before - np.wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });
});
