/**
 * WASM lifecycle tests for dispose via `using`
 *
 * Validates that `using` properly frees WASM memory by checking
 * wasmFreeBytes() before and after scoped array usage. Follows the same
 * warmup + iteration pattern as the wasm-leaks suite so allocator overhead
 * washes out and leak-per-iteration must be exactly zero.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, ones, wasmFreeBytes, zeros } from '../../src/core/index';

const ITERS = 10;
const SIZE = 10_000;

describe('using + wasmFreeBytes lifecycle', () => {
  beforeAll(() => {
    // Warmup: absorb one-time WASM heap initialization costs for each op.
    for (let i = 0; i < 3; i++) {
      zeros([SIZE]).dispose();
      ones([SIZE]).dispose();
      arange(SIZE).dispose();
    }
  });

  it('single array: frees WASM memory each iteration', () => {
    const before = wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using arr = zeros([SIZE]);
      expect(arr.size).toBe(SIZE);
      expect(wasmFreeBytes()).toBeLessThan(before);
    }

    const leak = before - wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });

  it('multiple arrays: all freed each iteration', () => {
    // Run once to absorb any fixed fragmentation from interleaving 3 alloc sizes
    /* eslint-disable @typescript-eslint/no-unused-vars -- using vars exist for dispose side-effect */
    {
      using _a = zeros([SIZE]);
      using _b = ones([SIZE]);
      using _c = arange(SIZE);
    }
    /* eslint-enable @typescript-eslint/no-unused-vars */

    const before = wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using a = zeros([SIZE]);
      using b = ones([SIZE]);
      using c = arange(SIZE);
      expect(a.size).toBe(SIZE);
      expect(b.size).toBe(SIZE);
      expect(c.size).toBe(SIZE);
      expect(wasmFreeBytes()).toBeLessThan(before);
    }

    const leak = before - wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });

  it('nested blocks: inner disposes before outer', () => {
    // Absorb fixed fragmentation from interleaving 2 alloc sizes
    /* eslint-disable @typescript-eslint/no-unused-vars -- using vars exist for dispose side-effect */
    {
      using _a = zeros([SIZE]);
      using _b = ones([SIZE]);
    }
    /* eslint-enable @typescript-eslint/no-unused-vars */

    const before = wasmFreeBytes();

    for (let i = 0; i < ITERS; i++) {
      using outer = zeros([SIZE]);
      const afterOuter = wasmFreeBytes();

      {
        using inner = ones([SIZE]);
        expect(wasmFreeBytes()).toBeLessThan(afterOuter);
        expect(inner.size).toBe(SIZE);
      }

      // Inner freed, outer still alive
      expect(wasmFreeBytes()).toBeLessThan(before);
      expect(outer.size).toBe(SIZE);
    }

    const leak = before - wasmFreeBytes();
    expect(leak / ITERS, `leaks ${leak} bytes over ${ITERS} iters`).toBe(0);
  });
});
