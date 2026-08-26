/**
 * Regression tests for stale-buffer bugs in boolean predicates.
 *
 * `ArrayStorage.empty()` may hand back a WASM buffer recycled from a previously
 * disposed array, so its contents are arbitrary. Several predicates had a fast
 * path that "knew" the answer was uniformly false (e.g. `iscomplex` on a real
 * array, `isnan` on an integer dtype) and returned the buffer without writing
 * to it, relying on it being zero-filled. When the pool handed back a block of
 * 1s, those ops reported every element as true.
 *
 * That was invisible to the rest of the suite because nothing dirtied the pool
 * first — a fresh allocation happens to be zeroed. These tests dirty it on
 * purpose, so the same class of bug cannot come back silently.
 *
 * Verified to fail when any of the fixes in src/common/ops/logic.ts is removed.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { wasmConfig } from '../../src/common/wasm/config';

const N = 4096;

const DTYPES = [
  'float64',
  'float32',
  'float16',
  'int64',
  'uint64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
  'complex128',
  'complex64',
] as const;

const PREDICATES = [
  'isnan',
  'isinf',
  'isfinite',
  'isnat',
  'iscomplex',
  'isreal',
  'isneginf',
  'isposinf',
  'signbit',
  'logical_not',
] as const;

/**
 * Return a recycled block to the allocator with every byte set.
 *
 * `equal(a, a)` is uniformly true, so its bool buffer is all 1s; disposing it
 * puts that block at the head of the free list, where the next same-sized
 * request picks it up.
 */
function dirtyBufferPool(size: number): void {
  const a = np.arange(0, size, 1, 'float64');
  const allTrue = np.equal(a, a);
  allTrue.dispose();
  a.dispose();
}

describe('predicates are immune to recycled WASM buffers', () => {
  beforeAll(() => {
    // Force the WASM path at every size so the pooled allocator is in play.
    wasmConfig.thresholdMultiplier = 0;
  });

  for (const dtype of DTYPES) {
    for (const op of PREDICATES) {
      it(`${op} (${dtype}) returns the same result after buffer reuse`, () => {
        let src: ReturnType<typeof np.arange>;
        try {
          src = np.arange(0, N, 1, dtype);
        } catch {
          return; // dtype not constructible this way
        }

        const fn = (
          np as unknown as Record<string, (x: unknown) => { toArray(): unknown; dispose(): void }>
        )[op];
        if (typeof fn !== 'function') {
          src.dispose();
          return;
        }

        let expected: string;
        try {
          const clean = fn(src);
          expected = JSON.stringify(clean.toArray());
          clean.dispose();
        } catch {
          src.dispose();
          return; // op undefined for this dtype (e.g. signbit on complex)
        }

        dirtyBufferPool(N);

        const actual = fn(src);
        const got = JSON.stringify(actual.toArray());
        actual.dispose();
        src.dispose();

        expect(got, `${op}(${dtype}) changed after an all-ones buffer was recycled`).toBe(expected);
      });
    }
  }
});
