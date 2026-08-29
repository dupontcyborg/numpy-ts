/**
 * Regression tests for 64-bit integer comparisons.
 *
 * Comparisons used to funnel every operand through `Number()`, which collapses
 * int64/uint64 values above 2^53. `equal(2^53, 2^53 + 1)` returned true and
 * `less` returned false, both disagreeing with NumPy.
 *
 * Two paths must stay correct: the WASM kernel (used above the size threshold,
 * comparing true 64-bit values) and the JS fallback (used below it, comparing
 * BigInt directly). The small cases here exercise the fallback; the large ones
 * exercise the kernel.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src';

const BIG = 9007199254740992n; // 2^53 — the first integer Number() cannot separate

describe('int64/uint64 comparisons keep full 64-bit precision', () => {
  for (const dtype of ['int64', 'uint64'] as const) {
    it(`${dtype}: values differing above 2^53 compare correctly (JS fallback)`, () => {
      const a = np.array([BIG, BIG + 1n, BIG + 2n], dtype);
      const b = np.array([BIG + 1n, BIG + 1n, BIG + 1n], dtype);
      expect(np.equal(a, b).toArray()).toEqual([0, 1, 0]);
      expect(np.not_equal(a, b).toArray()).toEqual([1, 0, 1]);
      expect(np.less(a, b).toArray()).toEqual([1, 0, 0]);
      expect(np.less_equal(a, b).toArray()).toEqual([1, 1, 0]);
      expect(np.greater(a, b).toArray()).toEqual([0, 0, 1]);
      expect(np.greater_equal(a, b).toArray()).toEqual([0, 1, 1]);
    });

    it(`${dtype}: same values past the WASM threshold (kernel path)`, () => {
      const n = 512; // comfortably above the kernel's 32-element threshold
      const av: bigint[] = [];
      const bv: bigint[] = [];
      for (let i = 0; i < n; i++) {
        av.push(BIG + BigInt(i % 3));
        bv.push(BIG + 1n);
      }
      const a = np.array(av, dtype);
      const b = np.array(bv, dtype);
      const eq = np.equal(a, b).toArray() as number[];
      const lt = np.less(a, b).toArray() as number[];
      for (let i = 0; i < n; i++) {
        const ai = BIG + BigInt(i % 3);
        expect(eq[i], `equal at ${i}`).toBe(ai === BIG + 1n ? 1 : 0);
        expect(lt[i], `less at ${i}`).toBe(ai < BIG + 1n ? 1 : 0);
      }
    });
  }
});
