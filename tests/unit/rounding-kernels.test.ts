/**
 * Regression tests for the WASM rounding kernels and the array-equality fast
 * paths added alongside them.
 *
 * Both changes introduce a second code path that only engages at or above 32
 * elements, so every case runs at 8 (JS fallback) and 128 (kernel/fast path) and
 * asserts the two agree. That agreement is the point: a kernel that is correct
 * but disagrees with its own fallback is still a bug.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src/index';

const SIZES = [8, 128] as const;
const FLOAT_DTYPES = ['float64', 'float32', 'float16'] as const;

/** Fill `n` elements by cycling `vals`. */
const cycle = (vals: number[], n: number): number[] =>
  Array.from({ length: n }, (_, i) => vals[i % vals.length]!);

describe('rounding kernels', () => {
  describe('exact ties round half to even', () => {
    // The kernel builds this from the add-magic/subtract-magic identity rather
    // than @round, which is round-half-away-from-zero.
    const ties = [0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5];
    const expected = [0, 2, 2, 4, 4, -0, -2, -2, -4];

    for (const dt of FLOAT_DTYPES) {
      for (const n of SIZES) {
        it(`rint ${dt} n=${n}`, () => {
          const got = np.rint(np.array(cycle(ties, n), dt)).toArray() as number[];
          const k = Math.min(n, ties.length);
          expect(got.slice(0, k)).toEqual(expected.slice(0, k));
        });
      }
    }
  });

  describe('near-ties are not ties', () => {
    // The JS path used to treat anything within 1e-10 of .5 as a tie. NumPy does
    // not: np.rint(2.5000000000001) is 3.
    for (const n of SIZES) {
      it(`rint float64 n=${n}`, () => {
        const vals = [2.5000000000001, -2.5000000000001, 2.4999999999999, -2.4999999999999];
        const got = np.rint(np.array(cycle(vals, n), 'float64')).toArray() as number[];
        expect(got.slice(0, 4)).toEqual([3, -3, 2, -2]);
      });
    }
  });

  describe('sign of zero survives', () => {
    // rint(-0.4) is -0.0 in NumPy. The kernel re-applies the input's sign bit
    // instead of negating; the JS path needed the same treatment so the two
    // agree either side of the 32-element threshold.
    for (const n of SIZES) {
      it(`negative inputs rounding to zero stay -0 (n=${n})`, () => {
        const got = np
          .rint(np.array(cycle([-0.4, -0.5, 0.4, -0], n), 'float64'))
          .toArray() as number[];
        expect(Object.is(got[0]!, -0)).toBe(true);
        expect(Object.is(got[1]!, -0)).toBe(true);
        expect(Object.is(got[2]!, 0)).toBe(true);
        expect(Object.is(got[3]!, -0)).toBe(true);
      });

      it(`floor/ceil/trunc keep zero signs (n=${n})`, () => {
        const a = np.array(cycle([-0.4, 0.4, -0, 0], n), 'float64');
        expect(Object.is((np.ceil(a).toArray() as number[])[0]!, -0)).toBe(true);
        expect(Object.is((np.trunc(a).toArray() as number[])[0]!, -0)).toBe(true);
        expect(Object.is((np.floor(a).toArray() as number[])[1]!, 0)).toBe(true);
      });
    }
  });

  describe('specials pass through', () => {
    for (const n of SIZES) {
      it(`inf/-inf/NaN and already-integral values (n=${n})`, () => {
        const big = 4503599627370496; // 2^52 — every double at or above is integral
        const vals = [Infinity, -Infinity, NaN, big, big + 2, 1e300, -1e300];
        for (const op of ['floor', 'ceil', 'trunc', 'fix', 'rint'] as const) {
          const got = (np as never as Record<string, (x: unknown) => { toArray(): number[] }>)[op]!(
            np.array(cycle(vals, n), 'float64'),
          ).toArray();
          expect(got[0]).toBe(Infinity);
          expect(got[1]).toBe(-Infinity);
          expect(got[2]).toBeNaN();
          expect(got[3]).toBe(big);
          expect(got[4]).toBe(big + 2);
          expect(got[5]).toBe(1e300);
          expect(got[6]).toBe(-1e300);
        }
      });
    }
  });

  describe('directed rounding', () => {
    for (const n of SIZES) {
      it(`floor/ceil/trunc disagree in the right directions (n=${n})`, () => {
        const a = np.array(cycle([1.7, -1.7, 1.2, -1.2], n), 'float64');
        expect((np.floor(a).toArray() as number[]).slice(0, 4)).toEqual([1, -2, 1, -2]);
        expect((np.ceil(a).toArray() as number[]).slice(0, 4)).toEqual([2, -1, 2, -1]);
        expect((np.trunc(a).toArray() as number[]).slice(0, 4)).toEqual([1, -1, 1, -1]);
        // fix is trunc under another name and shares its kernel.
        expect(np.fix(a).toArray()).toEqual(np.trunc(a).toArray());
      });
    }
  });

  describe('around/round with decimals', () => {
    for (const n of SIZES) {
      it(`decimals=0 equals rint (n=${n})`, () => {
        const a = np.array(cycle([0.5, 1.5, 2.5, -1.5, 7.25, 2.5000000000001], n), 'float64');
        expect(np.around(a, 0).toArray()).toEqual(np.rint(a).toArray());
        expect(np.round(a, 0).toArray()).toEqual(np.rint(a).toArray());
      });

      it(`decimals=2 scales, rounds half-to-even, unscales (n=${n})`, () => {
        // Values from NumPy 2.3.1. 2.675*100 is exactly 267.5, so the tie rule
        // applies and picks 268; 0.125*100 is exactly 12.5 and picks 12.
        const a = np.array(cycle([1.25, 1.35, -1.25, 2.675, 0.125], n), 'float64');
        const got = np.around(a, 2).toArray() as number[];
        expect(got[0]).toBeCloseTo(1.25, 12);
        expect(got[1]).toBeCloseTo(1.35, 12);
        expect(got[2]).toBeCloseTo(-1.25, 12);
        expect(got[3]).toBeCloseTo(2.68, 12);
        expect(got[4]).toBeCloseTo(0.12, 12);
      });

      it(`negative decimals round to tens (n=${n})`, () => {
        const a = np.array(cycle([15, 25, -15, -25, 4], n), 'float64');
        const got = np.around(a, -1).toArray() as number[];
        // 1.5 and 2.5 tens both tie; even wins: 20 and 20.
        expect(got.slice(0, 5)).toEqual([20, 20, -20, -20, 0]);
      });
    }
  });

  describe('integer dtypes are returned unchanged', () => {
    for (const dt of ['int64', 'int32', 'uint8'] as const) {
      it(`${dt} is already rounded`, () => {
        const vals =
          dt === 'int64'
            ? Array.from({ length: 128 }, (_, i) => BigInt(i - 64))
            : Array.from({ length: 128 }, (_, i) => (dt === 'uint8' ? i : i - 64));
        const a = np.array(vals as never, dt);
        for (const op of ['floor', 'ceil', 'trunc', 'fix'] as const) {
          const got = (np as never as Record<string, (x: unknown) => { toArray(): unknown[] }>)[
            op
          ]!(a).toArray();
          expect(got).toEqual(a.toArray());
        }
      });
    }
  });

  it('JS fallback and kernel agree across the threshold', () => {
    // The strongest statement available: run the same values at 8 and at 128 and
    // require the leading 8 results to be identical.
    const vals = [0.5, -0.5, 2.5, -2.5, 1.7, -1.7, 0.125, 2.675];
    for (const dt of FLOAT_DTYPES) {
      for (const op of ['floor', 'ceil', 'trunc', 'fix', 'rint'] as const) {
        const fn = (np as never as Record<string, (x: unknown) => { toArray(): number[] }>)[op]!;
        const small = fn(np.array(vals, dt)).toArray();
        const large = fn(np.array(cycle(vals, 128), dt))
          .toArray()
          .slice(0, 8);
        expect(large).toEqual(small);
      }
      // decimals === 0 does no scaling and must agree for every float dtype.
      // Non-zero decimals scale at the array's precision, which the kernel only
      // matches for float64 — see the note in wasm/rounding.ts, and finding 4.11
      // for the float32/float16 divergence from NumPy that predates this work.
      for (const d of dt === 'float64' ? [0, 1, 2] : [0]) {
        const small = np.around(np.array(vals, dt), d).toArray();
        const large = np
          .around(np.array(cycle(vals, 128), dt), d)
          .toArray()
          .slice(0, 8);
        expect(large).toEqual(small);
      }
    }
  });
});

describe('array_equal / array_equiv fast paths', () => {
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
    'bool',
  ] as const;

  const build = (dt: string, n: number, mutateAt = -1): unknown[] =>
    Array.from({ length: n }, (_, i) => {
      const base = dt === 'bool' ? i % 2 : i % 7;
      // bool stores any non-zero as 1, so base + 1 would be indistinguishable
      // from base for an odd index — flip it instead.
      const v = i === mutateAt ? (dt === 'bool' ? 1 - base : base + 1) : base;
      return dt === 'int64' || dt === 'uint64' ? BigInt(v) : v;
    });

  for (const dt of DTYPES) {
    for (const n of SIZES) {
      it(`${dt} n=${n}`, () => {
        const a = np.array(build(dt, n) as never, dt);
        const same = np.array(build(dt, n) as never, dt);
        // Mutate the LAST element: an early-exit loop that stopped short would
        // still report equal.
        const diff = np.array(build(dt, n, n - 1) as never, dt);

        expect(np.array_equal(a, same)).toBe(true);
        expect(np.array_equal(a, diff)).toBe(false);
        expect(np.array_equiv(a, same)).toBe(true);
        expect(np.array_equiv(a, diff)).toBe(false);
      });
    }
  }

  it('shape mismatch is never equal, even with matching data', () => {
    const flat = np.array(
      Array.from({ length: 128 }, (_, i) => i % 7),
      'int32',
    );
    const shaped = np.reshape(flat, [16, 8]);
    expect(np.array_equal(shaped, flat)).toBe(false);
  });

  it('non-contiguous views fall through to the generic path correctly', () => {
    const m = np.reshape(
      np.array(
        Array.from({ length: 128 }, (_, i) => i),
        'int32',
      ),
      [16, 8],
    );
    const m2 = np.reshape(
      np.array(
        Array.from({ length: 128 }, (_, i) => i),
        'int32',
      ),
      [16, 8],
    );
    expect(np.array_equal(np.transpose(m), np.transpose(m2))).toBe(true);
  });

  it('NaN is unequal by default and equal under equal_nan', () => {
    for (const dt of FLOAT_DTYPES) {
      const a = np.array(cycle([NaN, 1, 2, NaN, 3, 4, 5, 6], 128), dt);
      const b = np.array(cycle([NaN, 1, 2, NaN, 3, 4, 5, 6], 128), dt);
      expect(np.array_equal(a, b)).toBe(false);
      expect(np.array_equal(a, b, true)).toBe(true);
      // array_equiv has no equal_nan option, so NaN never matches.
      expect(np.array_equiv(a, b)).toBe(false);
    }
  });

  it('+0 and -0 compare equal, as they do in IEEE', () => {
    const pos = np.array(cycle([0], 128), 'float64');
    const neg = np.array(cycle([-0], 128), 'float64');
    expect(np.array_equal(pos, neg)).toBe(true);
  });

  it('infinities compare equal to themselves', () => {
    const a = np.array(cycle([Infinity, -Infinity, 1, 2], 128), 'float64');
    const b = np.array(cycle([Infinity, -Infinity, 1, 2], 128), 'float64');
    expect(np.array_equal(a, b)).toBe(true);
  });

  it('array_equiv still broadcasts rather than comparing pairwise', () => {
    // [3,1] against [1,3] must broadcast to [3,3]; the fast path is gated on
    // identical shapes precisely so this keeps working.
    const col = np.array([[1], [1], [1]], 'int32');
    const row = np.array([[1, 1, 1]], 'int32');
    expect(np.array_equiv(col, row)).toBe(true);
    expect(np.array_equiv(np.array([[1], [2], [3]], 'int32'), row)).toBe(false);
  });

  it('mixed dtypes still compare by value', () => {
    const i = np.array([1, 2, 3, 4, 5, 6, 7, 8], 'int32');
    const f = np.array([1, 2, 3, 4, 5, 6, 7, 8], 'float64');
    expect(np.array_equal(i, f)).toBe(true);
    expect(np.array_equiv(i, f)).toBe(true);
  });

  it('array_equiv compares complex by value, not object identity', () => {
    // `val1 !== val2` on two Complex instances is always true, so this returned
    // false for every complex input regardless of contents.
    const a = np.array([1, 2, 3, 4, 5, 6, 7, 8], 'complex128');
    const b = np.array([1, 2, 3, 4, 5, 6, 7, 8], 'complex128');
    expect(np.array_equiv(a, b)).toBe(true);
    expect(np.array_equal(a, b)).toBe(true);
    const c = np.array([1, 2, 3, 4, 5, 6, 7, 9], 'complex128');
    expect(np.array_equiv(a, c)).toBe(false);
  });
});
