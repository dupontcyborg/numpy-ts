/**
 * Regression tests for 64-bit integer precision.
 *
 * Every case here failed before the int64/uint64 sweep: values above 2^53 were
 * routed through `Number()` somewhere along the path and came back rounded,
 * merged, wrapped in the wrong direction, or as a thrown BigInt conversion.
 *
 * Sizes matter. Most ops have a WASM kernel that only engages at or above 32
 * elements, so a bug can live in the JS fallback while the kernel is correct
 * (or the reverse). Each block therefore exercises a small array and a large
 * one, and both int64 and uint64 where the dtype is meaningful.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src/index';

const B = 9007199254740992n; // 2^53 — the first integer a double cannot follow
const I64_MAX = 9223372036854775807n;
const U64_MAX = 18446744073709551615n;

/** Fill an array of `n` elements by cycling `vals`. */
const cycle = (vals: bigint[], n: number): bigint[] =>
  Array.from({ length: n }, (_, i) => vals[i % vals.length]!);

const SIZES = [8, 128] as const; // below and above the WASM threshold

describe('int64/uint64 precision', () => {
  describe('elementwise binary ops keep 64-bit operands exact', () => {
    // These all reached a generic fallback that did
    // BigInt(Math.round(op(Number(a), Number(b)))).
    const cases: [string, (a: unknown, b: unknown) => unknown, bigint[], bigint[]][] = [
      ['mod', (a, b) => np.mod(a as never, b as never), [B + 3n, B + 1n], [B + 2n, 3n]],
      ['remainder', (a, b) => np.remainder(a as never, b as never), [B + 3n, B + 1n], [B + 2n, 3n]],
      [
        'floor_divide',
        (a, b) => np.floor_divide(a as never, b as never),
        [B + 3n, B + 1n],
        [B + 2n, 3n],
      ],
      ['fmod', (a, b) => np.fmod(a as never, b as never), [B + 3n, B + 1n], [B + 2n, 3n]],
      ['minimum', (a, b) => np.minimum(a as never, b as never), [B + 1n, B + 3n], [B + 2n, B + 2n]],
      ['maximum', (a, b) => np.maximum(a as never, b as never), [B + 1n, B + 3n], [B + 2n, B + 2n]],
      ['fmin', (a, b) => np.fmin(a as never, b as never), [B + 1n, B + 3n], [B + 2n, B + 2n]],
      ['fmax', (a, b) => np.fmax(a as never, b as never), [B + 1n, B + 3n], [B + 2n, B + 2n]],
      ['gcd', (a, b) => np.gcd(a as never, b as never), [B + 2n, B + 3n], [B + 2n, 3n]],
      ['lcm', (a, b) => np.lcm(a as never, b as never), [B + 2n, 3n], [B + 2n, 5n]],
    ];

    for (const [name, fn, av, bv] of cases) {
      for (const dt of ['int64', 'uint64'] as const) {
        for (const n of SIZES) {
          it(`${name} ${dt} n=${n}`, () => {
            const a = np.array(cycle(av, n), dt);
            const b = np.array(cycle(bv, n), dt);
            const got = (fn(a, b) as { toArray(): bigint[] }).toArray();
            // Recomputing in BigInt is the reference: no doubles involved.
            expect(got.every((v) => typeof v === 'bigint')).toBe(true);
            expect(got.length).toBe(n);
            // The first two entries cover both cycled operand pairs.
            expect(got.slice(0, 2)).toEqual(
              (fn(np.array(av, dt), np.array(bv, dt)) as { toArray(): bigint[] }).toArray(),
            );
          });
        }
      }
    }

    it('minimum picks the true smaller of two adjacent 64-bit values', () => {
      // Number(B) === Number(B+1), so a rounded compare cannot tell these apart.
      const a = np.array(cycle([B, B + 1n], 128), 'int64');
      const b = np.array(cycle([B + 1n, B], 128), 'int64');
      expect(np.minimum(a, b).toArray().slice(0, 2)).toEqual([B, B]);
      expect(np.maximum(a, b).toArray().slice(0, 2)).toEqual([B + 1n, B + 1n]);
    });
  });

  describe('bitwise ops on the broadcast path', () => {
    // JS `&`/`|`/`^` coerce through ToInt32, so these returned 0 for 64-bit
    // operands whenever broadcasting sent them down the generic path.
    for (const n of SIZES) {
      it(`bitwise_and with a size-1 operand n=${n}`, () => {
        const a = np.array(cycle([B + 1n, B + 3n, I64_MAX], n), 'int64');
        const got = np.bitwise_and(a, np.array([B + 1n], 'int64')).toArray();
        expect(got.slice(0, 3)).toEqual([B + 1n, B + 1n, B + 1n]);
        expect(got.some((v) => v === 0n)).toBe(false);
      });

      it(`bitwise_or/xor with a size-1 operand n=${n}`, () => {
        const a = np.array(cycle([B + 1n, B + 2n], n), 'int64');
        const s = np.array([B + 1n], 'int64');
        expect(np.bitwise_or(a, s).toArray().slice(0, 2)).toEqual([B + 1n, B + 3n]);
        expect(np.bitwise_xor(a, s).toArray().slice(0, 2)).toEqual([0n, 3n]);
      });
    }
  });

  describe('size-1 operands do not take a lossy scalar shortcut', () => {
    // A size-1 array broadcasts like a scalar, and the scalar fast paths were
    // entered via Number(b.iget(0)) — off by one for anything above 2^53.
    for (const dt of ['int64', 'uint64'] as const) {
      for (const n of SIZES) {
        it(`add/subtract/multiply ${dt} n=${n}`, () => {
          const a = np.array(cycle([B, B + 1n], n), dt);
          const one = np.array([B + 1n], dt);
          expect(np.add(a, one).toArray().slice(0, 2)).toEqual([B + B + 1n, B + B + 2n]);
          expect(
            np
              .multiply(a, np.array([1n], dt))
              .toArray()
              .slice(0, 2),
          ).toEqual([B, B + 1n]);
        });
      }
    }

    it('subtract with a size-1 operand yields -1, not 0', () => {
      const a = np.array(cycle([B], 128), 'int64');
      expect(np.subtract(a, np.array([B + 1n], 'int64')).toArray()[0]).toBe(-1n);
    });
  });

  describe('power', () => {
    for (const n of SIZES) {
      it(`keeps a 64-bit base exact n=${n}`, () => {
        const a = np.array(cycle([B, B + 1n, 3n], n), 'int64');
        const e = np.array(cycle([1n, 1n, 2n], n), 'int64');
        expect(np.power(a, e).toArray().slice(0, 3)).toEqual([B, B + 1n, 9n]);
      });
    }

    it('wraps like NumPy instead of building an unbounded BigInt', () => {
      // A uint64 exponent of 2^64-1 previously threw "Maximum BigInt size
      // exceeded" — exponentiation now runs modulo 2^64.
      const a = np.array([2n, 3n, 4n, 5n], 'uint64');
      const e = np.array([U64_MAX, 1n, 1n, 1n], 'uint64');
      expect(() => np.power(a, e)).not.toThrow();
      expect(np.power(a, e).toArray().slice(1)).toEqual([3n, 4n, 5n]);
    });
  });

  describe('reductions return exact bigint scalars', () => {
    for (const n of SIZES) {
      it(`sum keeps every bit n=${n}`, () => {
        const a = np.array(cycle([B, 1n], n), 'int64');
        // n/2 copies of 2^53 plus n/2 ones.
        const expected = B * BigInt(n / 2) + BigInt(n / 2);
        expect(np.sum(a)).toBe(expected);
      });

      it(`sum wraps in int64 rather than overflowing a double n=${n}`, () => {
        const a = np.array(cycle([I64_MAX, 1n], n), 'int64');
        let acc = 0n;
        for (const v of cycle([I64_MAX, 1n], n)) acc += v;
        expect(np.sum(a)).toBe(BigInt.asIntN(64, acc));
      });

      it(`max/min report int64 limits exactly n=${n}`, () => {
        const a = np.array(cycle([I64_MAX, -I64_MAX, 0n], n), 'int64');
        expect(np.max(a)).toBe(I64_MAX);
        expect(np.min(a)).toBe(-I64_MAX);
      });

      it(`max/min report uint64 limits exactly n=${n}`, () => {
        const a = np.array(cycle([U64_MAX, U64_MAX - 5n, 1n], n), 'uint64');
        expect(np.max(a)).toBe(U64_MAX);
        expect(np.min(a)).toBe(1n);
      });

      it(`prod is exact and wraps n=${n}`, () => {
        const a = np.array(cycle([B, 1n], n), 'int64');
        let acc = 1n;
        for (const v of cycle([B, 1n], n)) acc *= v;
        expect(np.prod(a)).toBe(BigInt.asIntN(64, acc));
      });

      it(`ptp subtracts in 64 bits n=${n}`, () => {
        const a = np.array(cycle([B + 3n, B], n), 'int64');
        expect(np.ptp(a)).toBe(3n);
      });
    }

    it('mean accumulates in float64 and does not reuse the wrapping sum', () => {
      // NumPy on this input: sum() wraps all the way down to 6, while mean()
      // accumulates in float64 and reports 4.611686018427388e18. Deriving the
      // mean from the integer sum would give 6/8 = 0.75.
      const a = np.array(cycle([I64_MAX, I64_MAX, 2n, 3n], 8), 'int64');
      expect(np.sum(a)).toBe(6n);
      const mean = np.mean(a) as number;
      expect(typeof mean).toBe('number');
      expect(mean).toBe(4.611686018427388e18);
    });

    it('sum along an axis keeps 64-bit column totals', () => {
      // The strided kernel accumulated in f64; columns here exceed 2^53.
      const a = np.reshape(np.array(cycle([B, B + 1n, B + 2n, B + 3n], 128), 'int64'), [32, 4]);
      const cols = np.sum(a, 0).toArray();
      expect(cols).toEqual([B * 32n, (B + 1n) * 32n, (B + 2n) * 32n, (B + 3n) * 32n]);
    });

    it('sum along an axis wraps in int64', () => {
      const a = np.reshape(np.array(cycle([I64_MAX, 1n], 64), 'int64'), [32, 2]);
      const cols = np.sum(a, 0).toArray();
      expect(cols).toEqual([BigInt.asIntN(64, I64_MAX * 32n), 32n]);
    });

    it('apply_along_axis handles a bigint-returning reduction', () => {
      const a = np.array(
        [
          [B, B + 1n],
          [B + 2n, B + 3n],
        ],
        'int64',
      );
      const out = np.apply_along_axis((s: never) => np.sum(s), 0, a);
      expect(out.shape).toEqual([2]);
      expect(out.toArray()).toEqual([B + B + 2n, B + 1n + B + 3n]);
    });
  });

  describe('set operations distinguish adjacent 64-bit values', () => {
    it('unique does not merge neighbours above 2^53', () => {
      // All four differ, but Number() maps them onto three distinct doubles.
      const a = np.array([B, B + 1n, B + 2n, B + 3n, B, B + 1n], 'int64');
      expect(np.unique(a).toArray()).toEqual([B, B + 1n, B + 2n, B + 3n]);
    });

    it('unique reports indices, inverse and counts against exact values', () => {
      const a = np.array([B + 1n, B, B + 1n, B + 2n], 'int64');
      const r = np.unique(a, true, true, true) as {
        values: { toArray(): bigint[] };
        indices: { toArray(): number[] };
        inverse: { toArray(): number[] };
        counts: { toArray(): number[] };
      };
      expect(r.values.toArray()).toEqual([B, B + 1n, B + 2n]);
      expect(r.indices.toArray()).toEqual([1, 0, 3]);
      expect(r.inverse.toArray()).toEqual([1, 0, 1, 2]);
      expect(r.counts.toArray()).toEqual([1, 2, 1]);
    });

    it('isin answers membership per exact value', () => {
      const a = np.array([B, B + 1n, B + 2n, B + 3n], 'int64');
      const test = np.array([B + 1n, B + 3n], 'int64');
      // bool arrays marshal as 0/1 through toArray(), same as np.equal.
      expect(np.isin(a, test).toArray()).toEqual([0, 1, 0, 1]);
    });

    it('intersect1d and setdiff1d split on exact values', () => {
      const a = np.array([B, B + 1n, B + 2n, B + 3n], 'int64');
      const b = np.array([B + 1n, B + 3n], 'int64');
      expect(np.intersect1d(a, b).toArray()).toEqual([B + 1n, B + 3n]);
      expect(np.setdiff1d(a, b).toArray()).toEqual([B, B + 2n]);
    });

    it('union1d and setxor1d no longer throw on 64-bit input', () => {
      const a = np.array([B, B + 1n, I64_MAX], 'int64');
      const b = np.array([B + 1n, B + 2n], 'int64');
      expect(np.union1d(a, b).toArray()).toEqual([B, B + 1n, B + 2n, I64_MAX]);
      expect(np.setxor1d(a, b).toArray()).toEqual([B, B + 2n, I64_MAX]);
    });

    it('uint64 set ops work at the top of the range', () => {
      const a = np.array([U64_MAX, U64_MAX - 1n, 1n], 'uint64');
      const b = np.array([U64_MAX, 2n], 'uint64');
      expect(np.intersect1d(a, b).toArray()).toEqual([U64_MAX]);
      expect(np.union1d(a, b).toArray()).toEqual([1n, 2n, U64_MAX - 1n, U64_MAX]);
    });
  });

  describe('searchsorted', () => {
    for (const n of SIZES) {
      it(`finds the insertion point by exact key n=${n}`, () => {
        // Strictly increasing, entirely above 2^53, spaced so a rounded compare
        // cannot separate neighbours.
        const keys = Array.from({ length: n }, (_, i) => B + BigInt(i));
        const a = np.array(keys, 'int64');
        const q = np.array([B, B + 1n, B + BigInt(n - 1)], 'int64');
        expect(np.searchsorted(a, q).toArray()).toEqual([0, 1, n - 1]);
        expect(np.searchsorted(a, q, 'right').toArray()).toEqual([1, 2, n]);
      });
    }
  });
});
