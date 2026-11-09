/**
 * NumPy validation: isclose and allclose with equal_nan, infinities, array tolerances,
 * broadcasting, scalar forms and every dtype pair.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { allclose, array, isclose } from '../../src';
import { DTYPES, expectSame, lcg, sample, toJs, toPy } from './_dtype-grid';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

const SHAPE_PAIRS: [number[], number[]][] = [
  [[6], [6]],
  [
    [2, 3],
    [2, 3],
  ],
  [[2, 3], [3]],
  [
    [2, 1],
    [1, 3],
  ],
  [[1], [4]],
  [
    [2, 2, 3],
    [2, 3],
  ],
  [[0], [0]],
];

describe('NumPy Validation: isclose and allclose options', { timeout: 120_000 }, () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('matches NumPy for every dtype pair, with and without equal_nan', () => {
    const rng = lcg(77);
    for (const da of DTYPES) {
      for (const db of DTYPES) {
        const [sha, shb] = SHAPE_PAIRS[Math.floor(rng() * SHAPE_PAIRS.length)]!;
        const sa = sample(rng, da, sha);
        const sb = sample(rng, db, shb);
        const a = toJs(sa, da);
        const b = toJs(sb, db);
        const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
        for (const equalNan of [false, true]) {
          const flag = equalNan ? 'True' : 'False';
          const label = `${da}/${db} ${sha}|${shb} equal_nan=${flag} a=${sa.re} b=${sb.re}`;
          expectSame(
            isclose(a, b, 1e-5, 1e-8, equalNan),
            runNumPy(`${py}result = np.isclose(a, b, equal_nan=${flag})`),
            `isclose ${label}`,
          );
          expect(allclose(a, b, 1e-5, 1e-8, equalNan), `allclose ${label}`).toBe(
            runNumPy(`${py}result = bool(np.allclose(a, b, equal_nan=${flag}))`).value,
          );
        }
      }
    }
  });

  it('treats NaN as equal only with equal_nan, and same-sign infinities as close', () => {
    const a = array([Number.NaN, Infinity, -Infinity, Infinity, 1, Number.NaN]);
    const b = array([Number.NaN, Infinity, -Infinity, -Infinity, Number.NaN, 1]);
    const want = (flag: string) =>
      runNumPy(
        `result = np.isclose(np.array([np.nan, np.inf, -np.inf, np.inf, 1, np.nan]), np.array([np.nan, np.inf, -np.inf, -np.inf, np.nan, 1]), equal_nan=${flag})`,
      );
    expect(isclose(a, b).toArray()).toEqual([0, 1, 1, 0, 0, 0]);
    expectSame(isclose(a, b), want('False'), 'default');
    expectSame(isclose(a, b, 1e-5, 1e-8, true), want('True'), 'equal_nan');
    expect(allclose(a, b, 1e-5, 1e-8, true)).toBe(false);
    expect(allclose(array([Number.NaN, 1]), array([Number.NaN, 1]), 1e-5, 1e-8, true)).toBe(true);
    expect(allclose(array([Number.NaN, 1]), array([Number.NaN, 1]))).toBe(false);
  });

  it('applies array rtol and atol with broadcasting', () => {
    const rng = lcg(88);
    for (const da of ['float64', 'float32', 'int32', 'complex128'] as const) {
      for (const tolDtype of ['float64', 'float32', 'int32'] as const) {
        const sa = sample(rng, da, [2, 3], { nan: false });
        const sb = sample(rng, da, [2, 3], { nan: false });
        const a = toJs(sa, da);
        const b = toJs(sb, da);
        const rtolShapes: number[][] = [[3], [2, 1], [1], [2, 3]];
        for (const rs of rtolShapes) {
          const rtolS = {
            re:
              rs.reduce((x, y) => x * y, 1) === 1
                ? [0.5]
                : Array.from(
                    { length: rs.reduce((x, y) => x * y, 1) },
                    (_, i) => [0, 1, 2, 0.5, 3, 1][i % 6]!,
                  ),
            shape: rs,
          };
          const atolS = { re: [0, 1, 0.5], shape: [3] };
          const rtol = toJs(rtolS, tolDtype);
          const atol = toJs(atolS, tolDtype);
          const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, da)}\nrt = ${toPy(rtolS, tolDtype)}\nat = ${toPy(atolS, tolDtype)}\n`;
          const label = `${da} tol ${tolDtype} rtol shape ${rs} a=${sa.re} b=${sb.re}`;
          expectSame(
            isclose(a, b, rtol, atol),
            runNumPy(`${py}result = np.isclose(a, b, rtol=rt, atol=at)`),
            label,
          );
          expectSame(
            isclose(a, b, rtol, 1e-8),
            runNumPy(`${py}result = np.isclose(a, b, rtol=rt, atol=1e-8)`),
            `${label} rtol only`,
          );
          expectSame(
            isclose(a, b, 1e-5, atol),
            runNumPy(`${py}result = np.isclose(a, b, atol=at)`),
            `${label} atol only`,
          );
        }
      }
    }
  });

  it('matches NumPy with scalar b for every dtype and tolerance', () => {
    const rng = lcg(99);
    for (const da of DTYPES) {
      const sa = sample(rng, da, [3, 4]);
      const a = toJs(sa, da);
      const py = `a = ${toPy(sa, da)}\n`;
      for (const b of [1, 0, 0.5, -2, 1.00001, Infinity, Number.NaN]) {
        for (const [rtol, atol] of [
          [1e-5, 1e-8],
          [0, 0],
          [0.1, 0],
          [0, 1],
        ] as const) {
          const bPy = Number.isNaN(b)
            ? "float('nan')"
            : b === Infinity
              ? "float('inf')"
              : String(b);
          for (const equalNan of [false, true]) {
            const flag = equalNan ? 'True' : 'False';
            expectSame(
              isclose(a, b, rtol, atol, equalNan),
              runNumPy(
                `${py}result = np.isclose(a, ${bPy}, rtol=${rtol}, atol=${atol}, equal_nan=${flag})`,
              ),
              `${da} b=${b} rtol=${rtol} atol=${atol} equal_nan=${flag} a=${sa.re}`,
            );
          }
        }
      }
    }
  });

  it('compares 64-bit integers as float64 does in NumPy', () => {
    const big = 2n ** 60n;
    const a = array([big, big + 1n, 5n, -big], 'int64');
    const b = array([big + 1n, big, 5n, -big - 7n], 'int64');
    for (const [rtol, atol] of [
      [0, 0],
      [1e-5, 1e-8],
      [0, 4],
    ] as const) {
      expectSame(
        isclose(a, b, rtol, atol),
        runNumPy(`a = np.array([2**60, 2**60 + 1, 5, -2**60], dtype='int64')
b = np.array([2**60 + 1, 2**60, 5, -2**60 - 7], dtype='int64')
result = np.isclose(a, b, rtol=${rtol}, atol=${atol})`),
        `int64 rtol=${rtol} atol=${atol}`,
      );
    }
  });

  it('works on non-contiguous views', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6.00001],
    ]);
    const b = array([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expectSame(
      isclose(a, b.T),
      runNumPy(`a = np.array([[1, 2, 3], [4, 5, 6.00001]])
b = np.array([[1, 4], [2, 5], [3, 6]])
result = np.isclose(a, b.T)`),
      'transposed',
    );
  });

  it('rejects shapes that cannot broadcast', () => {
    expect(() => isclose(array([1, 2, 3]), array([1, 2]))).toThrow(/broadcast/);
    expect(() => isclose(array([1, 2, 3]), array([1, 2, 3]), array([1, 2]))).toThrow(/broadcast/);
  });
});
