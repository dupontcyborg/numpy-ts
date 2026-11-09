/**
 * NumPy validation: the invert, assume_unique and return_indices options of the set functions,
 * swept over every dtype pair, shapes, NaN, infinities and complex values. assume_unique is also
 * fed inputs that are not unique, where NumPy's answer depends on its internal method choice.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { array, in1d, intersect1d, isin, setdiff1d, setxor1d, union1d } from '../../src';
import {
  DTYPES,
  expectSame,
  lcg,
  sample,
  skipMixed,
  toJs,
  toPy,
  uniqueSample,
} from './_dtype-grid';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

const SHAPES: number[][] = [[0], [1], [6], [2, 3], [2, 2, 3], [1, 4]];
const SECOND_SIZES = [0, 1, 4, 20, 30];

describe('NumPy Validation: set function options', { timeout: 120_000 }, () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('isin and in1d: invert', () => {
    it('is the negation of the default for every dtype pair and shape', () => {
      const rng = lcg(11);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          const shape = SHAPES[Math.floor(rng() * SHAPES.length)]!;
          const sa = sample(rng, da, shape);
          const sb = sample(rng, db, [SECOND_SIZES[Math.floor(rng() * SECOND_SIZES.length)]!]);
          const a = toJs(sa, da);
          const b = toJs(sb, db);
          const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
          const label = `${da}/${db} shape ${shape.join('x')}`;
          for (const invert of [false, true]) {
            const flag = invert ? 'True' : 'False';
            expectSame(
              isin(a, b, false, invert),
              runNumPy(`${py}result = np.isin(a, b, invert=${flag})`),
              `isin invert=${flag} ${label}`,
            );
            expectSame(
              in1d(a, b, false, invert),
              runNumPy(`${py}result = np.in1d(a, b, invert=${flag})`),
              `in1d invert=${flag} ${label}`,
            );
          }
        }
      }
    });

    it('keeps the first argument shape for isin and flattens for in1d', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([2, 3]);
      expect(Array.from(isin(a, b, false, true).shape)).toEqual([2, 2]);
      expect(isin(a, b, false, true).toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
      expect(Array.from(in1d(a, b).shape)).toEqual([4]);
      expect(in1d(a, b, false, true).toArray()).toEqual([1, 0, 0, 1]);
    });

    it('reports NaN as absent, so invert reports it present', () => {
      const a = array([Number.NaN, 1, Number.NaN]);
      const b = array([Number.NaN, 1]);
      const np = (inv: string) =>
        runNumPy(
          `result = np.isin(np.array([np.nan, 1, np.nan]), np.array([np.nan, 1]), invert=${inv})`,
        );
      expectSame(isin(a, b), np('False'), 'isin');
      expectSame(isin(a, b, false, true), np('True'), 'isin invert');
    });
  });

  describe('assume_unique', () => {
    const FUNCTIONS: [string, (a: any, b: any, au: boolean) => any, string][] = [
      ['isin', (a, b, au) => isin(a, b, au), 'np.isin(a, b, assume_unique=True)'],
      [
        'isin invert',
        (a, b, au) => isin(a, b, au, true),
        'np.isin(a, b, assume_unique=True, invert=True)',
      ],
      ['in1d', (a, b, au) => in1d(a, b, au), 'np.in1d(a, b, assume_unique=True)'],
      ['setdiff1d', (a, b, au) => setdiff1d(a, b, au), 'np.setdiff1d(a, b, assume_unique=True)'],
      ['setxor1d', (a, b, au) => setxor1d(a, b, au), 'np.setxor1d(a, b, assume_unique=True)'],
      [
        'intersect1d',
        (a, b, au) => intersect1d(a, b, au),
        'np.intersect1d(a, b, assume_unique=True)',
      ],
    ];

    it('equals the default for truly unique input, for every dtype pair', () => {
      const rng = lcg(21);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          if (skipMixed(rng, da, db, 0.5)) continue;
          const sa = uniqueSample(rng, da, 1 + Math.floor(rng() * 6));
          const sb = uniqueSample(rng, db, Math.floor(rng() * 6));
          const a = toJs(sa, da);
          const b = toJs(sb, db);
          const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
          for (const [name, fn, expr] of FUNCTIONS) {
            const withFlag = runNumPy(`${py}result = ${expr}`);
            // setdiff1d keeps the order of ar1 with the flag, so only results NumPy sorts or
            // maps one-to-one onto ar1 are compared against the default call.
            const sameAsDefault = name !== 'setdiff1d';
            if (sameAsDefault) {
              expect(fn(a, b, true).toArray(), `${name} ${da}/${db} vs default`).toEqual(
                fn(a, b, false).toArray(),
              );
            }
            expectSame(fn(a, b, true), withFlag, `${name} assume_unique ${da}/${db}`);
          }
        }
      }
    });

    it('follows NumPy on input that is not unique, in each of its membership methods', () => {
      const rng = lcg(33);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          if (skipMixed(rng, da, db, 0.5)) continue;
          for (const wide of [rng() < 0.5]) {
            const sa = sample(rng, da, [[0], [1], [5], [12]][Math.floor(rng() * 4)]!, { wide });
            const sb = sample(rng, db, [SECOND_SIZES[Math.floor(rng() * SECOND_SIZES.length)]!], {
              wide,
            });
            const a = toJs(sa, da);
            const b = toJs(sb, db);
            const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
            for (const [name, fn, expr] of FUNCTIONS) {
              expectSame(
                fn(a, b, true),
                runNumPy(`${py}result = ${expr}`),
                `${name} assume_unique ${da}/${db} wide=${wide} a=${sa.re} b=${sb.re}`,
              );
            }
          }
        }
      }
    });

    it('keeps the order of ar1 for setdiff1d', () => {
      const a = array([5, 3, 9, 1]);
      const b = array([3, 1]);
      expect(setdiff1d(a, b, true).toArray()).toEqual([5, 9]);
      expect(setdiff1d(a, b).toArray()).toEqual([5, 9]);
      const c = array([9, 5, 3]);
      expect(setdiff1d(c, array([5]), true).toArray()).toEqual([9, 3]);
      expect(setdiff1d(c, array([5])).toArray()).toEqual([3, 9]);
    });
  });

  describe('default behaviour, NaN and promoted dtypes', () => {
    it('matches NumPy for random input of every dtype pair', () => {
      const rng = lcg(44);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          const sa = sample(rng, da, [Math.floor(rng() * 8)]);
          const sb = sample(rng, db, [Math.floor(rng() * 8)]);
          const a = toJs(sa, da);
          const b = toJs(sb, db);
          const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
          const label = `${da}/${db} a=${sa.re} b=${sb.re}`;
          expectSame(
            intersect1d(a, b),
            runNumPy(`${py}result = np.intersect1d(a, b)`),
            `intersect1d ${label}`,
          );
          expectSame(
            setxor1d(a, b),
            runNumPy(`${py}result = np.setxor1d(a, b)`),
            `setxor1d ${label}`,
          );
          expectSame(
            setdiff1d(a, b),
            runNumPy(`${py}result = np.setdiff1d(a, b)`),
            `setdiff1d ${label}`,
          );
          expectSame(union1d(a, b), runNumPy(`${py}result = np.union1d(a, b)`), `union1d ${label}`);
          expectSame(isin(a, b), runNumPy(`${py}result = np.isin(a, b)`), `isin ${label}`);
        }
      }
    });
  });

  describe('intersect1d: return_indices', () => {
    function check(
      sa: ReturnType<typeof sample>,
      da: (typeof DTYPES)[number],
      sb: ReturnType<typeof sample>,
      db: (typeof DTYPES)[number],
      assumeUnique: boolean,
      label: string,
    ) {
      const a = toJs(sa, da);
      const b = toJs(sb, db);
      const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
      const au = assumeUnique ? 'True' : 'False';
      const [common, i1, i2] = intersect1d(a, b, assumeUnique, true);
      const run = (part: number) =>
        runNumPy(
          `${py}r = np.intersect1d(a, b, assume_unique=${au}, return_indices=True)\nresult = r[${part}]`,
        );
      expectSame(common, run(0), `${label} values`);
      // Index arrays are int32 here and intp in NumPy, so only their values are compared.
      for (const [got, part] of [
        [i1, 1],
        [i2, 2],
      ] as const) {
        const want = run(part);
        expect(Array.from(got.shape), `${label} index ${part} shape`).toEqual(want.shape);
        expect(got.toArray().map(Number), `${label} index ${part}`).toEqual(
          (want.value as number[]).map(Number),
        );
      }
    }

    it('matches NumPy for 1-D and N-D input of every dtype pair', () => {
      const rng = lcg(55);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          if (skipMixed(rng, da, db, 0.5)) continue;
          const shapeA = [[6], [2, 3], [2, 2, 2], [0]][Math.floor(rng() * 4)]!;
          const shapeB = [[7], [3, 2], [1, 5], [0]][Math.floor(rng() * 4)]!;
          const sa = sample(rng, da, shapeA);
          const sb = sample(rng, db, shapeB);
          check(sa, da, sb, db, false, `${da}/${db} ${shapeA}/${shapeB} a=${sa.re} b=${sb.re}`);
        }
      }
    });

    it('matches NumPy with assume_unique on unique and non-unique input', () => {
      const rng = lcg(66);
      for (const da of DTYPES) {
        for (const db of DTYPES) {
          if (skipMixed(rng, da, db, 0.5)) continue;
          const ua = uniqueSample(rng, da, 5);
          const ub = uniqueSample(rng, db, 5);
          check(ua, da, ub, db, true, `unique ${da}/${db}`);
          const sa = sample(rng, da, [6]);
          const sb = sample(rng, db, [7]);
          check(sa, da, sb, db, true, `non-unique ${da}/${db} a=${sa.re} b=${sb.re}`);
        }
      }
    });

    it('returns first-occurrence indices into the flattened inputs', () => {
      const a = array([
        [3, 1],
        [1, 2],
      ]);
      const b = array([2, 1, 1, 7]);
      const [common, i1, i2] = intersect1d(a, b, false, true);
      expect(common.toArray()).toEqual([1, 2]);
      expect(i1.toArray()).toEqual([1, 3]);
      expect(i2.toArray()).toEqual([1, 0]);
    });

    it('returns a bare array by default and a tuple only when asked', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 3, 4]);
      expect(Array.isArray(intersect1d(a, b))).toBe(false);
      expect(Array.isArray(intersect1d(a, b, false, false))).toBe(false);
      const t = intersect1d(a, b, undefined, true);
      expect(t).toHaveLength(3);
      const dynamic: boolean = Math.random() < 2;
      expect(Array.isArray(intersect1d(a, b, false, dynamic))).toBe(true);
    });
  });
});
