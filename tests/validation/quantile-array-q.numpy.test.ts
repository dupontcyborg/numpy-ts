/**
 * NumPy validation tests for percentile, quantile, nanpercentile and nanquantile with an NDArray
 * q, NaN propagation in the non-nan variants, and q range errors.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import * as np from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

const NAN = Number.NaN;

type Fn = 'percentile' | 'quantile' | 'nanpercentile' | 'nanquantile';
const FUNCS: Fn[] = ['percentile', 'quantile', 'nanpercentile', 'nanquantile'];
// Loosely typed so one call site covers all four functions and every q shape.
const FN: Record<Fn, (...args: unknown[]) => never> = {
  percentile: np.percentile as never,
  quantile: np.quantile as never,
  nanpercentile: np.nanpercentile as never,
  nanquantile: np.nanquantile as never,
};
const scaleOf = (fn: Fn): number => (fn.endsWith('percentile') ? 100 : 1);

function values(n: number, withNaN: boolean): number[] {
  return Array.from({ length: n }, (_, i) => {
    if (withNaN && i % 5 === 2) return NAN;
    return ((i * 7) % 11) + i / 10;
  });
}

function qValues(n: number, scale: number): number[] {
  return Array.from({ length: n }, (_, i) => (((i * 37 + 11) % 101) / 100) * scale);
}

const lit = (v: number): string => (Number.isNaN(v) ? 'np.nan' : String(v));

function pyArr(vals: number[], shape: number[], dtype: string): string {
  return `np.array([${vals.map(lit).join(', ')}], dtype='${dtype}').reshape(${JSON.stringify(shape)})`;
}

interface Case {
  fn: Fn;
  vals: number[];
  shape: number[];
  dtype?: string;
  qShape: number[];
  qDtype?: string;
  axis?: number;
  keepdims?: boolean;
}

function run(c: Case): void {
  const dtype = c.dtype ?? 'float64';
  const qDtype = c.qDtype ?? 'float64';
  const qn = c.qShape.reduce((x, y) => x * y, 1);
  const qv = qValues(qn, scaleOf(c.fn));
  const a = np.array(c.vals, dtype as never).reshape(c.shape);
  const q = np.array(qv, qDtype as never).reshape(c.qShape);
  const r = FN[c.fn](a, q, c.axis, c.keepdims);
  const expected = runNumPy(`
a = ${pyArr(c.vals, c.shape, dtype)}
q = ${pyArr(qv, c.qShape, qDtype)}
result = np.${c.fn}(a, q, axis=${c.axis ?? 'None'}, keepdims=${c.keepdims ? 'True' : 'False'})
`);
  expect(Array.from(r.shape)).toEqual(expected.shape);
  expect(r.dtype).toBe(expected.dtype);
  const rtol = expected.dtype === 'float32' ? 1e-5 : 1e-9;
  expect(arraysClose(r.toArray(), expected.value, rtol, rtol)).toBe(true);
}

describe('NumPy Validation: percentile and quantile with NDArray q', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  for (const fn of FUNCS) {
    describe(fn, () => {
      const qShapes: number[][] = [[], [1], [3], [2, 3], [0], [1, 2]];
      const dataShapes: number[][] = [[7], [2, 3], [3, 4], [2, 3, 4]];
      for (const shape of dataShapes) {
        const n = shape.reduce((x, y) => x * y, 1);
        const axes: (number | undefined)[] = [undefined];
        for (let ax = -shape.length; ax < shape.length; ax++) axes.push(ax);
        for (const withNaN of [false, true]) {
          for (const qShape of qShapes) {
            for (const axis of axes) {
              for (const keepdims of [false, true]) {
                // Trim the cross product: full axis sweep only for the 3-D shape and for q
                // shapes that exercise a distinct stacking layout.
                if (shape.length < 3 && qShape.length > 1 && axis !== undefined && axis < 0) {
                  continue;
                }
                it(`data ${JSON.stringify(shape)}${withNaN ? ' NaN' : ''} q ${JSON.stringify(qShape)} axis ${axis} keepdims ${keepdims}`, () => {
                  run({
                    fn,
                    vals: values(n, withNaN),
                    shape,
                    qShape,
                    axis,
                    keepdims,
                  });
                });
              }
            }
          }
        }
      }

      it('all-NaN lanes', () => {
        run({
          fn,
          vals: [NAN, NAN, 1, 2, NAN, NAN, 3, 4],
          shape: [2, 4],
          qShape: [2],
          axis: 1,
        });
        run({ fn, vals: [NAN, NAN, NAN], shape: [3], qShape: [2] });
      });

      it('integer data gives float64', () => {
        for (const dtype of ['int8', 'int32', 'uint16', 'int64']) {
          run({ fn, vals: [3, 1, 4, 1, 5, 9, 2, 6], shape: [2, 4], dtype, qShape: [3], axis: 0 });
        }
      });

      it('float32 data with float64 q widens to float64', () => {
        run({ fn, vals: values(12, false), shape: [3, 4], dtype: 'float32', qShape: [3], axis: 1 });
      });

      it('float32 data with float32 q stays float32', () => {
        run({
          fn,
          vals: values(12, false),
          shape: [3, 4],
          dtype: 'float32',
          qDtype: 'float32',
          qShape: [3],
          axis: 1,
        });
      });

      it('q at the ends of its range', () => {
        const scale = scaleOf(fn);
        const a = np.array(values(9, false)).reshape([3, 3]);
        const q = np.array([0, scale]);
        const r = FN[fn](a, q, 0);
        const expected = runNumPy(`
a = ${pyArr(values(9, false), [3, 3], 'float64')}
result = np.${fn}(a, np.array([0, ${scale}]), axis=0)
`);
        expect(arraysClose(r.toArray(), expected.value, 1e-9, 1e-9)).toBe(true);
      });

      it('NDArray method form', () => {
        const a = np.array(values(12, true)).reshape([3, 4]);
        const q = np.array(qValues(4, scaleOf(fn))).reshape([2, 2]);
        const r = (a as unknown as Record<Fn, (...args: unknown[]) => NDArray>)[fn]!(q, 1, true);
        const expected = runNumPy(`
a = ${pyArr(values(12, true), [3, 4], 'float64')}
q = ${pyArr(qValues(4, scaleOf(fn)), [2, 2], 'float64')}
result = np.${fn}(a, q, axis=1, keepdims=True)
`);
        expect(Array.from(r.shape)).toEqual(expected.shape);
        expect(arraysClose(r.toArray(), expected.value, 1e-9, 1e-9)).toBe(true);
      });

      describe('range errors', () => {
        const scale = scaleOf(fn);
        const a = np.array([1, 2, 3, 4]);
        for (const bad of [scale * 1.01, -0.01, NAN, Infinity]) {
          it(`scalar q ${bad}`, () => {
            expect(() => FN[fn](a, bad)).toThrow(/must be in the range/);
            expect(() =>
              runNumPy(
                `result = np.${fn}(np.array([1,2,3,4]), ${lit(bad)})`.replace('Infinity', 'np.inf'),
              ),
            ).toThrow(/must be in the range/);
          });
          it(`array q containing ${bad}`, () => {
            const q = np.array([scale / 2, bad]);
            expect(() => FN[fn](a, q)).toThrow(/must be in the range/);
          });
        }
        it('rejects q with more than two dimensions', () => {
          const q = np.array([scale / 2, scale / 4]).reshape([2, 1, 1]);
          expect(() => FN[fn](a, q)).toThrow(/q must be a scalar or 1d/);
          expect(() =>
            runNumPy(
              `result = np.${fn}(np.array([1,2,3,4]), np.array([${scale / 2}, ${scale / 4}]).reshape(2, 1, 1))`,
            ),
          ).toThrow(/q must be a scalar or 1d/);
        });
        it('rejects a plain array q', () => {
          expect(() => FN[fn](a, [scale / 2])).toThrow(TypeError);
        });
        it('rejects a complex q', () => {
          const q = np.array([0.5]).astype('complex128');
          expect(() => FN[fn](a, q)).toThrow(/not supported for complex/);
        });
      });
    });
  }

  describe('scalar q propagates NaN like NumPy', () => {
    const cases: [Fn, number][] = [
      ['percentile', 30],
      ['quantile', 0.3],
    ];
    for (const [fn, q] of cases) {
      for (const shape of [[7], [3, 4], [2, 3, 4]]) {
        const n = shape.reduce((x, y) => x * y, 1);
        for (const axis of [undefined, 0, -1]) {
          it(`${fn} ${JSON.stringify(shape)} axis ${axis}`, () => {
            const vals = values(n, true);
            const a = np.array(vals).reshape(shape);
            const r = FN[fn](a, q, axis);
            const expected = runNumPy(`
a = ${pyArr(vals, shape, 'float64')}
result = np.${fn}(a, ${q}, axis=${axis ?? 'None'})
`);
            const got = typeof r === 'number' ? r : (r as NDArray).toArray();
            expect(arraysClose(got, expected.value, 1e-9, 1e-9)).toBe(true);
          });
        }
      }
      it(`${fn} NaN lane in float32 and a non-contiguous view`, () => {
        const vals = values(12, true);
        const a = np.array(vals, 'float32').reshape([3, 4]);
        const r = FN[fn](a.T, q, 0);
        const expected = runNumPy(`
a = ${pyArr(vals, [3, 4], 'float32')}
result = np.${fn}(a.T, ${q}, axis=0)
`);
        expect(arraysClose(r.toArray(), expected.value, 1e-5, 1e-5)).toBe(true);
        const whole = FN[fn](a.T, q);
        expect(whole).toBeNaN();
      });
    }
  });
});
