/**
 * DType Sweep: Reduction functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Reductions (scalar)', () => {
  const ops: { name: string; fn: (a: any) => any; npFn?: string }[] = [
    { name: 'sum', fn: np.sum },
    { name: 'prod', fn: np.prod },
    { name: 'any', fn: np.any },
    { name: 'all', fn: np.all },
    { name: 'count_nonzero', fn: np.count_nonzero },
    { name: 'mean', fn: np.mean },
    { name: 'std', fn: np.std },
    { name: 'variance', fn: np.variance, npFn: 'var' },
    { name: 'max', fn: np.max },
    { name: 'min', fn: np.min },
    { name: 'argmax', fn: np.argmax },
    { name: 'argmin', fn: np.argmin },
    { name: 'median', fn: np.median },
    { name: 'ptp', fn: np.ptp },
  ];

  for (const { name, fn, npFn } of ops) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = dtype === 'bool' ? [1, 0, 1, 1, 0, 1] : isComplex(dtype) ? [1, 2, 3, 4, 5, 6] : SMALL_DATA;
          const a = array(data, dtype);
          const jsResult = fn(a);
          const castExpr = isComplex(dtype) && (name === 'sum' || name === 'prod')
            ? `complex(np.${npFn || name}(a))`
            : `float(np.${npFn || name}(a))`;
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${castExpr}
          `);
          expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 3);
        });
      }
    });
  }
});

describe('DType Sweep: Nan-aware reductions', () => {
  const ops: { name: string; fn: (a: any) => any }[] = [
    { name: 'nansum', fn: np.nansum },
    { name: 'nanprod', fn: np.nanprod },
    { name: 'nanmean', fn: np.nanmean },
    { name: 'nanstd', fn: np.nanstd },
    { name: 'nanvar', fn: np.nanvar },
    { name: 'nanmax', fn: np.nanmax },
    { name: 'nanmin', fn: np.nanmin },
    { name: 'nanargmax', fn: np.nanargmax },
    { name: 'nanargmin', fn: np.nanargmin },
    { name: 'nanmedian', fn: np.nanmedian },
  ];

  for (const { name, fn } of ops) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
          const a = array(data, dtype);
          const jsResult = fn(a);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.${name}(a))
          `);
          expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 3);
        });
      }
    });
  }
});

describe('DType Sweep: Quantile/percentile', () => {
  for (const dtype of ALL_DTYPES) {
    it(`quantile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.quantile(a, 0.5);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.quantile(a, 0.5))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`nanquantile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.nanquantile(a, 0.5);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.nanquantile(a, 0.5))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`percentile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.percentile(a, 50);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.percentile(a, 50))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`nanpercentile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.nanpercentile(a, 50);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.nanpercentile(a, 50))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });
  }
});

describe('DType Sweep: Cumulative & misc reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`cumsum ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.cumsum(a);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.cumsum(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`cumprod ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 1, 0, 1] : [1, 2, 3, 4];
      const a = array(data, dtype);
      const jsResult = np.cumprod(a);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.cumprod(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`average ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.average(a);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.average(a))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`ediff1d ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : SMALL_DATA;
      const a = array(data, dtype);
      const jsResult = np.ediff1d(a);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.ediff1d(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });
  }
});

describe('DType Sweep: Axis reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`sum axis=0 ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0, 1], [0, 1, 0]] : SMALL_2D;
      const a = array(data, dtype);
      const jsResult = np.sum(a, 0);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.sum(a, axis=0).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`mean axis=1 ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0, 1], [0, 1, 0]] : SMALL_2D;
      const a = array(data, dtype);
      const jsResult = np.mean(a, 1);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.mean(a, axis=1)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`max axis=0 ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0, 1], [0, 1, 0]] : SMALL_2D;
      const a = array(data, dtype);
      const jsResult = np.max(a, 0);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.max(a, axis=0).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
