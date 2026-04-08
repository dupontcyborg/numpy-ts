/**
 * DType Sweep: Scalar reduction functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  expectBothReject,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

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

const nanOps: { name: string; fn: (a: any) => any }[] = [
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

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data =
      dtype === 'bool' ? [1, 0, 1, 1, 0, 1] : isComplex(dtype) ? [1, 2, 3, 4, 5, 6] : SMALL_DATA;

    for (const { name, npFn } of ops) {
      const castExpr =
        isComplex(dtype) && (name === 'sum' || name === 'prod')
          ? `complex(np.${npFn || name}(a))`
          : `float(np.${npFn || name}(a))`;
      snippets[`${name}_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${castExpr}`;
    }

    const nanData = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
    for (const { name } of nanOps) {
      snippets[`${name}_${dtype}`] = `
a = np.array(${JSON.stringify(nanData)}, dtype=${npDtype(dtype)})
result = float(np.${name}(a))`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Reductions (scalar)', () => {
  for (const { name, fn } of ops) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data =
            dtype === 'bool'
              ? [1, 0, 1, 1, 0, 1]
              : isComplex(dtype)
                ? [1, 2, 3, 4, 5, 6]
                : SMALL_DATA;
          const a = array(data, dtype);
          // ptp uses subtract internally — NumPy rejects bool subtract
          if (name === 'ptp' && dtype === 'bool') {
            const pyCode = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = float(np.ptp(a))`;
            const _r = expectBothReject(
              'ptp uses subtract, not supported for bool',
              () => fn(a),
              pyCode
            );
            if (_r === 'both-reject') return;
          }
          const jsResult = fn(a);
          const py = oracle.get(`${name}_${dtype}`)!;
          expect(Number(jsResult)).toBeCloseTo(Number(py.value), 3);
        });
      }
    });
  }
});

describe('DType Sweep: Nan-aware reductions', () => {
  for (const { name, fn } of nanOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
          const a = array(data, dtype);
          const jsResult = fn(a);
          const py = oracle.get(`${name}_${dtype}`)!;
          expect(Number(jsResult)).toBeCloseTo(Number(py.value), 3);
        });
      }
    });
  }
});
