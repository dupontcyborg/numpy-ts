/**
 * DType Sweep: Quantile, cumulative, and axis reduction functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  pyScalarCast,
  pyArrayCast,
  scalarClose,
  expectBothReject,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  // Build all Python snippets up front
  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const sc = pyScalarCast(dtype);
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1],
            [0, 1, 0],
          ]
        : SMALL_2D;

    // Quantile/percentile (scalar results)
    snippets[`quantile_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.quantile(a, 0.5))`;

    snippets[`nanquantile_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.nanquantile(a, 0.5))`;

    snippets[`percentile_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.percentile(a, 50))`;

    snippets[`nanpercentile_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.nanpercentile(a, 50))`;

    // Cumulative (array results with _result_orig)
    snippets[`cumsum_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.cumsum(a)
result = _result_orig.astype(${ac})`;

    const cumprodData = dtype === 'bool' ? [1, 1, 0, 1] : [1, 2, 3, 4];
    snippets[`cumprod_${dtype}`] = `
a = np.array(${JSON.stringify(cumprodData)}, dtype=${npDtype(dtype)})
_result_orig = np.cumprod(a)
result = _result_orig.astype(${ac})`;

    snippets[`average_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.average(a))`;

    const ediffData = dtype === 'bool' ? [1, 0, 1, 0] : SMALL_DATA;
    snippets[`ediff1d_${dtype}`] = `
a = np.array(${JSON.stringify(ediffData)}, dtype=${npDtype(dtype)})
_result_orig = np.ediff1d(a)
result = _result_orig.astype(${ac})`;

    // Axis reductions
    snippets[`sum_axis0_${dtype}`] = `
a = np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})
_result_orig = np.sum(a, axis=0)
result = _result_orig.astype(${ac})`;

    snippets[`mean_axis1_${dtype}`] = `
a = np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})
_result_orig = np.mean(a, axis=1)
result = _result_orig`;

    snippets[`max_axis0_${dtype}`] = `
a = np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})
_result_orig = np.max(a, axis=0)
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Quantile/percentile', () => {
  for (const dtype of ALL_DTYPES) {
    it(`quantile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = float(np.quantile(a, 0.5))`;
        const _r = expectBothReject(
          'quantile uses subtract internally, not supported for bool',
          () => np.quantile(array(data, dtype), 0.5),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      if (isComplex(dtype)) {
        const sc = pyScalarCast(dtype);
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = ${sc}(np.quantile(a, 0.5))`;
        const _r = expectBothReject(
          'quantile is not supported for complex dtype',
          () => np.quantile(array(data, dtype), 0.5),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.quantile(array(data, dtype), 0.5);
      const py = oracle.get(`quantile_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`nanquantile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = float(np.nanquantile(a, 0.5))`;
        const _r = expectBothReject(
          'nanquantile uses subtract internally, not supported for bool',
          () => np.nanquantile(array(data, dtype), 0.5),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      if (isComplex(dtype)) {
        const sc = pyScalarCast(dtype);
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = ${sc}(np.nanquantile(a, 0.5))`;
        const _r = expectBothReject(
          'nanquantile is not supported for complex dtype',
          () => np.nanquantile(array(data, dtype), 0.5),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.nanquantile(array(data, dtype), 0.5);
      const py = oracle.get(`nanquantile_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`percentile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = float(np.percentile(a, 50))`;
        const _r = expectBothReject(
          'percentile uses subtract internally, not supported for bool',
          () => np.percentile(array(data, dtype), 50),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      if (isComplex(dtype)) {
        const sc = pyScalarCast(dtype);
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = ${sc}(np.percentile(a, 50))`;
        const _r = expectBothReject(
          'percentile is not supported for complex dtype',
          () => np.percentile(array(data, dtype), 50),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.percentile(array(data, dtype), 50);
      const py = oracle.get(`percentile_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`nanpercentile ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = float(np.nanpercentile(a, 50))`;
        const _r = expectBothReject(
          'nanpercentile uses subtract internally, not supported for bool',
          () => np.nanpercentile(array(data, dtype), 50),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      if (isComplex(dtype)) {
        const sc = pyScalarCast(dtype);
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\nresult = ${sc}(np.nanpercentile(a, 50))`;
        const _r = expectBothReject(
          'nanpercentile is not supported for complex dtype',
          () => np.nanpercentile(array(data, dtype), 50),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.nanpercentile(array(data, dtype), 50);
      const py = oracle.get(`nanpercentile_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });
  }
});

describe('DType Sweep: Cumulative & misc reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`cumsum ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const jsResult = np.cumsum(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`cumsum_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`cumprod ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 1, 0, 1] : [1, 2, 3, 4];
      const jsResult = np.cumprod(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`cumprod_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`average ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA;
      const jsResult = np.average(array(data, dtype));
      const py = oracle.get(`average_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`ediff1d ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : SMALL_DATA;
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})\n_result_orig = np.ediff1d(a)\nresult = _result_orig.astype(np.float64)`;
        const _r = expectBothReject(
          'ediff1d uses subtract internally, not supported for bool',
          () => np.ediff1d(array(data, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.ediff1d(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`ediff1d_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

describe('DType Sweep: Axis reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`sum axis=0 ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0, 1],
              [0, 1, 0],
            ]
          : SMALL_2D;
      const jsResult = np.sum(array(data, dtype), 0);
      expectMatchPre(jsResult, oracle.get(`sum_axis0_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`mean axis=1 ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0, 1],
              [0, 1, 0],
            ]
          : SMALL_2D;
      const jsResult = np.mean(array(data, dtype), 1);
      expectMatchPre(jsResult, oracle.get(`mean_axis1_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`max axis=0 ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0, 1],
              [0, 1, 0],
            ]
          : SMALL_2D;
      const jsResult = np.max(array(data, dtype), 0);
      expectMatchPre(jsResult, oracle.get(`max_axis0_${dtype}`)!);
    });
  }
});
