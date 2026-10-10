/**
 * DType Sweep: Quantile, cumulative, and axis reduction functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothReject,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';

const { array } = np;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

// One definition per fixture, shared by the oracle snippets and the test bodies —
// two copies would let the JS and Python sides drift apart silently.
const data1d = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1, 1] : SMALL_DATA, dtype);
const data2d = (dtype: string) =>
  asDtypeData(
    dtype === 'bool'
      ? [
          [1, 0, 1],
          [0, 1, 0],
        ]
      : SMALL_2D,
    dtype,
  );
const cumprodData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 1, 0, 1] : [1, 2, 3, 4], dtype);
const ediffData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : SMALL_DATA, dtype);

// Quantile-family ops share one shape: a 1-D fixture, a quantile argument, and a
// scalar result. Both sides are generated from this table so they cannot diverge.
const quantileOps: { name: string; fn: (a: any) => any; pyCall: string }[] = [
  { name: 'quantile', fn: (a) => np.quantile(a, 0.5), pyCall: 'np.quantile(a, 0.5)' },
  { name: 'nanquantile', fn: (a) => np.nanquantile(a, 0.5), pyCall: 'np.nanquantile(a, 0.5)' },
  { name: 'percentile', fn: (a) => np.percentile(a, 50), pyCall: 'np.percentile(a, 50)' },
  { name: 'nanpercentile', fn: (a) => np.nanpercentile(a, 50), pyCall: 'np.nanpercentile(a, 50)' },
];

const quantilePy = (dtype: string, pyCall: string, cast: string) =>
  `a = np.array(${data1d(dtype).py}, dtype=${npDtype(dtype)})\nresult = ${cast}(${pyCall})`;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  // Build all Python snippets up front
  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const sc = pyScalarCast(dtype);
    const ac = pyArrayCast(dtype);
    const data = data1d(dtype);
    const d2 = data2d(dtype);

    for (const { name, pyCall } of quantileOps) {
      snippets[`${name}_${dtype}`] = quantilePy(dtype, pyCall, sc);
    }

    // Cumulative (array results with _result_orig)
    snippets[`cumsum_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
_result_orig = np.cumsum(a)
result = _result_orig.astype(${ac})`;

    snippets[`cumprod_${dtype}`] = `
a = np.array(${cumprodData(dtype).py}, dtype=${npDtype(dtype)})
_result_orig = np.cumprod(a)
result = _result_orig.astype(${ac})`;

    snippets[`average_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
result = ${sc}(np.average(a))`;

    snippets[`ediff1d_${dtype}`] = `
a = np.array(${ediffData(dtype).py}, dtype=${npDtype(dtype)})
_result_orig = np.ediff1d(a)
result = _result_orig.astype(${ac})`;

    // Axis reductions
    snippets[`sum_axis0_${dtype}`] = `
a = np.array(${d2.py}, dtype=${npDtype(dtype)})
_result_orig = np.sum(a, axis=0)
result = _result_orig.astype(${ac})`;

    snippets[`mean_axis1_${dtype}`] = `
a = np.array(${d2.py}, dtype=${npDtype(dtype)})
_result_orig = np.mean(a, axis=1)
result = _result_orig`;

    snippets[`max_axis0_${dtype}`] = `
a = np.array(${d2.py}, dtype=${npDtype(dtype)})
_result_orig = np.max(a, axis=0)
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Quantile/percentile', () => {
  for (const dtype of ALL_DTYPES) {
    for (const { name, fn, pyCall } of quantileOps) {
      it(`${name} ${dtype}`, () => {
        const a = array(data1d(dtype).js as never, dtype);
        expectComplexFixture(a, dtype, `${name} ${dtype}`);
        if (dtype === 'bool') {
          const _r = expectBothReject(
            `${name} uses subtract internally, not supported for bool`,
            () => fn(a),
            quantilePy(dtype, pyCall, 'float'),
          );
          if (_r === 'both-reject') return;
        }
        if (isComplex(dtype)) {
          const _r = expectBothReject(
            `${name} is not supported for complex dtype`,
            () => fn(a),
            quantilePy(dtype, pyCall, pyScalarCast(dtype)),
          );
          if (_r === 'both-reject') return;
        }
        const jsResult = fn(a);
        const py = oracle.get(`${name}_${dtype}`)!;
        scalarClose(jsResult, py.value);
      });
    }
  }
});

describe('DType Sweep: Cumulative & misc reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`cumsum ${dtype}`, () => {
      const a = array(data1d(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `cumsum ${dtype}`);
      const jsResult = np.cumsum(a);
      expectMatchPre(jsResult, oracle.get(`cumsum_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`cumprod ${dtype}`, () => {
      const a = array(cumprodData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `cumprod ${dtype}`);
      const jsResult = np.cumprod(a);
      expectMatchPre(jsResult, oracle.get(`cumprod_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`average ${dtype}`, () => {
      const a = array(data1d(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `average ${dtype}`);
      const jsResult = np.average(a);
      const py = oracle.get(`average_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`ediff1d ${dtype}`, () => {
      const data = ediffData(dtype);
      const a = array(data.js as never, dtype);
      expectComplexFixture(a, dtype, `ediff1d ${dtype}`);
      if (dtype === 'bool') {
        const pyCode = `a = np.array(${data.py}, dtype=${npDtype(dtype)})\n_result_orig = np.ediff1d(a)\nresult = _result_orig.astype(np.float64)`;
        const _r = expectBothReject(
          'ediff1d uses subtract internally, not supported for bool',
          () => np.ediff1d(a),
          pyCode,
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.ediff1d(a);
      expectMatchPre(jsResult, oracle.get(`ediff1d_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

describe('DType Sweep: Axis reductions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`sum axis=0 ${dtype}`, () => {
      const a = array(data2d(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `sum axis=0 ${dtype}`);
      const jsResult = np.sum(a, 0);
      expectMatchPre(jsResult, oracle.get(`sum_axis0_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`mean axis=1 ${dtype}`, () => {
      const a = array(data2d(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `mean axis=1 ${dtype}`);
      const jsResult = np.mean(a, 1);
      expectMatchPre(jsResult, oracle.get(`mean_axis1_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`max axis=0 ${dtype}`, () => {
      const a = array(data2d(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `max axis=0 ${dtype}`);
      const jsResult = np.max(a, 0);
      expectMatchPre(jsResult, oracle.get(`max_axis0_${dtype}`)!);
    });
  }
});
