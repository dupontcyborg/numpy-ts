/**
 * DType Sweep: Logical operations + isnan/isinf/isfinite, validated against NumPy.
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
  expectBothReject,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : isComplex(dtype) ? [1, 0, 1, 0] : [1, 1, 0, 0];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 1, 0, 0] : [1, 0, 1, 0];
    const checkData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];

    snippets[`logical_and_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.logical_and(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_or_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.logical_or(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_not_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
_result_orig = np.logical_not(a)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_xor_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.logical_xor(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`isnan_${dtype}`] = `
_result_orig = np.isnan(np.array(${JSON.stringify(checkData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isinf_${dtype}`] = `
_result_orig = np.isinf(np.array(${JSON.stringify(checkData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isfinite_${dtype}`] = `
_result_orig = np.isfinite(np.array(${JSON.stringify(checkData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isneginf_${dtype}`] = `
_result_orig = np.isneginf(np.array(${JSON.stringify(checkData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isposinf_${dtype}`] = `
_result_orig = np.isposinf(np.array(${JSON.stringify(checkData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Logical', () => {
  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : isComplex(dtype) ? [1, 0, 1, 0] : [1, 1, 0, 0];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 1, 0, 0] : [1, 0, 1, 0];

    it(`logical_and ${dtype}`, () => {
      const jsResult = np.logical_and(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_and_${dtype}`)!);
    });

    it(`logical_or ${dtype}`, () => {
      const jsResult = np.logical_or(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_or_${dtype}`)!);
    });

    it(`logical_not ${dtype}`, () => {
      const jsResult = np.logical_not(array(data1, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_not_${dtype}`)!);
    });

    it(`logical_xor ${dtype}`, () => {
      const jsResult = np.logical_xor(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_xor_${dtype}`)!);
    });

    it(`isnan ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isnan(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`isnan_${dtype}`)!);
    });

    it(`isinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isinf(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`isinf_${dtype}`)!);
    });

    it(`isfinite ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isfinite(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`isfinite_${dtype}`)!);
    });

    it(`isneginf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const pyCode = `
_result_orig = np.isneginf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'isneginf is not defined for complex numbers',
            () => np.isneginf(array(data, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.isneginf(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`isneginf_${dtype}`)!);
    });

    it(`isposinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const pyCode = `
_result_orig = np.isposinf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'isposinf is not defined for complex numbers',
            () => np.isposinf(array(data, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.isposinf(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`isposinf_${dtype}`)!);
    });
  }
});
