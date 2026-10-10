/**
 * DType Sweep: Logical operations + isnan/isinf/isfinite, validated against NumPy.
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
  runNumPyBatch,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippet and the test body:
// the two sides must build the same input, and a separate declaration on each
// side drifts silently. Complex dtypes get a nonzero imaginary part, so truthiness
// here depends on the half of the value only the complex path reads.
const operandA = (dtype: string) =>
  asDtypeData(
    dtype === 'bool' ? [1, 1, 0, 0] : isComplex(dtype) ? [1, 0, 1, 0] : [1, 1, 0, 0],
    dtype,
  );
const operandB = (dtype: string) =>
  asDtypeData(
    dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 1, 0, 0] : [1, 0, 1, 0],
    dtype,
  );
const checkOperand = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data1 = operandA(dtype);
    const data2 = operandB(dtype);
    const checkData = checkOperand(dtype);

    snippets[`logical_and_${dtype}`] = `
a = np.array(${data1.py}, dtype=${npDtype(dtype)})
b = np.array(${data2.py}, dtype=${npDtype(dtype)})
_result_orig = np.logical_and(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_or_${dtype}`] = `
a = np.array(${data1.py}, dtype=${npDtype(dtype)})
b = np.array(${data2.py}, dtype=${npDtype(dtype)})
_result_orig = np.logical_or(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_not_${dtype}`] = `
a = np.array(${data1.py}, dtype=${npDtype(dtype)})
_result_orig = np.logical_not(a)
result = _result_orig.astype(np.float64)`;

    snippets[`logical_xor_${dtype}`] = `
a = np.array(${data1.py}, dtype=${npDtype(dtype)})
b = np.array(${data2.py}, dtype=${npDtype(dtype)})
_result_orig = np.logical_xor(a, b)
result = _result_orig.astype(np.float64)`;

    snippets[`isnan_${dtype}`] = `
_result_orig = np.isnan(np.array(${checkData.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isinf_${dtype}`] = `
_result_orig = np.isinf(np.array(${checkData.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isfinite_${dtype}`] = `
_result_orig = np.isfinite(np.array(${checkData.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isneginf_${dtype}`] = `
_result_orig = np.isneginf(np.array(${checkData.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`isposinf_${dtype}`] = `
_result_orig = np.isposinf(np.array(${checkData.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Logical', () => {
  for (const dtype of ALL_DTYPES) {
    it(`logical_and ${dtype}`, () => {
      const d1 = operandA(dtype);
      const d2 = operandB(dtype);
      const a = array(d1.js, dtype);
      expectComplexFixture(a, dtype, `logical_and ${dtype}`);
      const jsResult = np.logical_and(a, array(d2.js, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_and_${dtype}`)!);
    });

    it(`logical_or ${dtype}`, () => {
      const d1 = operandA(dtype);
      const d2 = operandB(dtype);
      const a = array(d1.js, dtype);
      expectComplexFixture(a, dtype, `logical_or ${dtype}`);
      const jsResult = np.logical_or(a, array(d2.js, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_or_${dtype}`)!);
    });

    it(`logical_not ${dtype}`, () => {
      const d1 = operandA(dtype);
      const a = array(d1.js, dtype);
      expectComplexFixture(a, dtype, `logical_not ${dtype}`);
      const jsResult = np.logical_not(a);
      expectMatchPre(jsResult, oracle.get(`logical_not_${dtype}`)!);
    });

    it(`logical_xor ${dtype}`, () => {
      const d1 = operandA(dtype);
      const d2 = operandB(dtype);
      const a = array(d1.js, dtype);
      expectComplexFixture(a, dtype, `logical_xor ${dtype}`);
      const jsResult = np.logical_xor(a, array(d2.js, dtype));
      expectMatchPre(jsResult, oracle.get(`logical_xor_${dtype}`)!);
    });

    it(`isnan ${dtype}`, () => {
      const d = checkOperand(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `isnan ${dtype}`);
      const jsResult = np.isnan(a);
      expectMatchPre(jsResult, oracle.get(`isnan_${dtype}`)!);
    });

    it(`isinf ${dtype}`, () => {
      const d = checkOperand(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `isinf ${dtype}`);
      const jsResult = np.isinf(a);
      expectMatchPre(jsResult, oracle.get(`isinf_${dtype}`)!);
    });

    it(`isfinite ${dtype}`, () => {
      const d = checkOperand(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `isfinite ${dtype}`);
      const jsResult = np.isfinite(a);
      expectMatchPre(jsResult, oracle.get(`isfinite_${dtype}`)!);
    });

    it(`isneginf ${dtype}`, () => {
      const d = checkOperand(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `isneginf ${dtype}`);
      const pyCode = `
_result_orig = np.isneginf(np.array(${d.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'isneginf is not defined for complex numbers',
          () => np.isneginf(a),
          pyCode,
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.isneginf(a);
      expectMatchPre(jsResult, oracle.get(`isneginf_${dtype}`)!);
    });

    it(`isposinf ${dtype}`, () => {
      const d = checkOperand(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `isposinf ${dtype}`);
      const pyCode = `
_result_orig = np.isposinf(np.array(${d.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'isposinf is not defined for complex numbers',
          () => np.isposinf(a),
          pyCode,
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.isposinf(a);
      expectMatchPre(jsResult, oracle.get(`isposinf_${dtype}`)!);
    });
  }
});
