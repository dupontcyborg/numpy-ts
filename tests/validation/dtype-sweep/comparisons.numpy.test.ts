/**
 * DType Sweep: Comparison functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectComplexFixture,
  expectMatchPre,
  npDtype,
  pyArrayCast,
  runNumPyBatch,
} from './_helpers';

const { array } = np;

const compOps = ['greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'];

// One definition per fixture, shared by the oracle snippets and the test bodies.
// A second copy would let the two sides compare different inputs.
const compData1 = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
const compData2 = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [0, 1, 1, 0] : [4, 3, 2, 1], dtype);
const closeData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const name of compOps) {
    for (const dtype of ALL_DTYPES) {
      const ac = pyArrayCast(dtype);
      const data1 = compData1(dtype);
      const data2 = compData2(dtype);
      snippets[`${name}_${dtype}`] = `
a = np.array(${data1.py}, dtype=${npDtype(dtype)})
b = np.array(${data2.py}, dtype=${npDtype(dtype)})
_result_orig = np.${name}(a, b)
result = _result_orig.astype(${ac})`;
    }
  }

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = closeData(dtype);
    snippets[`isclose_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
b = np.array(${data.py}, dtype=${npDtype(dtype)})
_result_orig = np.isclose(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`allclose_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
result = bool(np.allclose(a, a))`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Comparisons', () => {
  for (const name of compOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = compData1(dtype);
          const data2 = compData2(dtype);
          const a = array(data1.js, dtype);
          const b = array(data2.js, dtype);
          expectComplexFixture(a, dtype, `${name} ${dtype} lhs`);
          expectComplexFixture(b, dtype, `${name} ${dtype} rhs`);
          const jsResult = (np as any)[name](a, b);
          expectMatchPre(jsResult, oracle.get(`${name}_${dtype}`)!);
        });
      }
    });
  }
});

describe('DType Sweep: Close comparisons', () => {
  for (const dtype of ALL_DTYPES) {
    it(`isclose ${dtype}`, () => {
      const data = closeData(dtype);
      const a = array(data.js, dtype);
      expectComplexFixture(a, dtype, `isclose ${dtype}`);
      const jsResult = np.isclose(a, array(data.js, dtype));
      expectMatchPre(jsResult, oracle.get(`isclose_${dtype}`)!);
    });

    it(`allclose ${dtype}`, () => {
      const data = closeData(dtype);
      const a = array(data.js, dtype);
      expectComplexFixture(a, dtype, `allclose ${dtype}`);
      const jsResult = np.allclose(a, array(data.js, dtype));
      const py = oracle.get(`allclose_${dtype}`)!;
      expect(Boolean(jsResult)).toBe(Boolean(py.value));
    });
  }
});
