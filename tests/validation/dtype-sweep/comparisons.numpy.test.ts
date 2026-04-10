/**
 * DType Sweep: Comparison functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  pyArrayCast,
  runNumPyBatch,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

const compOps = ['greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'];

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const name of compOps) {
    for (const dtype of ALL_DTYPES) {
      const ac = pyArrayCast(dtype);
      const data1 =
        dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 2, 3, 4] : [1, 2, 3, 4];
      const data2 =
        dtype === 'bool' ? [0, 1, 1, 0] : isComplex(dtype) ? [4, 3, 2, 1] : [4, 3, 2, 1];
      snippets[`${name}_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.${name}(a, b)
result = _result_orig.astype(${ac})`;
    }
  }

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data1 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
    const data2 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
    snippets[`isclose_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.isclose(a, b)
result = _result_orig.astype(${ac})`;

    const data = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
    snippets[`allclose_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = bool(np.allclose(a, a))`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Comparisons', () => {
  for (const name of compOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 =
            dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 2, 3, 4] : [1, 2, 3, 4];
          const data2 =
            dtype === 'bool' ? [0, 1, 1, 0] : isComplex(dtype) ? [4, 3, 2, 1] : [4, 3, 2, 1];
          const a = array(data1, dtype);
          const b = array(data2, dtype);
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
      const data1 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const data2 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const jsResult = np.isclose(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, oracle.get(`isclose_${dtype}`)!);
    });

    it(`allclose ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const jsResult = np.allclose(array(data, dtype), array(data, dtype));
      const py = oracle.get(`allclose_${dtype}`)!;
      expect(Boolean(jsResult)).toBe(Boolean(py.value));
    });
  }
});
