/**
 * DType Sweep: Sorting & searching functions.
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
  pyArrayCast,
  scalarClose,
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
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : [5, 2, 8, 1, 9, 3];

    snippets[`sort_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.sort(a)
result = _result_orig.astype(${ac})`;

    snippets[`argsort_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.argsort(a)
result = _result_orig`;

    const sorted = dtype === 'bool' ? [0, 0, 1, 1, 1] : [1, 3, 5, 7, 9];
    const vals = dtype === 'bool' ? [0, 1] : [2, 4, 6];
    snippets[`searchsorted_${dtype}`] = `
a = np.array(${JSON.stringify(sorted)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(vals)}, dtype=${npDtype(dtype)})
_result_orig = np.searchsorted(a, v)
result = _result_orig`;

    snippets[`partition_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
r = np.partition(a, 2)
result = np.array([r[2]]).astype(${ac})`;

    snippets[`argpartition_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
idx = np.argpartition(a, 2)
result = np.array([a[idx[2]]]).astype(${ac})`;

    const sortComplexData = isComplex(dtype) ? [3, 1, 2] : dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
    snippets[`sort_complex_${dtype}`] = `
a = np.array(${JSON.stringify(sortComplexData)}, dtype=${npDtype(dtype)})
_result_orig = np.sort_complex(a)
result = _result_orig.astype(np.complex128)`;

    const keys1 = dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
    const keys2 = dtype === 'bool' ? [0, 1, 0] : [1, 3, 2];
    snippets[`lexsort_${dtype}`] = `
k1 = np.array(${JSON.stringify(keys1)}, dtype=${npDtype(dtype)})
k2 = np.array(${JSON.stringify(keys2)}, dtype=${npDtype(dtype)})
_result_orig = np.lexsort((k1, k2))
result = _result_orig`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Sorting', () => {
  for (const dtype of ALL_DTYPES) {
    const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : [5, 2, 8, 1, 9, 3];

    it(`sort ${dtype}`, () => {
      const jsResult = np.sort(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`sort_${dtype}`)!);
    });

    it(`argsort ${dtype}`, () => {
      const jsResult = np.argsort(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`argsort_${dtype}`)!, { indexResult: true });
    });

    it(`searchsorted ${dtype}`, () => {
      const sorted = dtype === 'bool' ? [0, 0, 1, 1, 1] : [1, 3, 5, 7, 9];
      const vals = dtype === 'bool' ? [0, 1] : [2, 4, 6];
      const jsResult = np.searchsorted(array(sorted, dtype), array(vals, dtype));
      expectMatchPre(jsResult, oracle.get(`searchsorted_${dtype}`)!, { indexResult: true });
    });

    it(`partition ${dtype}`, () => {
      const jsResult = np.partition(array(data, dtype), 2);
      const py = oracle.get(`partition_${dtype}`)!;
      const jsKth = jsResult.toArray()[2];
      scalarClose(jsKth, py.value[0]);
    });

    it(`argpartition ${dtype}`, () => {
      const a = array(data, dtype);
      const jsResult = np.argpartition(a, 2);
      const py = oracle.get(`argpartition_${dtype}`)!;
      const jsIdx = Number(jsResult.toArray()[2]);
      const jsKthVal = a.toArray()[jsIdx];
      scalarClose(jsKthVal, py.value[0]);
    });

    it(`sort_complex ${dtype}`, () => {
      const d = isComplex(dtype) ? [3, 1, 2] : dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
      const jsResult = np.sort_complex(array(d, dtype));
      expectMatchPre(jsResult, oracle.get(`sort_complex_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`lexsort ${dtype}`, () => {
      const keys1 = dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
      const keys2 = dtype === 'bool' ? [0, 1, 0] : [1, 3, 2];
      const jsResult = np.lexsort([array(keys1, dtype), array(keys2, dtype)]);
      expectMatchPre(jsResult, oracle.get(`lexsort_${dtype}`)!, { indexResult: true });
    });
  }
});
