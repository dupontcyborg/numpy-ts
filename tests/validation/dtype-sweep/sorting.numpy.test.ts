/**
 * DType Sweep: Sorting & searching functions.
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
  expectComplexFixture,
  expectMatchPre,
  npDtype,
  pyArrayCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippets and the test bodies —
// two copies would let the JS and Python sides drift apart silently.
const unsortedData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : [5, 2, 8, 1, 9, 3], dtype);
const sortedData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [0, 0, 1, 1, 1] : [1, 3, 5, 7, 9], dtype);
const searchValues = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1] : [2, 4, 6], dtype);
const smallData = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [3, 1, 2], dtype);
const lexKeys2 = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1, 0] : [1, 3, 2], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = unsortedData(dtype);

    snippets[`sort_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
_result_orig = np.sort(a)
result = _result_orig.astype(${ac})`;

    snippets[`argsort_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
_result_orig = np.argsort(a)
result = _result_orig`;

    const sorted = sortedData(dtype);
    const vals = searchValues(dtype);
    snippets[`searchsorted_${dtype}`] = `
a = np.array(${sorted.py}, dtype=${npDtype(dtype)})
v = np.array(${vals.py}, dtype=${npDtype(dtype)})
_result_orig = np.searchsorted(a, v)
result = _result_orig`;

    snippets[`partition_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
r = np.partition(a, 2)
result = np.array([r[2]]).astype(${ac})`;

    snippets[`argpartition_${dtype}`] = `
a = np.array(${data.py}, dtype=${npDtype(dtype)})
idx = np.argpartition(a, 2)
result = np.array([a[idx[2]]]).astype(${ac})`;

    const sortComplexData = smallData(dtype);
    snippets[`sort_complex_${dtype}`] = `
a = np.array(${sortComplexData.py}, dtype=${npDtype(dtype)})
_result_orig = np.sort_complex(a)
result = _result_orig.astype(np.complex128)`;

    const keys1 = smallData(dtype);
    const keys2 = lexKeys2(dtype);
    snippets[`lexsort_${dtype}`] = `
k1 = np.array(${keys1.py}, dtype=${npDtype(dtype)})
k2 = np.array(${keys2.py}, dtype=${npDtype(dtype)})
_result_orig = np.lexsort((k1, k2))
result = _result_orig`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Sorting', () => {
  for (const dtype of ALL_DTYPES) {
    it(`sort ${dtype}`, () => {
      const a = array(unsortedData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `sort ${dtype}`);
      const jsResult = np.sort(a);
      expectMatchPre(jsResult, oracle.get(`sort_${dtype}`)!);
    });

    it(`argsort ${dtype}`, () => {
      const a = array(unsortedData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `argsort ${dtype}`);
      const jsResult = np.argsort(a);
      expectMatchPre(jsResult, oracle.get(`argsort_${dtype}`)!, { indexResult: true });
    });

    it(`searchsorted ${dtype}`, () => {
      const a = array(sortedData(dtype).js as never, dtype);
      const v = array(searchValues(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `searchsorted ${dtype}`);
      const jsResult = np.searchsorted(a, v);
      expectMatchPre(jsResult, oracle.get(`searchsorted_${dtype}`)!, { indexResult: true });
    });

    it(`partition ${dtype}`, () => {
      const a = array(unsortedData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `partition ${dtype}`);
      const jsResult = np.partition(a, 2);
      const py = oracle.get(`partition_${dtype}`)!;
      const jsKth = jsResult.toArray()[2];
      scalarClose(jsKth, py.value[0]);
    });

    it(`argpartition ${dtype}`, () => {
      const a = array(unsortedData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `argpartition ${dtype}`);
      const jsResult = np.argpartition(a, 2);
      const py = oracle.get(`argpartition_${dtype}`)!;
      const jsIdx = Number(jsResult.toArray()[2]);
      const jsKthVal = a.toArray()[jsIdx];
      scalarClose(jsKthVal, py.value[0]);
    });

    it(`sort_complex ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `sort_complex ${dtype}`);
      const jsResult = np.sort_complex(a);
      expectMatchPre(jsResult, oracle.get(`sort_complex_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`lexsort ${dtype}`, () => {
      const k1 = array(smallData(dtype).js as never, dtype);
      const k2 = array(lexKeys2(dtype).js as never, dtype);
      expectComplexFixture(k1, dtype, `lexsort ${dtype}`);
      const jsResult = np.lexsort([k1, k2]);
      expectMatchPre(jsResult, oracle.get(`lexsort_${dtype}`)!, { indexResult: true });
    });
  }
});
