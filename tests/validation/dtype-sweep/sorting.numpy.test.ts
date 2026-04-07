/**
 * DType Sweep: Sorting & searching functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Sorting', () => {
  for (const dtype of ALL_DTYPES) {
    const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : [5, 2, 8, 1, 9, 3];

    it(`sort ${dtype}`, () => {
      const jsResult = np.sort(array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.sort(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`argsort ${dtype}`, () => {
      const jsResult = np.argsort(array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.argsort(a)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`searchsorted ${dtype}`, () => {
      const sorted = dtype === 'bool' ? [0, 0, 1, 1, 1] : [1, 3, 5, 7, 9];
      const vals = dtype === 'bool' ? [0, 1] : [2, 4, 6];
      const jsResult = np.searchsorted(array(sorted, dtype), array(vals, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(sorted)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(vals)}, dtype=${npDtype(dtype)})
result = np.searchsorted(a, v)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`partition ${dtype}`, () => {
      const jsResult = np.partition(array(data, dtype), 2);
      // Partition only guarantees element at kth is correct
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
r = np.partition(a, 2)
result = np.array([r[2]]).astype(np.float64)
      `);
      const jsKth = jsResult.toArray()[2];
      expect(Number(jsKth)).toBeCloseTo(Number(pyResult.value[0]), 4);
    });

    it(`argpartition ${dtype}`, () => {
      const a = array(data, dtype);
      const jsResult = np.argpartition(a, 2);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
idx = np.argpartition(a, 2)
result = np.array([a[idx[2]]]).astype(np.float64)
      `);
      // The element at kth position should match
      const jsIdx = Number(jsResult.toArray()[2]);
      const jsKthVal = Number(a.toArray()[jsIdx]);
      expect(jsKthVal).toBeCloseTo(Number(pyResult.value[0]), 4);
    });

    it(`sort_complex ${dtype}`, () => {
      const d = isComplex(dtype) ? [3, 1, 2] : dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
      const jsResult = np.sort_complex(array(d, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(d)}, dtype=${npDtype(dtype)})
result = np.sort_complex(a).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`lexsort ${dtype}`, () => {
      const keys1 = dtype === 'bool' ? [1, 0, 1] : [3, 1, 2];
      const keys2 = dtype === 'bool' ? [0, 1, 0] : [1, 3, 2];
      const jsResult = np.lexsort([array(keys1, dtype), array(keys2, dtype)]);
      const pyResult = runNumPy(`
k1 = np.array(${JSON.stringify(keys1)}, dtype=${npDtype(dtype)})
k2 = np.array(${JSON.stringify(keys2)}, dtype=${npDtype(dtype)})
result = np.lexsort((k1, k2))
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
