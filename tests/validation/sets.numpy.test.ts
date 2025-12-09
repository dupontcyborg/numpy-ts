/**
 * Python NumPy validation tests for set operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  unique,
  in1d,
  intersect1d,
  isin,
  setdiff1d,
  setxor1d,
  union1d,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Set Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('unique()', () => {
    it('validates unique() on 1D array', () => {
      const arr = array([1, 2, 2, 3, 3, 3, 4]);
      const result = unique(arr);

      const npResult = runNumPy(`
arr = np.array([1, 2, 2, 3, 3, 3, 4])
result = np.unique(arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates unique() on 2D array (flattened)', () => {
      const arr = array([
        [1, 2, 1],
        [2, 3, 3],
      ]);
      const result = unique(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 1], [2, 3, 3]])
result = np.unique(arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates unique() with return_counts', () => {
      const arr = array([1, 2, 2, 3, 3, 3]);
      const result = unique(arr, false, false, true);

      const npResult = runNumPy(`
arr = np.array([1, 2, 2, 3, 3, 3])
values, counts = np.unique(arr, return_counts=True)
result = {"values": values.tolist(), "counts": counts.tolist()}
`);

      expect(arraysClose((result as any).values.toArray(), npResult.value.values)).toBe(true);
      expect(arraysClose((result as any).counts.toArray(), npResult.value.counts)).toBe(true);
    });

    it('validates unique() with return_inverse', () => {
      const arr = array([1, 2, 1, 3, 2]);
      const result = unique(arr, false, true);

      const npResult = runNumPy(`
arr = np.array([1, 2, 1, 3, 2])
values, inverse = np.unique(arr, return_inverse=True)
result = {"values": values.tolist(), "inverse": inverse.tolist()}
`);

      expect(arraysClose((result as any).values.toArray(), npResult.value.values)).toBe(true);
      expect(arraysClose((result as any).inverse.toArray(), npResult.value.inverse)).toBe(true);
    });
  });

  describe('in1d()', () => {
    it('validates in1d() basic membership', () => {
      const ar1 = array([1, 2, 3, 4, 5]);
      const ar2 = array([2, 4]);
      const result = in1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3, 4, 5])
ar2 = np.array([2, 4])
result = np.in1d(ar1, ar2).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates in1d() with no matches', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([4, 5, 6]);
      const result = in1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3])
ar2 = np.array([4, 5, 6])
result = np.in1d(ar1, ar2).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('intersect1d()', () => {
    it('validates intersect1d() basic intersection', () => {
      const ar1 = array([1, 2, 3, 4]);
      const ar2 = array([3, 4, 5, 6]);
      const result = intersect1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3, 4])
ar2 = np.array([3, 4, 5, 6])
result = np.intersect1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates intersect1d() with no intersection', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([3, 4]);
      const result = intersect1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2])
ar2 = np.array([3, 4])
result = np.intersect1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates intersect1d() with duplicates', () => {
      const ar1 = array([1, 1, 2, 2, 3]);
      const ar2 = array([2, 2, 3, 3, 4]);
      const result = intersect1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 1, 2, 2, 3])
ar2 = np.array([2, 2, 3, 3, 4])
result = np.intersect1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('isin()', () => {
    it('validates isin() basic membership', () => {
      const element = array([1, 2, 3, 4, 5]);
      const testElements = array([2, 4]);
      const result = isin(element, testElements);

      const npResult = runNumPy(`
element = np.array([1, 2, 3, 4, 5])
test_elements = np.array([2, 4])
result = np.isin(element, test_elements).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isin() preserves shape', () => {
      const element = array([
        [1, 2],
        [3, 4],
      ]);
      const testElements = array([2, 3]);
      const result = isin(element, testElements);

      const npResult = runNumPy(`
element = np.array([[1, 2], [3, 4]])
test_elements = np.array([2, 3])
result = np.isin(element, test_elements).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('setdiff1d()', () => {
    it('validates setdiff1d() basic difference', () => {
      const ar1 = array([1, 2, 3, 4, 5]);
      const ar2 = array([3, 4]);
      const result = setdiff1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3, 4, 5])
ar2 = np.array([3, 4])
result = np.setdiff1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates setdiff1d() with empty difference', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([1, 2, 3]);
      const result = setdiff1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2])
ar2 = np.array([1, 2, 3])
result = np.setdiff1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates setdiff1d() with duplicates', () => {
      const ar1 = array([1, 1, 2, 2, 3]);
      const ar2 = array([2]);
      const result = setdiff1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 1, 2, 2, 3])
ar2 = np.array([2])
result = np.setdiff1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('setxor1d()', () => {
    it('validates setxor1d() symmetric difference', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([2, 3, 4]);
      const result = setxor1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3])
ar2 = np.array([2, 3, 4])
result = np.setxor1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates setxor1d() with identical arrays', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([1, 2, 3]);
      const result = setxor1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3])
ar2 = np.array([1, 2, 3])
result = np.setxor1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates setxor1d() with disjoint arrays', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([3, 4]);
      const result = setxor1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2])
ar2 = np.array([3, 4])
result = np.setxor1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('union1d()', () => {
    it('validates union1d() basic union', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([3, 4, 5]);
      const result = union1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3])
ar2 = np.array([3, 4, 5])
result = np.union1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates union1d() with identical arrays', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([1, 2, 3]);
      const result = union1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 2, 3])
ar2 = np.array([1, 2, 3])
result = np.union1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates union1d() with duplicates', () => {
      const ar1 = array([1, 1, 2, 2]);
      const ar2 = array([2, 2, 3, 3]);
      const result = union1d(ar1, ar2);

      const npResult = runNumPy(`
ar1 = np.array([1, 1, 2, 2])
ar2 = np.array([2, 2, 3, 3])
result = np.union1d(ar1, ar2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
