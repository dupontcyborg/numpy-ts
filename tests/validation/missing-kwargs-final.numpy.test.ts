/**
 * NumPy validation tests for the final batch of missing kwargs:
 *  - concatenate(arrays, axis=null)
 *  - diff(a, n, axis, prepend, append)
 *  - apply_along_axis(func1d, axis, arr, ...args)
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { apply_along_axis, arange, array, concat, concatenate, diff } from '../../src';
import type { NDArray } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: final-batch kwargs', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('concatenate axis=null', () => {
    it('flattens and concatenates 2-D inputs', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6, 7],
        [8, 9, 10],
      ]);
      const result = concatenate([a, b], null) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6, 7], [8, 9, 10]])
result = np.concatenate([a, b], axis=None)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('concat alias also supports axis=null', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = concat([a, array([5, 6])], null) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 2], [3, 4]])
result = np.concatenate([a, np.array([5, 6])], axis=None)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });

  describe('diff prepend/append', () => {
    it('scalar prepend and append on 1-D', () => {
      const a = array([1, 2, 4, 7, 11]);
      const result = diff(a, 1, undefined, 0, 100) as NDArray;
      const np = runNumPy(`
a = np.array([1, 2, 4, 7, 11])
result = np.diff(a, prepend=0, append=100)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('array prepend on 2-D along default axis', () => {
      const a = array([
        [1, 3, 6, 10],
        [2, 4, 8, 16],
      ]);
      const result = diff(a, 1, undefined, [[0], [0]]) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 3, 6, 10], [2, 4, 8, 16]])
result = np.diff(a, prepend=[[0], [0]])
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('scalar prepend on 2-D along axis=0', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = diff(a, 1, 0, 0) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
result = np.diff(a, axis=0, prepend=0)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });

  describe('apply_along_axis args forwarding', () => {
    it('passes scalar args through to the callback', () => {
      const a = arange(12).reshape(3, 4);
      // Custom: sum elements > threshold
      const sumGt = (row: NDArray, threshold: number) =>
        ((row.toArray() as number[]).filter((x) => x > threshold).reduce((s, x) => s + x, 0) as number);
      const result = apply_along_axis(sumGt as never, 1, a, 5) as NDArray;
      const np = runNumPy(`
a = np.arange(12).reshape(3, 4)
def sum_gt(row, threshold):
    return float(np.sum(row[row > threshold]))
result = np.apply_along_axis(sum_gt, 1, a, 5)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });
});
