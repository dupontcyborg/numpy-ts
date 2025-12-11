/**
 * Python NumPy validation tests for sorting and searching operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  sort,
  argsort,
  lexsort,
  partition,
  argpartition,
  sort_complex,
  nonzero,
  argwhere,
  flatnonzero,
  where,
  searchsorted,
  extract,
  count_nonzero,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Sorting Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('sort()', () => {
    it('validates sort() on 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      const result = sort(arr);

      const npResult = runNumPy(`
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = np.sort(arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sort() on 2D array along axis=-1', () => {
      const arr = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = sort(arr, -1);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 2], [6, 4, 5]])
result = np.sort(arr, axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sort() on 2D array along axis=0', () => {
      const arr = array([
        [3, 1, 2],
        [0, 4, 1],
      ]);
      const result = sort(arr, 0);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 2], [0, 4, 1]])
result = np.sort(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('argsort()', () => {
    it('validates argsort() on 1D array', () => {
      const arr = array([3, 1, 4, 1, 5]);
      const result = argsort(arr);

      const npResult = runNumPy(`
arr = np.array([3, 1, 4, 1, 5])
result = np.argsort(arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argsort() on 2D array along axis=-1', () => {
      const arr = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = argsort(arr, -1);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 2], [6, 4, 5]])
result = np.argsort(arr, axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('lexsort()', () => {
    it('validates lexsort() with two keys', () => {
      const key1 = array([1, 2, 1, 2]);
      const key2 = array([3, 1, 2, 4]);
      const result = lexsort([key1, key2]);

      const npResult = runNumPy(`
key1 = np.array([1, 2, 1, 2])
key2 = np.array([3, 1, 2, 4])
result = np.lexsort((key1, key2))
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('partition()', () => {
    it('validates partition() kth element is in correct position', () => {
      const arr = array([3, 4, 2, 1]);
      const result = partition(arr, 2);

      const npResult = runNumPy(`
arr = np.array([3, 4, 2, 1])
result = np.partition(arr, 2)
`);

      // The element at index 2 should be the same as NumPy's
      const resultArr = (result as any).toArray() as number[];
      const npResultArr = npResult.value as number[];
      expect(resultArr[2]).toBe(npResultArr[2]);
    });

    it('validates partition() on 2D array along axis=-1', () => {
      const arr = array([
        [3, 1, 4, 2],
        [8, 5, 7, 6],
      ]);
      const result = partition(arr, 2, -1);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 4, 2], [8, 5, 7, 6]])
result = np.partition(arr, 2, axis=-1)
`);

      // Check that kth element in each row is in correct position
      const resultArr = (result as any).toArray() as number[][];
      const npResultArr = npResult.value as number[][];
      expect(resultArr[0]![2]).toBe(npResultArr[0]![2]);
      expect(resultArr[1]![2]).toBe(npResultArr[1]![2]);
    });

    it('validates partition() on 2D array along axis=0', () => {
      const arr = array([
        [3, 1, 4],
        [2, 5, 0],
        [8, 7, 6],
      ]);
      const result = partition(arr, 1, 0);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 4], [2, 5, 0], [8, 7, 6]])
result = np.partition(arr, 1, axis=0)
`);

      // Check that kth element in each column is in correct position
      const resultArr = (result as any).toArray() as number[][];
      const npResultArr = npResult.value as number[][];
      expect(resultArr[1]![0]).toBe(npResultArr[1]![0]);
      expect(resultArr[1]![1]).toBe(npResultArr[1]![1]);
      expect(resultArr[1]![2]).toBe(npResultArr[1]![2]);
    });

    it('validates partition() with kth=0 (first element)', () => {
      const arr = array([5, 2, 8, 1, 9]);
      const result = partition(arr, 0);

      const npResult = runNumPy(`
arr = np.array([5, 2, 8, 1, 9])
result = np.partition(arr, 0)
`);

      // Element at index 0 should be the minimum
      const resultArr = (result as any).toArray() as number[];
      const npResultArr = npResult.value as number[];
      expect(resultArr[0]).toBe(npResultArr[0]);
      expect(resultArr[0]).toBe(1); // Should be minimum
    });

    it('validates partition() with kth=-1 (last element)', () => {
      const arr = array([5, 2, 8, 1, 9]);
      const result = partition(arr, -1);

      const npResult = runNumPy(`
arr = np.array([5, 2, 8, 1, 9])
result = np.partition(arr, -1)
`);

      // Element at last index should be the maximum
      const resultArr = (result as any).toArray() as number[];
      const npResultArr = npResult.value as number[];
      expect(resultArr[4]).toBe(npResultArr[4]);
      expect(resultArr[4]).toBe(9); // Should be maximum
    });
  });

  describe('argpartition()', () => {
    it('validates argpartition() kth element', () => {
      const arr = array([3, 4, 2, 1]);
      const result = argpartition(arr, 2);

      const npResult = runNumPy(`
arr = np.array([3, 4, 2, 1])
result = np.argpartition(arr, 2)
`);

      // The original value at the kth index should be the same
      const resultIdx = (result as any).get([2]) as number;
      const npResultIdx = npResult.value[2];
      const originalVal = arr.get([resultIdx]);
      const npOriginalVal = [3, 4, 2, 1][npResultIdx];
      expect(originalVal).toBe(npOriginalVal);
    });

    it('validates argpartition() on 2D array along axis=-1', () => {
      const arr = array([
        [3, 1, 4, 2],
        [8, 5, 7, 6],
      ]);
      const result = argpartition(arr, 2, -1);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 4, 2], [8, 5, 7, 6]])
result = np.argpartition(arr, 2, axis=-1)
`);

      // Check that indices at kth position point to correct values
      const resultArr = (result as any).toArray() as number[][];
      const npResultArr = npResult.value as number[][];
      const originalArr = arr.toArray() as number[][];

      // Value at kth index should be the same
      const tsIdx0 = resultArr[0]![2]!;
      const npIdx0 = npResultArr[0]![2]!;
      expect(originalArr[0]![tsIdx0]).toBe(originalArr[0]![npIdx0]);

      const tsIdx1 = resultArr[1]![2]!;
      const npIdx1 = npResultArr[1]![2]!;
      expect(originalArr[1]![tsIdx1]).toBe(originalArr[1]![npIdx1]);
    });

    it('validates argpartition() on 2D array along axis=0', () => {
      const arr = array([
        [3, 1, 4],
        [2, 5, 0],
        [8, 7, 6],
      ]);
      const result = argpartition(arr, 1, 0);

      const npResult = runNumPy(`
arr = np.array([[3, 1, 4], [2, 5, 0], [8, 7, 6]])
result = np.argpartition(arr, 1, axis=0)
`);

      // Check that indices at kth position point to correct values
      const resultArr = (result as any).toArray() as number[][];
      const npResultArr = npResult.value as number[][];
      const originalArr = arr.toArray() as number[][];

      // Check each column's kth element
      for (let col = 0; col < 3; col++) {
        const tsIdx = resultArr[1]![col]!;
        const npIdx = npResultArr[1]![col]!;
        expect(originalArr[tsIdx]![col]).toBe(originalArr[npIdx]![col]);
      }
    });

    it('validates argpartition() with kth=0 (first element)', () => {
      const arr = array([5, 2, 8, 1, 9]);
      const result = argpartition(arr, 0);

      const npResult = runNumPy(`
arr = np.array([5, 2, 8, 1, 9])
result = np.argpartition(arr, 0)
`);

      // Index at position 0 should point to minimum value
      const resultArr = (result as any).toArray() as number[];
      const npResultArr = npResult.value as number[];
      const originalArr = arr.toArray() as number[];

      const tsIdx = resultArr[0]!;
      const npIdx = npResultArr[0]!;
      expect(originalArr[tsIdx]).toBe(originalArr[npIdx]);
      expect(originalArr[tsIdx]).toBe(1); // Should be minimum
    });

    it('validates argpartition() with kth=-1 (last element)', () => {
      const arr = array([5, 2, 8, 1, 9]);
      const result = argpartition(arr, -1);

      const npResult = runNumPy(`
arr = np.array([5, 2, 8, 1, 9])
result = np.argpartition(arr, -1)
`);

      // Index at last position should point to maximum value
      const resultArr = (result as any).toArray() as number[];
      const npResultArr = npResult.value as number[];
      const originalArr = arr.toArray() as number[];

      const tsIdx = resultArr[4]!;
      const npIdx = npResultArr[4]!;
      expect(originalArr[tsIdx]).toBe(originalArr[npIdx]);
      expect(originalArr[tsIdx]).toBe(9); // Should be maximum
    });
  });

  describe('sort_complex()', () => {
    it('validates sort_complex() on real array', () => {
      const arr = array([3, 1, 4, 2]);
      const result = sort_complex(arr);

      const npResult = runNumPy(`
arr = np.array([3, 1, 4, 2])
result = np.sort_complex(arr).real
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});

describe('NumPy Validation: Searching Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('nonzero()', () => {
    it('validates nonzero() on 1D array', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = [x.tolist() for x in np.nonzero(arr)]
`);

      expect(result[0]!.toArray()).toEqual(npResult.value[0]);
    });

    it('validates nonzero() on 2D array', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = [x.tolist() for x in np.nonzero(arr)]
`);

      expect(result[0]!.toArray()).toEqual(npResult.value[0]);
      expect(result[1]!.toArray()).toEqual(npResult.value[1]);
    });
  });

  describe('argwhere()', () => {
    it('validates argwhere() on 1D array', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = np.argwhere(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argwhere() on 2D array', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = np.argwhere(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argwhere() on all-zero array', () => {
      const arr = array([
        [0, 0],
        [0, 0],
      ]);
      const result = argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([[0, 0], [0, 0]])
result = np.argwhere(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect((result as any).toArray()).toEqual(npResult.value);
    });

    it('validates argwhere() on 3D array', () => {
      const arr = array([
        [
          [1, 0],
          [0, 2],
        ],
        [
          [0, 3],
          [4, 0],
        ],
      ]);
      const result = argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([[[1, 0], [0, 2]], [[0, 3], [4, 0]]])
result = np.argwhere(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argwhere() with all non-zero elements', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.argwhere(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('flatnonzero()', () => {
    it('validates flatnonzero() on 2D array', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = flatnonzero(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = np.flatnonzero(arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('where()', () => {
    it('validates where() with condition only (like nonzero)', () => {
      const arr = array([1, 0, 2]);
      const result = where(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2])
result = [x.tolist() for x in np.where(arr)]
`);

      expect((result as any[])[0].toArray()).toEqual(npResult.value[0]);
    });

    it('validates where() with x and y', () => {
      const condition = array([1, 0, 1, 0]);
      const x = array([1, 2, 3, 4]);
      const y = array([10, 20, 30, 40]);
      const result = where(condition, x, y);

      const npResult = runNumPy(`
condition = np.array([1, 0, 1, 0])
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
result = np.where(condition, x, y)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('searchsorted()', () => {
    it('validates searchsorted() with side=left', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([2, 3.5, 6]);
      const result = searchsorted(arr, values, 'left');

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
values = np.array([2, 3.5, 6])
result = np.searchsorted(arr, values, side='left')
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates searchsorted() with side=right', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([2, 3, 6]);
      const result = searchsorted(arr, values, 'right');

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
values = np.array([2, 3, 6])
result = np.searchsorted(arr, values, side='right')
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates searchsorted() with duplicates', () => {
      const arr = array([1, 2, 2, 3]);
      const values = array([2]);
      const resultLeft = searchsorted(arr, values, 'left');
      const resultRight = searchsorted(arr, values, 'right');

      const npResultLeft = runNumPy(`
arr = np.array([1, 2, 2, 3])
values = np.array([2])
result = np.searchsorted(arr, values, side='left')
`);

      const npResultRight = runNumPy(`
arr = np.array([1, 2, 2, 3])
values = np.array([2])
result = np.searchsorted(arr, values, side='right')
`);

      expect(arraysClose((resultLeft as any).toArray(), npResultLeft.value)).toBe(true);
      expect(arraysClose((resultRight as any).toArray(), npResultRight.value)).toBe(true);
    });
  });

  describe('extract()', () => {
    it('validates extract() with condition', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const condition = array([1, 0, 1, 0, 1]);
      const result = extract(condition, arr);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
condition = np.array([1, 0, 1, 0, 1])
result = np.extract(condition, arr)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('count_nonzero()', () => {
    it('validates count_nonzero() on full array', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = count_nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = np.count_nonzero(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates count_nonzero() along axis=0', () => {
      const arr = array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = count_nonzero(arr, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 0, 2], [0, 3, 0]])
result = np.count_nonzero(arr, axis=0)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates count_nonzero() along axis=1', () => {
      const arr = array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = count_nonzero(arr, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 0, 2], [0, 3, 0]])
result = np.count_nonzero(arr, axis=1)
`);

      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});
