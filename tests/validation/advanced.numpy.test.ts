/**
 * Python NumPy validation tests for advanced array operations
 *
 * Tests: broadcast_to, broadcast_arrays, take, put, choose, array_equal
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  zeros,
  broadcast_to,
  broadcast_arrays,
  take,
  put,
  choose,
  array_equal,
  take_along_axis,
  compress,
  select,
  fill_diagonal,
  diag_indices,
  tril_indices,
  triu_indices,
  indices,
  ix_,
  ravel_multi_index,
  unravel_index,
  iindex,
  bindex,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Advanced Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  // ========================================
  // broadcast_to
  // ========================================
  describe('broadcast_to()', () => {
    it('validates broadcast_to 1D to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [3, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.broadcast_to(arr, (3, 3))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to scalar-like', () => {
      const arr = array([5]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([5])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to column vector', () => {
      const arr = array([[1], [2], [3]]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([[1], [2], [3]])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to row vector', () => {
      const arr = array([[1, 2, 3, 4]]);
      const result = broadcast_to(arr, [3, 4]);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4]])
result = np.broadcast_to(arr, (3, 4))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcast_to 2D to 3D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = broadcast_to(arr, [2, 2, 2]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.broadcast_to(arr, (2, 2, 2))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // broadcast_arrays
  // ========================================
  describe('broadcast_arrays()', () => {
    it('validates broadcast_arrays two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const [ra, rb] = broadcast_arrays(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
ra, rb = np.broadcast_arrays(a, b)
result = np.array([ra.tolist(), rb.tolist()])
`);

      expect(ra.shape).toEqual([3, 3]);
      expect(rb.shape).toEqual([3, 3]);
      expect(arraysClose(ra.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(rb.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates broadcast_arrays three arrays', () => {
      const a = array([1, 2, 3, 4]);
      const b = array([[1], [2]]);
      const c = array([[[1]]]);
      const [ra, rb, rc] = broadcast_arrays(a, b, c);

      const npResult = runNumPy(`
a = np.array([1, 2, 3, 4])
b = np.array([[1], [2]])
c = np.array([[[1]]])
ra, rb, rc = np.broadcast_arrays(a, b, c)
result = np.array([ra.tolist(), rb.tolist(), rc.tolist()])
`);

      expect(ra.shape).toEqual([1, 2, 4]);
      expect(rb.shape).toEqual([1, 2, 4]);
      expect(rc.shape).toEqual([1, 2, 4]);
      expect(arraysClose(ra.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(rb.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(rc.toArray(), npResult.value[2])).toBe(true);
    });
  });

  // ========================================
  // take
  // ========================================
  describe('take()', () => {
    it('validates take from flattened array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = take(arr, [0, 2, 3]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.take(arr, [0, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = take(arr, [0, 2], 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = np.take(arr, [0, 2], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = take(arr, [0, 2], 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.take(arr, [0, 2], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take with negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = take(arr, [-1, -2]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.take(arr, [-1, -2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take with duplicate indices', () => {
      const arr = array([1, 2, 3]);
      const result = take(arr, [0, 0, 1, 1]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.take(arr, [0, 0, 1, 1])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // put
  // ========================================
  describe('put()', () => {
    it('validates put scalar value', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [0, 2], 99);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 2], 99)
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put array values', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 2], values);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 2], [10, 20])
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put with negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [-1, -2], 0);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [-1, -2], 0)
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates put cycling values', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 1, 2, 3], values);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 1, 2, 3], [10, 20])
result = arr
`);

      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // choose
  // ========================================
  describe('choose()', () => {
    it('validates choose from array of choices', () => {
      const choices = [array([1, 2, 3]), array([10, 20, 30]), array([100, 200, 300])];
      const indices = array([0, 1, 2]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([100, 200, 300])]
indices = np.array([0, 1, 2])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates choose with 2D indices', () => {
      const choices = [array([0, 0]), array([1, 1])];
      const indices = array([
        [0, 1],
        [1, 0],
      ]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([0, 0]), np.array([1, 1])]
indices = np.array([[0, 1], [1, 0]])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates choose with broadcasting', () => {
      const choices = [array([10]), array([20])];
      const indices = array([0, 1, 0, 1]);
      const result = choose(indices, choices);

      const npResult = runNumPy(`
choices = [np.array([10]), np.array([20])]
indices = np.array([0, 1, 0, 1])
result = np.choose(indices, choices)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // array_equal
  // ========================================
  describe('array_equal()', () => {
    it('validates array_equal for equal arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal for different values', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const result = array_equal(a, b);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
result = np.array_equal(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal with NaN (equal_nan=False)', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      const result = array_equal(a, b, false);

      const npResult = runNumPy(`
a = np.array([1, np.nan, 3])
b = np.array([1, np.nan, 3])
result = np.array_equal(a, b, equal_nan=False)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates array_equal with NaN (equal_nan=True)', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      const result = array_equal(a, b, true);

      const npResult = runNumPy(`
a = np.array([1, np.nan, 3])
b = np.array([1, np.nan, 3])
result = np.array_equal(a, b, equal_nan=True)
`);

      expect(result).toBe(npResult.value);
    });
  });

  // ========================================
  // take_along_axis
  // ========================================
  describe('take_along_axis()', () => {
    it('validates take_along_axis along axis 0', () => {
      const arr = array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const idx = array([
        [0, 1, 0],
        [1, 0, 1],
      ]);
      const result = take_along_axis(arr, idx, 0);

      const npResult = runNumPy(`
arr = np.array([[10, 20, 30], [40, 50, 60]])
idx = np.array([[0, 1, 0], [1, 0, 1]])
result = np.take_along_axis(arr, idx, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates take_along_axis along axis 1', () => {
      const arr = array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const idx = array([[2], [0]]);
      const result = take_along_axis(arr, idx, 1);

      const npResult = runNumPy(`
arr = np.array([[10, 20, 30], [40, 50, 60]])
idx = np.array([[2], [0]])
result = np.take_along_axis(arr, idx, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // compress
  // ========================================
  describe('compress()', () => {
    it('validates compress along flattened array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const cond = array([1, 0, 1, 0, 1]);
      const result = compress(cond, arr);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
cond = np.array([1, 0, 1, 0, 1])
result = np.compress(cond, arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates compress along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const cond = array([1, 0, 1]);
      const result = compress(cond, arr, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
cond = np.array([1, 0, 1])
result = np.compress(cond, arr, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // iindex (integer array indexing)
  // ========================================
  describe('iindex()', () => {
    it('validates iindex selects rows from 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = iindex(arr, [0, 2], 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[[0, 2], :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates iindex selects columns from 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = iindex(arr, [0, 2], 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.take(arr, [0, 2], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates iindex with negative indices', () => {
      const arr = array([10, 20, 30, 40, 50]);
      const result = iindex(arr, [-1, -3], 0);

      const npResult = runNumPy(`
arr = np.array([10, 20, 30, 40, 50])
result = np.take(arr, [-1, -3], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates iindex with NDArray indices', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const indices = array([0, 2]);
      const result = iindex(arr, indices, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = np.take(arr, [0, 2], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // bindex (boolean array indexing)
  // ========================================
  describe('bindex()', () => {
    it('validates bindex selects elements with boolean mask', () => {
      const arr = array([10, 20, 30, 40, 50]);
      const mask = array([true, false, true, false, true], 'bool');
      const result = bindex(arr, mask);

      const npResult = runNumPy(`
arr = np.array([10, 20, 30, 40, 50])
mask = np.array([True, False, True, False, True])
result = arr[mask]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates bindex selects rows with axis=0', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const mask = array([true, false, true], 'bool');
      const result = bindex(arr, mask, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([True, False, True])
result = np.compress(mask, arr, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates bindex selects columns with axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const mask = array([true, false, true], 'bool');
      const result = bindex(arr, mask, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
mask = np.array([True, False, True])
result = np.compress(mask, arr, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates bindex with comparison-generated mask', () => {
      const arr = array([1, 5, 3, 8, 2, 9, 4]);
      const mask = arr.greater(4);
      const result = bindex(arr, mask);

      const npResult = runNumPy(`
arr = np.array([1, 5, 3, 8, 2, 9, 4])
result = arr[arr > 4]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates bindex returns empty array when no matches', () => {
      const arr = array([1, 2, 3]);
      const mask = array([false, false, false], 'bool');
      const result = bindex(arr, mask);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
mask = np.array([False, False, False])
result = arr[mask]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(result.size).toBe(0);
    });
  });

  // ========================================
  // select
  // ========================================
  describe('select()', () => {
    it('validates select from choicelist', () => {
      const x = array([0, 1, 2, 3]);
      const cond1 = x.less(1);
      const cond2 = x.greater(2);
      const result = select([cond1, cond2], [array([10, 10, 10, 10]), array([20, 20, 20, 20])], 0);

      const npResult = runNumPy(`
x = np.array([0, 1, 2, 3])
cond1 = x < 1
cond2 = x > 2
result = np.select([cond1, cond2], [np.full(4, 10), np.full(4, 20)], default=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // diag_indices
  // ========================================
  describe('diag_indices()', () => {
    it('validates diag_indices 2D', () => {
      const [rows, cols] = diag_indices(3);

      const npResult = runNumPy(`
rows, cols = np.diag_indices(3)
result = [rows.tolist(), cols.tolist()]
`);

      expect(arraysClose(rows.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(cols.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates diag_indices 3D', () => {
      const result = diag_indices(2, 3);

      const npResult = runNumPy(`
result = np.diag_indices(2, 3)
result = [r.tolist() for r in result]
`);

      expect(result.length).toBe(3);
      for (let i = 0; i < 3; i++) {
        expect(arraysClose(result[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });
  });

  // ========================================
  // tril_indices
  // ========================================
  describe('tril_indices()', () => {
    it('validates tril_indices basic', () => {
      const [rows, cols] = tril_indices(3);

      const npResult = runNumPy(`
rows, cols = np.tril_indices(3)
result = [rows.tolist(), cols.tolist()]
`);

      expect(arraysClose(rows.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(cols.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates tril_indices with k=1', () => {
      const [rows, cols] = tril_indices(3, 1);

      const npResult = runNumPy(`
rows, cols = np.tril_indices(3, k=1)
result = [rows.tolist(), cols.tolist()]
`);

      expect(arraysClose(rows.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(cols.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // triu_indices
  // ========================================
  describe('triu_indices()', () => {
    it('validates triu_indices basic', () => {
      const [rows, cols] = triu_indices(3);

      const npResult = runNumPy(`
rows, cols = np.triu_indices(3)
result = [rows.tolist(), cols.tolist()]
`);

      expect(arraysClose(rows.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(cols.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates triu_indices with k=1', () => {
      const [rows, cols] = triu_indices(3, 1);

      const npResult = runNumPy(`
rows, cols = np.triu_indices(3, k=1)
result = [rows.tolist(), cols.tolist()]
`);

      expect(arraysClose(rows.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(cols.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // indices
  // ========================================
  describe('indices()', () => {
    it('validates indices 2D grid', () => {
      const result = indices([2, 3]);

      const npResult = runNumPy(`
result = np.indices((2, 3))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // ix_
  // ========================================
  describe('ix_()', () => {
    it('validates ix_ open mesh', () => {
      const result = ix_(array([0, 1]), array([2, 4, 6]));

      const npResult = runNumPy(`
r0, r1 = np.ix_(np.array([0, 1]), np.array([2, 4, 6]))
result = [r0.tolist(), r1.tolist()]
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // ravel_multi_index
  // ========================================
  describe('ravel_multi_index()', () => {
    it('validates ravel_multi_index', () => {
      const result = ravel_multi_index([array([0, 1, 2]), array([0, 1, 2])], [3, 3]);

      const npResult = runNumPy(`
result = np.ravel_multi_index(([0, 1, 2], [0, 1, 2]), (3, 3))
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates ravel_multi_index wrap mode', () => {
      const result = ravel_multi_index([array([3]), array([1])], [3, 3], 'wrap');

      const npResult = runNumPy(`
result = np.ravel_multi_index(([3], [1]), (3, 3), mode='wrap')
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // unravel_index
  // ========================================
  describe('unravel_index()', () => {
    it('validates unravel_index', () => {
      const result = unravel_index(array([0, 4, 8]), [3, 3]);

      const npResult = runNumPy(`
result = np.unravel_index([0, 4, 8], (3, 3))
result = [np.array(r).tolist() for r in result]
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });

    it('validates unravel_index 3D shape', () => {
      const result = unravel_index(array([0, 5, 23]), [2, 3, 4]);

      const npResult = runNumPy(`
result = np.unravel_index([0, 5, 23], (2, 3, 4))
result = [np.array(r).tolist() for r in result]
`);

      expect(result.length).toBe(3);
      for (let i = 0; i < 3; i++) {
        expect(arraysClose(result[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });
  });

  describe('fill_diagonal()', () => {
    it('validates fill_diagonal() with scalar', () => {
      const arr = zeros([3, 3]);
      fill_diagonal(arr, 5);

      const npArr = runNumPy(`
arr = np.zeros((3, 3))
np.fill_diagonal(arr, 5)
result = arr
`);

      expect(arraysClose(arr.toArray(), npArr.value)).toBe(true);
    });

    it('validates fill_diagonal() with array', () => {
      const arr = zeros([3, 3]);
      fill_diagonal(arr, array([1, 2, 3]));

      const npArr = runNumPy(`
arr = np.zeros((3, 3))
np.fill_diagonal(arr, [1, 2, 3])
result = arr
`);

      expect(arraysClose(arr.toArray(), npArr.value)).toBe(true);
    });

    it('validates fill_diagonal() on non-square matrix', () => {
      const arr = zeros([3, 4]);
      fill_diagonal(arr, 7);

      const npArr = runNumPy(`
arr = np.zeros((3, 4))
np.fill_diagonal(arr, 7)
result = arr
`);

      expect(arraysClose(arr.toArray(), npArr.value)).toBe(true);
    });
  });
});
