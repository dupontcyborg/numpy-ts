/**
 * Unit tests for advanced array functions
 *
 * Tests: broadcast_to, broadcast_arrays, take, put, choose, array_equal
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  broadcast_to,
  broadcast_arrays,
  broadcast_shapes,
  take,
  put,
  choose,
  array_equal,
  fill_diagonal,
  compress,
  select,
  take_along_axis,
  put_along_axis,
  putmask,
  place,
  copyto,
  where,
  extract,
  count_nonzero,
  nonzero,
  flatnonzero,
  searchsorted,
  apply_along_axis,
  may_share_memory,
  shares_memory,
  copy,
  partition,
} from '../../src';

describe('Advanced Functions', () => {
  // ========================================
  // broadcast_to
  // ========================================
  describe('broadcast_to', () => {
    it('broadcasts 1D array to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [3, 3]);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
      ]);
    });

    it('broadcasts scalar-like array', () => {
      const arr = array([5]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
      ]);
    });

    it('broadcasts column vector', () => {
      const arr = array([[1], [2], [3]]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
      ]);
    });

    it('broadcasts row vector', () => {
      const arr = array([[1, 2, 3, 4]]);
      const result = broadcast_to(arr, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
      ]);
    });

    it('returns a view', () => {
      const arr = array([1, 2, 3]);
      const result = broadcast_to(arr, [2, 3]);
      expect(result.base).toBe(arr);
    });

    it('throws error for incompatible shapes', () => {
      const arr = array([1, 2, 3]);
      expect(() => broadcast_to(arr, [2, 4])).toThrow();
    });

    it('throws error when target has fewer dimensions', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => broadcast_to(arr, [4])).toThrow();
    });
  });

  // ========================================
  // broadcast_arrays
  // ========================================
  describe('broadcast_arrays', () => {
    it('broadcasts multiple arrays to common shape', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const [ra, rb] = broadcast_arrays(a, b);
      expect(ra.shape).toEqual([3, 3]);
      expect(rb.shape).toEqual([3, 3]);
      expect(ra.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
      ]);
      expect(rb.toArray()).toEqual([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
      ]);
    });

    it('handles single array', () => {
      const arr = array([1, 2, 3]);
      const [result] = broadcast_arrays(arr);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles arrays with same shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const [ra, rb] = broadcast_arrays(a, b);
      expect(ra.shape).toEqual([2, 2]);
      expect(rb.shape).toEqual([2, 2]);
    });

    it('broadcasts three arrays', () => {
      const a = array([1, 2, 3, 4]);
      const b = array([[1], [2]]);
      const c = array([[[1]]]);
      const [ra, rb, rc] = broadcast_arrays(a, b, c);
      expect(ra.shape).toEqual([1, 2, 4]);
      expect(rb.shape).toEqual([1, 2, 4]);
      expect(rc.shape).toEqual([1, 2, 4]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2]);
      expect(() => broadcast_arrays(a, b)).toThrow();
    });

    it('handles empty array', () => {
      const result = broadcast_arrays();
      expect(result).toEqual([]);
    });
  });

  // ========================================
  // take
  // ========================================
  describe('take', () => {
    it('takes elements from flattened array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = take(arr, [0, 2, 3]);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 3, 4]);
    });

    it('takes elements along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = take(arr, [0, 2], 0);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [5, 6],
      ]);
    });

    it('takes elements along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = take(arr, [0, 2], 1);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [4, 6],
      ]);
    });

    it('handles negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = take(arr, [-1, -2]);
      expect(result.toArray()).toEqual([5, 4]);
    });

    it('handles duplicate indices', () => {
      const arr = array([1, 2, 3]);
      const result = take(arr, [0, 0, 1, 1]);
      expect(result.toArray()).toEqual([1, 1, 2, 2]);
    });

    it('takes elements from BigInt array', () => {
      const arr = array([1n, 2n, 3n, 4n, 5n], 'int64');
      const result = take(arr, [0, 2, 4]);
      expect(result.toArray()).toEqual([1n, 3n, 5n]);
      expect(result.dtype).toBe('int64');
    });

    it('throws error for invalid axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => take(arr, [0], 5)).toThrow('axis 5 is out of bounds');
    });

    it('throws error for out-of-bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => take(arr, [10])).toThrow();
    });
  });

  // ========================================
  // put
  // ========================================
  describe('put', () => {
    it('puts scalar value at indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [0, 2], 99);
      expect(arr.toArray()).toEqual([99, 2, 99, 4, 5]);
    });

    it('puts array values at indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 2], values);
      expect(arr.toArray()).toEqual([10, 2, 20, 4, 5]);
    });

    it('handles negative indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      put(arr, [-1, -2], 0);
      expect(arr.toArray()).toEqual([1, 2, 3, 0, 0]);
    });

    it('cycles values if not enough provided', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([10, 20]);
      put(arr, [0, 1, 2, 3], values);
      expect(arr.toArray()).toEqual([10, 20, 10, 20, 5]);
    });

    it('throws error for out-of-bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => put(arr, [10], 5)).toThrow();
    });
  });

  // ========================================
  // choose
  // ========================================
  describe('choose', () => {
    it('chooses from array of choices', () => {
      const choices = [array([1, 2, 3]), array([10, 20, 30]), array([100, 200, 300])];
      const indices = array([0, 1, 2]);
      const result = choose(indices, choices);
      expect(result.toArray()).toEqual([1, 20, 300]);
    });

    it('handles 2D indices', () => {
      const choices = [array([0, 0]), array([1, 1])];
      const indices = array([
        [0, 1],
        [1, 0],
      ]);
      const result = choose(indices, choices);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 1],
        [1, 0],
      ]);
    });

    it('broadcasts choices to index shape', () => {
      const choices = [array([10]), array([20])];
      const indices = array([0, 1, 0, 1]);
      const result = choose(indices, choices);
      expect(result.toArray()).toEqual([10, 20, 10, 20]);
    });

    it('throws error for out-of-bounds choice index', () => {
      const choices = [array([1, 2, 3])];
      const indices = array([0, 1]); // index 1 is out of bounds
      expect(() => choose(indices, choices)).toThrow();
    });
  });

  // ========================================
  // array_equal
  // ========================================
  describe('array_equal', () => {
    it('returns true for equal arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      expect(array_equal(a, b)).toBe(true);
    });

    it('returns false for different values', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('returns false for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('returns false for different sizes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('handles 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(array_equal(a, b)).toBe(true);
    });

    it('handles NaN values with equal_nan=false', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      expect(array_equal(a, b, false)).toBe(false);
    });

    it('handles NaN values with equal_nan=true', () => {
      const a = array([1, NaN, 3]);
      const b = array([1, NaN, 3]);
      expect(array_equal(a, b, true)).toBe(true);
    });

    it('handles empty arrays', () => {
      const a = array([]);
      const b = array([]);
      expect(array_equal(a, b)).toBe(true);
    });
  });

  // ========================================
  // broadcast_shapes
  // ========================================
  describe('broadcast_shapes', () => {
    it('computes broadcast shape for compatible shapes', () => {
      const r = broadcast_shapes([3, 1], [1, 4]);
      expect(r).toEqual([3, 4]);
    });
  });

  // ========================================
  // copyto
  // ========================================
  describe('copyto', () => {
    it('copies values from source to destination', () => {
      const dst = zeros([3]);
      const src = array([1, 2, 3]);
      copyto(dst, src);
      expect(dst.toArray()).toEqual([1, 2, 3]);
    });
  });

  // ========================================
  // where
  // ========================================
  describe('where', () => {
    it('selects elements based on condition', () => {
      const cond = array([1, 0, 1], 'bool');
      const r = where(cond, array([1, 2, 3]), array([4, 5, 6]));
      expect(r.toArray()).toEqual([1, 5, 3]);
    });
  });

  // ========================================
  // extract
  // ========================================
  describe('extract', () => {
    it('extracts elements where condition is true', () => {
      const a = array([1, 2, 3, 4]);
      const cond = array([1, 0, 1, 0], 'bool');
      expect(extract(cond, a).toArray()).toEqual([1, 3]);
    });
  });

  // ========================================
  // count_nonzero
  // ========================================
  describe('count_nonzero', () => {
    it('counts non-zero elements', () => {
      const a = array([0, 1, 2, 0, 3]);
      expect(count_nonzero(a)).toBe(3);
    });
  });

  // ========================================
  // nonzero
  // ========================================
  describe('nonzero', () => {
    it('returns indices of non-zero elements', () => {
      const a = array([0, 1, 2, 0, 3]);
      const r = nonzero(a);
      expect(r.length).toBeGreaterThan(0);
    });
  });

  // ========================================
  // flatnonzero
  // ========================================
  describe('flatnonzero', () => {
    it('returns flat indices of non-zero elements', () => {
      const a = array([0, 1, 0, 2]);
      const r = flatnonzero(a);
      expect(r.toArray()).toEqual([1, 3]);
    });
  });

  // ========================================
  // searchsorted
  // ========================================
  describe('searchsorted', () => {
    it('finds insertion indices to maintain sorted order', () => {
      const a = array([1, 3, 5, 7]);
      const r = searchsorted(a, array([2, 4, 6]));
      expect(r.toArray()).toEqual([1, 2, 3]);
    });
  });

  // ========================================
  // apply_along_axis
  // ========================================
  describe('apply_along_axis', () => {
    it('applies function along axis', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const r = apply_along_axis(
        (x: any) => {
          let s = 0;
          for (let i = 0; i < x.size; i++) s += x.iget(i);
          return s;
        },
        1,
        a
      );
      expect(r.toArray()).toEqual([6, 15]);
    });
  });

  // ========================================
  // may_share_memory / shares_memory
  // ========================================
  describe('may_share_memory', () => {
    it('returns true for slices of same array', () => {
      const a = array([1, 2, 3]);
      const b = a.slice('1:3');
      expect(may_share_memory(a, b)).toBe(true);
    });
  });

  describe('shares_memory', () => {
    it('returns false for independent copies', () => {
      const a = array([1, 2, 3]);
      const b = copy(a);
      expect(shares_memory(a, b)).toBe(false);
    });
  });

  // ========================================
  // partition (sorting-related advanced op)
  // ========================================
  describe('partition', () => {
    it('partitions array around kth element', () => {
      const a = array([3, 1, 4, 1, 5, 9]);
      const r = partition(a, 2);
      expect(r.size).toBe(6);
    });
  });
});

// Import new indexing functions
import {
  diag_indices,
  diag_indices_from,
  tril_indices,
  tril_indices_from,
  triu_indices,
  triu_indices_from,
  mask_indices,
  indices,
  ix_,
  ravel_multi_index,
  unravel_index,
  tril,
} from '../../src';

describe('Indexing Functions', () => {
  // ========================================
  // take_along_axis
  // ========================================
  describe('take_along_axis', () => {
    it('takes values along axis 0', () => {
      const arr = array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const idx = array([
        [0, 1, 0],
        [1, 0, 1],
      ]);
      const result = take_along_axis(arr, idx, 0);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [10, 50, 30],
        [40, 20, 60],
      ]);
    });

    it('takes values along axis 1', () => {
      const arr = array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const idx = array([[2], [0]]);
      const result = take_along_axis(arr, idx, 1);
      expect(result.shape).toEqual([2, 1]);
      expect(result.toArray()).toEqual([[30], [40]]);
    });

    it('handles negative axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const idx = array([
        [1, 0],
        [0, 1],
      ]);
      const result = take_along_axis(arr, idx, -1);
      expect(result.toArray()).toEqual([
        [2, 1],
        [3, 4],
      ]);
    });
  });

  // ========================================
  // put_along_axis
  // ========================================
  describe('put_along_axis', () => {
    it('puts values along axis 0', () => {
      const arr = array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const idx = array([[0], [1]]);
      const vals = array([[99], [99]]);
      put_along_axis(arr, idx, vals, 1);
      expect(arr.get([0, 0])).toBe(99);
      expect(arr.get([1, 1])).toBe(99);
    });

    it('modifies array in place', () => {
      const arr = zeros([2, 3]);
      const idx = array([
        [0, 1, 2],
        [2, 1, 0],
      ]);
      const vals = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      put_along_axis(arr, idx, vals, 1);
      expect(arr.get([0, 0])).toBe(1);
      expect(arr.get([0, 1])).toBe(2);
      expect(arr.get([0, 2])).toBe(3);
    });
  });

  // ========================================
  // putmask
  // ========================================
  describe('putmask', () => {
    it('puts values where mask is true', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const mask = array([1, 0, 1, 0, 1]);
      putmask(arr, mask, 99);
      expect(arr.toArray()).toEqual([99, 2, 99, 4, 99]);
    });

    it('cycles values when needed', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const mask = array([1, 1, 1, 1, 1]);
      const vals = array([10, 20]);
      putmask(arr, mask, vals);
      expect(arr.toArray()).toEqual([10, 20, 10, 20, 10]);
    });
  });

  // ========================================
  // compress
  // ========================================
  describe('compress', () => {
    it('compresses along flattened array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const cond = array([1, 0, 1, 0, 1]);
      const result = compress(cond, arr);
      expect(result.toArray()).toEqual([1, 3, 5]);
    });

    it('compresses along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const cond = array([1, 0, 1]);
      const result = compress(cond, arr, 0);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [5, 6],
      ]);
    });

    it('compresses along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const cond = array([1, 0, 1]);
      const result = compress(cond, arr, 1);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [4, 6],
      ]);
    });
  });

  // ========================================
  // select
  // ========================================
  describe('select', () => {
    it('selects from choicelist based on conditions', () => {
      const x = array([0, 1, 2, 3]);
      const cond1 = x.less(1);
      const cond2 = x.greater(2);
      const result = select([cond1, cond2], [array([10, 10, 10, 10]), array([20, 20, 20, 20])], 0);
      expect(result.toArray()).toEqual([10, 0, 0, 20]);
    });

    it('uses default value when no condition matches', () => {
      const cond = array([0, 0, 0]);
      const choice = array([1, 2, 3]);
      const result = select([cond], [choice], 99);
      expect(result.toArray()).toEqual([99, 99, 99]);
    });
  });

  // ========================================
  // place
  // ========================================
  describe('place', () => {
    it('places values where mask is true', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const mask = array([1, 0, 1, 0, 1]);
      const vals = array([10, 20, 30]);
      place(arr, mask, vals);
      expect(arr.toArray()).toEqual([10, 2, 20, 4, 30]);
    });

    it('cycles values if not enough provided', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const mask = array([1, 0, 1, 0, 1]);
      const vals = array([10, 20]);
      place(arr, mask, vals);
      expect(arr.toArray()).toEqual([10, 2, 20, 4, 10]);
    });
  });

  // ========================================
  // diag_indices
  // ========================================
  describe('diag_indices', () => {
    it('returns indices for 2D diagonal', () => {
      const [rows, cols] = diag_indices(3);
      expect(rows.toArray()).toEqual([0, 1, 2]);
      expect(cols.toArray()).toEqual([0, 1, 2]);
    });

    it('returns indices for 3D diagonal', () => {
      const result = diag_indices(2, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([0, 1]);
      expect(result[1]!.toArray()).toEqual([0, 1]);
      expect(result[2]!.toArray()).toEqual([0, 1]);
    });
  });

  // ========================================
  // diag_indices_from
  // ========================================
  describe('diag_indices_from', () => {
    it('returns indices from array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const [rows, cols] = diag_indices_from(arr);
      expect(rows.toArray()).toEqual([0, 1, 2]);
      expect(cols.toArray()).toEqual([0, 1, 2]);
    });

    it('throws for non-square array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => diag_indices_from(arr)).toThrow();
    });
  });

  // ========================================
  // tril_indices
  // ========================================
  describe('tril_indices', () => {
    it('returns lower triangle indices', () => {
      const [rows, cols] = tril_indices(3);
      expect(rows.toArray()).toEqual([0, 1, 1, 2, 2, 2]);
      expect(cols.toArray()).toEqual([0, 0, 1, 0, 1, 2]);
    });

    it('handles diagonal offset k=1', () => {
      const [rows, cols] = tril_indices(3, 1);
      expect(rows.toArray()).toEqual([0, 0, 1, 1, 1, 2, 2, 2]);
      expect(cols.toArray()).toEqual([0, 1, 0, 1, 2, 0, 1, 2]);
    });

    it('handles rectangular matrix', () => {
      const [rows, cols] = tril_indices(3, 0, 4);
      expect(rows.toArray()).toEqual([0, 1, 1, 2, 2, 2]);
      expect(cols.toArray()).toEqual([0, 0, 1, 0, 1, 2]);
    });
  });

  // ========================================
  // tril_indices_from
  // ========================================
  describe('tril_indices_from', () => {
    it('returns indices from array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const [rows, cols] = tril_indices_from(arr);
      expect(rows.toArray()).toEqual([0, 1, 1, 2, 2, 2]);
      expect(cols.toArray()).toEqual([0, 0, 1, 0, 1, 2]);
    });
  });

  // ========================================
  // triu_indices
  // ========================================
  describe('triu_indices', () => {
    it('returns upper triangle indices', () => {
      const [rows, cols] = triu_indices(3);
      expect(rows.toArray()).toEqual([0, 0, 0, 1, 1, 2]);
      expect(cols.toArray()).toEqual([0, 1, 2, 1, 2, 2]);
    });

    it('handles diagonal offset k=1', () => {
      const [rows, cols] = triu_indices(3, 1);
      expect(rows.toArray()).toEqual([0, 0, 1]);
      expect(cols.toArray()).toEqual([1, 2, 2]);
    });
  });

  // ========================================
  // triu_indices_from
  // ========================================
  describe('triu_indices_from', () => {
    it('returns indices from array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const [rows, cols] = triu_indices_from(arr);
      expect(rows.toArray()).toEqual([0, 0, 0, 1, 1, 2]);
      expect(cols.toArray()).toEqual([0, 1, 2, 1, 2, 2]);
    });
  });

  // ========================================
  // mask_indices
  // ========================================
  describe('mask_indices', () => {
    it('returns indices using mask function', () => {
      // Use tril as mask function
      const [rows, cols] = mask_indices(3, (n, k) => tril(zeros([n, n]).add(1), k));
      expect(rows.toArray()).toEqual([0, 1, 1, 2, 2, 2]);
      expect(cols.toArray()).toEqual([0, 0, 1, 0, 1, 2]);
    });
  });

  // ========================================
  // indices
  // ========================================
  describe('indices', () => {
    it('returns indices for 2D grid', () => {
      const result = indices([2, 3]);
      expect(result.shape).toEqual([2, 2, 3]);
      // First slice is row indices
      expect(result.get([0, 0, 0])).toBe(0);
      expect(result.get([0, 0, 1])).toBe(0);
      expect(result.get([0, 0, 2])).toBe(0);
      expect(result.get([0, 1, 0])).toBe(1);
      expect(result.get([0, 1, 1])).toBe(1);
      expect(result.get([0, 1, 2])).toBe(1);
      // Second slice is column indices
      expect(result.get([1, 0, 0])).toBe(0);
      expect(result.get([1, 0, 1])).toBe(1);
      expect(result.get([1, 0, 2])).toBe(2);
    });
  });

  // ========================================
  // ix_
  // ========================================
  describe('ix_', () => {
    it('creates open mesh arrays', () => {
      const result = ix_(array([0, 1]), array([2, 4, 6]));
      expect(result.length).toBe(2);
      expect(result[0]!.shape).toEqual([2, 1]);
      expect(result[1]!.shape).toEqual([1, 3]);
      expect(result[0]!.toArray()).toEqual([[0], [1]]);
      expect(result[1]!.toArray()).toEqual([[2, 4, 6]]);
    });

    it('creates 3D open mesh', () => {
      const result = ix_(array([0, 1]), array([2, 3]), array([4]));
      expect(result.length).toBe(3);
      expect(result[0]!.shape).toEqual([2, 1, 1]);
      expect(result[1]!.shape).toEqual([1, 2, 1]);
      expect(result[2]!.shape).toEqual([1, 1, 1]);
    });
  });

  // ========================================
  // ravel_multi_index
  // ========================================
  describe('ravel_multi_index', () => {
    it('converts multi-index to flat index', () => {
      const result = ravel_multi_index([array([0, 1, 2]), array([0, 1, 2])], [3, 3]);
      expect(result.toArray()).toEqual([0, 4, 8]);
    });

    it('handles wrap mode', () => {
      const result = ravel_multi_index([array([3]), array([1])], [3, 3], 'wrap');
      expect(result.toArray()).toEqual([1]); // 3%3=0, so row 0, col 1 = 1
    });

    it('handles clip mode', () => {
      const result = ravel_multi_index([array([5]), array([1])], [3, 3], 'clip');
      expect(result.toArray()).toEqual([7]); // clipped to row 2, col 1 = 7
    });
  });

  // ========================================
  // unravel_index
  // ========================================
  describe('unravel_index', () => {
    it('converts flat index to multi-index', () => {
      const result = unravel_index(array([0, 4, 8]), [3, 3]);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([0, 1, 2]);
      expect(result[1]!.toArray()).toEqual([0, 1, 2]);
    });

    it('handles scalar input', () => {
      const result = unravel_index(5, [3, 3]);
      expect(result.length).toBe(2);
      // Scalar output - access first element of data
      expect(result[0]!.data[0]).toBe(1);
      expect(result[1]!.data[0]).toBe(2);
    });

    it('handles 3D shape', () => {
      const result = unravel_index(array([0, 5, 23]), [2, 3, 4]);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([0, 0, 1]);
      expect(result[1]!.toArray()).toEqual([0, 1, 2]);
      expect(result[2]!.toArray()).toEqual([0, 1, 3]);
    });
  });

  describe('fill_diagonal', () => {
    it('fills diagonal with scalar value', () => {
      const arr = zeros([3, 3]);
      fill_diagonal(arr, 5);
      expect(arr.toArray()).toEqual([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 5],
      ]);
    });

    it('fills diagonal with array values', () => {
      const arr = zeros([3, 3]);
      fill_diagonal(arr, array([1, 2, 3]));
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
    });

    it('handles non-square matrices', () => {
      const arr = zeros([3, 4]);
      fill_diagonal(arr, 7);
      expect(arr.toArray()).toEqual([
        [7, 0, 0, 0],
        [0, 7, 0, 0],
        [0, 0, 7, 0],
      ]);
    });

    it('handles tall matrices', () => {
      const arr = zeros([4, 3]);
      fill_diagonal(arr, 7);
      expect(arr.toArray()).toEqual([
        [7, 0, 0],
        [0, 7, 0],
        [0, 0, 7],
        [0, 0, 0],
      ]);
    });

    it('cycles through values if array is shorter', () => {
      const arr = zeros([3, 3]);
      fill_diagonal(arr, array([1, 2]));
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
      ]);
    });
  });
});
