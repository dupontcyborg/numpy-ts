/**
 * Unit tests for sorting and searching operations
 */

import { describe, it, expect } from 'vitest';
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
  zeros,
  ones,
  Complex,
} from '../../src';

describe('Sorting Functions', () => {
  describe('sort()', () => {
    it('sorts a 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      const result = sort(arr);
      expect(result.toArray()).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
    });

    it('sorts a 2D array along last axis (default)', () => {
      const arr = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = sort(arr);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('sorts a 2D array along axis 0', () => {
      const arr = array([
        [3, 1, 2],
        [0, 4, 1],
      ]);
      const result = sort(arr, 0);
      expect(result.toArray()).toEqual([
        [0, 1, 1],
        [3, 4, 2],
      ]);
    });

    it('handles negative axis', () => {
      const arr = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = sort(arr, -1);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('handles NaN values (placed at end)', () => {
      const arr = array([3, NaN, 1, NaN, 2]);
      const result = sort(arr);
      const resultArr = result.toArray() as number[];
      expect(resultArr.slice(0, 3)).toEqual([1, 2, 3]);
      expect(isNaN(resultArr[3]!)).toBe(true);
      expect(isNaN(resultArr[4]!)).toBe(true);
    });

    it('uses method syntax', () => {
      const arr = array([3, 1, 2]);
      const result = arr.sort();
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('sorts BigInt arrays', () => {
      const arr = array([5n, 2n, 8n, 1n, 9n], 'int64');
      const result = sort(arr);
      expect(result.toArray()).toEqual([1n, 2n, 5n, 8n, 9n]);
      expect(result.dtype).toBe('int64');
    });

    it('sorts 2D BigInt array along axis', () => {
      const arr = array(
        [
          [3n, 1n, 2n],
          [6n, 4n, 5n],
        ],
        'int64'
      );
      const result = sort(arr, 1);
      expect(result.toArray()).toEqual([
        [1n, 2n, 3n],
        [4n, 5n, 6n],
      ]);
    });
  });

  describe('argsort()', () => {
    it('returns indices that would sort a 1D array', () => {
      const arr = array([3, 1, 4, 1, 5]);
      const result = argsort(arr);
      expect(result.toArray()).toEqual([1, 3, 0, 2, 4]);
    });

    it('returns indices that would sort along axis', () => {
      const arr = array([
        [3, 1, 2],
        [6, 4, 5],
      ]);
      const result = argsort(arr, -1);
      expect(result.toArray()).toEqual([
        [1, 2, 0],
        [1, 2, 0],
      ]);
    });

    it('works with axis 0', () => {
      const arr = array([
        [3, 1],
        [1, 4],
      ]);
      const result = argsort(arr, 0);
      expect(result.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([3, 1, 2]);
      const result = arr.argsort();
      expect(result.toArray()).toEqual([1, 2, 0]);
    });
  });

  describe('lexsort()', () => {
    it('sorts by multiple keys', () => {
      const surnames = array([1, 2, 1, 2]); // 1=Smith, 2=Jones (secondary)
      const firstNames = array([3, 1, 2, 4]); // First names (primary)
      const result = lexsort([surnames, firstNames]);
      // Should sort primarily by firstNames, then by surnames
      expect(result.toArray()).toEqual([1, 2, 0, 3]);
    });

    it('handles single key', () => {
      const arr = array([3, 1, 2]);
      const result = lexsort([arr]);
      expect(result.toArray()).toEqual([1, 2, 0]);
    });

    it('returns empty for empty keys', () => {
      const result = lexsort([]);
      expect(result.toArray()).toEqual([]);
    });
  });

  describe('partition()', () => {
    it('partially sorts array so kth element is in sorted position', () => {
      const arr = array([3, 4, 2, 1]);
      const result = partition(arr, 2);
      // Element at index 2 in sorted array would be 3
      // Elements before index 2 should be <= 3, elements after >= 3
      const resultArr = result.toArray() as number[];
      // The result is fully sorted for our simple implementation
      expect(resultArr[2]).toBe(3);
    });

    it('works with negative kth', () => {
      const arr = array([3, 4, 2, 1]);
      const result = partition(arr, -1);
      const resultArr = result.toArray() as number[];
      expect(resultArr[3]).toBe(4); // Last element should be max
    });

    it('uses method syntax', () => {
      const arr = array([3, 1, 2]);
      const result = arr.partition(1);
      const resultArr = result.toArray() as number[];
      expect(resultArr[1]).toBe(2);
    });
  });

  describe('argpartition()', () => {
    it('returns indices for partition', () => {
      const arr = array([3, 4, 2, 1]);
      const result = argpartition(arr, 2);
      // Since our implementation fully sorts, this should be same as argsort
      expect(result.shape).toEqual([4]);
    });

    it('uses method syntax', () => {
      const arr = array([3, 1, 2]);
      const result = arr.argpartition(1);
      expect(result.shape).toEqual([3]);
    });
  });

  describe('sort_complex()', () => {
    it('returns sorted 1D array for real arrays (as complex128)', () => {
      const arr = array([
        [3, 1],
        [4, 2],
      ]);
      const result = sort_complex(arr);
      expect(result.shape).toEqual([4]);
      // NumPy always returns complex128 for sort_complex
      expect(result.dtype).toBe('complex128');
      const values = result.toArray() as Complex[];
      expect(values.map((c) => c.re)).toEqual([1, 2, 3, 4]);
      expect(values.every((c) => c.im === 0)).toBe(true);
    });
  });
});

describe('Searching Functions', () => {
  describe('nonzero()', () => {
    it('finds indices of non-zero elements in 1D array', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = nonzero(arr);
      expect(result.length).toBe(1);
      expect(result[0]!.toArray()).toEqual([0, 2, 4]);
    });

    it('finds indices of non-zero elements in 2D array', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = nonzero(arr);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([0, 1]); // row indices
      expect(result[1]!.toArray()).toEqual([0, 1]); // col indices
    });

    it('returns empty arrays for all-zero array', () => {
      const arr = zeros([3, 3]);
      const result = nonzero(arr);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([]);
      expect(result[1]!.toArray()).toEqual([]);
    });

    it('uses method syntax', () => {
      const arr = array([1, 0, 2]);
      const result = arr.nonzero();
      expect(result[0]!.toArray()).toEqual([0, 2]);
    });
  });

  describe('argwhere()', () => {
    it('finds indices of non-zero elements in 1D array', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = argwhere(arr);
      expect(result.shape).toEqual([3, 1]);
      expect(result.toArray()).toEqual([[0], [2], [4]]);
    });

    it('finds indices of non-zero elements in 2D array', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = argwhere(arr);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 0],
        [1, 1],
      ]);
    });

    it('returns empty array for all-zero array', () => {
      const arr = zeros([3, 3]);
      const result = argwhere(arr);
      expect(result.shape).toEqual([0, 2]);
      expect(result.toArray()).toEqual([]);
    });

    it('works with 3D array', () => {
      const arr = zeros([2, 2, 2]);
      arr.set([0, 0, 0], 1);
      arr.set([1, 1, 1], 2);
      const result = argwhere(arr);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [0, 0, 0],
        [1, 1, 1],
      ]);
    });

    it('works with all non-zero elements', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = argwhere(arr);
      expect(result.shape).toEqual([4, 2]);
      expect(result.toArray()).toEqual([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1, 0, 2]);
      const result = arr.argwhere();
      expect(result.shape).toEqual([2, 1]);
      expect(result.toArray()).toEqual([[0], [2]]);
    });
  });

  describe('flatnonzero()', () => {
    it('returns flat indices of non-zero elements', () => {
      const arr = array([
        [1, 0],
        [0, 2],
      ]);
      const result = flatnonzero(arr);
      expect(result.toArray()).toEqual([0, 3]);
    });
  });

  describe('where()', () => {
    it('returns indices when only condition given (like nonzero)', () => {
      const arr = array([1, 0, 2]);
      const result = where(arr);
      expect(Array.isArray(result)).toBe(true);
      expect((result as any[])[0].toArray()).toEqual([0, 2]);
    });

    it('selects from x or y based on condition', () => {
      const condition = array([1, 0, 1, 0]);
      const x = array([1, 2, 3, 4]);
      const y = array([10, 20, 30, 40]);
      const result = where(condition, x, y);
      expect((result as any).toArray()).toEqual([1, 20, 3, 40]);
    });

    it('works with scalar broadcasting', () => {
      const condition = array([1, 0, 1]);
      const x = array([1, 1, 1]);
      const y = array([0, 0, 0]);
      const result = where(condition, x, y);
      expect((result as any).toArray()).toEqual([1, 0, 1]);
    });
  });

  describe('searchsorted()', () => {
    it('finds insertion points in sorted array (left)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([2, 3.5, 6]);
      const result = searchsorted(arr, values, 'left');
      expect(result.toArray()).toEqual([1, 3, 5]);
    });

    it('finds insertion points in sorted array (right)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const values = array([2, 3, 6]);
      const result = searchsorted(arr, values, 'right');
      expect(result.toArray()).toEqual([2, 3, 5]);
    });

    it('handles duplicates', () => {
      const arr = array([1, 2, 2, 3]);
      const values = array([2]);
      const resultLeft = searchsorted(arr, values, 'left');
      const resultRight = searchsorted(arr, values, 'right');
      expect(resultLeft.toArray()).toEqual([1]);
      expect(resultRight.toArray()).toEqual([3]);
    });

    it('uses method syntax', () => {
      const arr = array([1, 2, 3]);
      const values = array([2]);
      const result = arr.searchsorted(values);
      expect(result.toArray()).toEqual([1]);
    });
  });

  describe('extract()', () => {
    it('extracts elements where condition is true', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const condition = array([1, 0, 1, 0, 1]);
      const result = extract(condition, arr);
      expect(result.toArray()).toEqual([1, 3, 5]);
    });

    it('returns empty array when no elements match', () => {
      const arr = array([1, 2, 3]);
      const condition = zeros([3]);
      const result = extract(condition, arr);
      expect(result.toArray()).toEqual([]);
    });
  });

  describe('count_nonzero()', () => {
    it('counts non-zero elements', () => {
      const arr = array([1, 0, 2, 0, 3]);
      const result = count_nonzero(arr);
      expect(result).toBe(3);
    });

    it('counts along axis', () => {
      const arr = array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = count_nonzero(arr, 0);
      expect((result as any).toArray()).toEqual([1, 1, 1]);
    });

    it('counts along axis 1', () => {
      const arr = array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = count_nonzero(arr, 1);
      expect((result as any).toArray()).toEqual([2, 1]);
    });

    it('handles all-zero array', () => {
      const arr = zeros([3, 3]);
      const result = count_nonzero(arr);
      expect(result).toBe(0);
    });

    it('handles all-one array', () => {
      const arr = ones([3, 3]);
      const result = count_nonzero(arr);
      expect(result).toBe(9);
    });
  });
});

describe('Dtype Branch Coverage', () => {
  it('sorts float32 array', () => {
    const a = array([3, 1, 2], 'float32');
    expect(sort(a).toArray()).toEqual([1, 2, 3]);
  });

  it('argsort on int32 array', () => {
    const a = array([3, 1, 2], 'int32');
    expect(argsort(a).toArray()).toEqual([1, 2, 0]);
  });

  it('partition on int64 BigInt array', () => {
    const a = array([3n, 1n, 2n, 5n, 4n], 'int64');
    const r = partition(a, 2);
    expect(r.size).toBe(5);
  });
});

describe('Edge Cases', () => {
  it('sort handles empty array', () => {
    const arr = array([]);
    const result = sort(arr);
    expect(result.toArray()).toEqual([]);
  });

  it('argsort handles single element', () => {
    const arr = array([42]);
    const result = argsort(arr);
    expect(result.toArray()).toEqual([0]);
  });

  it('nonzero handles 3D array', () => {
    const arr = zeros([2, 2, 2]);
    arr.set([0, 0, 0], 1);
    arr.set([1, 1, 1], 2);
    const result = nonzero(arr);
    expect(result.length).toBe(3);
    expect(result[0]!.toArray()).toEqual([0, 1]);
    expect(result[1]!.toArray()).toEqual([0, 1]);
    expect(result[2]!.toArray()).toEqual([0, 1]);
  });

  it('searchsorted handles empty array', () => {
    const arr = array([]);
    const values = array([1, 2, 3]);
    const result = searchsorted(arr, values);
    expect(result.toArray()).toEqual([0, 0, 0]);
  });
});
