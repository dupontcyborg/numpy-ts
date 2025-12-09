/**
 * Unit tests for set operations
 */

import { describe, it, expect } from 'vitest';
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

describe('Set Operations', () => {
  describe('unique()', () => {
    it('finds unique elements in 1D array', () => {
      const arr = array([1, 2, 2, 3, 3, 3, 4]);
      const result = unique(arr);
      expect((result as any).toArray()).toEqual([1, 2, 3, 4]);
    });

    it('finds unique elements in 2D array (flattened)', () => {
      const arr = array([
        [1, 2, 1],
        [2, 3, 3],
      ]);
      const result = unique(arr);
      expect((result as any).toArray()).toEqual([1, 2, 3]);
    });

    it('returns indices with returnIndex=true', () => {
      const arr = array([4, 2, 2, 1, 3]);
      const result = unique(arr, true);
      expect((result as any).values.toArray()).toEqual([1, 2, 3, 4]);
      expect((result as any).indices.toArray()).toEqual([3, 1, 4, 0]);
    });

    it('returns inverse with returnInverse=true', () => {
      const arr = array([1, 2, 1, 3, 2]);
      const result = unique(arr, false, true);
      expect((result as any).values.toArray()).toEqual([1, 2, 3]);
      expect((result as any).inverse.toArray()).toEqual([0, 1, 0, 2, 1]);
    });

    it('returns counts with returnCounts=true', () => {
      const arr = array([1, 2, 2, 3, 3, 3]);
      const result = unique(arr, false, false, true);
      expect((result as any).values.toArray()).toEqual([1, 2, 3]);
      expect((result as any).counts.toArray()).toEqual([1, 2, 3]);
    });

    it('returns all outputs when all flags are true', () => {
      const arr = array([3, 1, 2, 1]);
      const result = unique(arr, true, true, true);
      expect((result as any).values.toArray()).toEqual([1, 2, 3]);
      expect((result as any).indices).toBeDefined();
      expect((result as any).inverse).toBeDefined();
      expect((result as any).counts).toBeDefined();
    });
  });

  describe('in1d()', () => {
    it('tests membership in second array', () => {
      const ar1 = array([1, 2, 3, 4, 5]);
      const ar2 = array([2, 4]);
      const result = in1d(ar1, ar2);
      expect(result.toArray()).toEqual([0, 1, 0, 1, 0]);
    });

    it('handles empty test array', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([]);
      const result = in1d(ar1, ar2);
      expect(result.toArray()).toEqual([0, 0, 0]);
    });

    it('handles no matches', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([4, 5, 6]);
      const result = in1d(ar1, ar2);
      expect(result.toArray()).toEqual([0, 0, 0]);
    });
  });

  describe('intersect1d()', () => {
    it('finds intersection of two arrays', () => {
      const ar1 = array([1, 2, 3, 4]);
      const ar2 = array([3, 4, 5, 6]);
      const result = intersect1d(ar1, ar2);
      expect(result.toArray()).toEqual([3, 4]);
    });

    it('handles no intersection', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([3, 4]);
      const result = intersect1d(ar1, ar2);
      expect(result.toArray()).toEqual([]);
    });

    it('handles duplicate elements', () => {
      const ar1 = array([1, 1, 2, 2, 3]);
      const ar2 = array([2, 2, 3, 3, 4]);
      const result = intersect1d(ar1, ar2);
      expect(result.toArray()).toEqual([2, 3]);
    });

    it('returns sorted result', () => {
      const ar1 = array([4, 2, 1]);
      const ar2 = array([1, 4, 3]);
      const result = intersect1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 4]);
    });
  });

  describe('isin()', () => {
    it('tests membership element-wise', () => {
      const element = array([1, 2, 3, 4, 5]);
      const testElements = array([2, 4]);
      const result = isin(element, testElements);
      expect(result.toArray()).toEqual([0, 1, 0, 1, 0]);
    });

    it('preserves shape of input', () => {
      const element = array([
        [1, 2],
        [3, 4],
      ]);
      const testElements = array([2, 3]);
      const result = isin(element, testElements);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 1],
        [1, 0],
      ]);
    });

    it('handles all elements in test set', () => {
      const element = array([1, 2, 3]);
      const testElements = array([1, 2, 3, 4, 5]);
      const result = isin(element, testElements);
      expect(result.toArray()).toEqual([1, 1, 1]);
    });
  });

  describe('setdiff1d()', () => {
    it('finds set difference', () => {
      const ar1 = array([1, 2, 3, 4, 5]);
      const ar2 = array([3, 4]);
      const result = setdiff1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 5]);
    });

    it('handles empty difference', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([1, 2, 3]);
      const result = setdiff1d(ar1, ar2);
      expect(result.toArray()).toEqual([]);
    });

    it('handles duplicate elements', () => {
      const ar1 = array([1, 1, 2, 2, 3]);
      const ar2 = array([2]);
      const result = setdiff1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 3]);
    });
  });

  describe('setxor1d()', () => {
    it('finds symmetric difference', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([2, 3, 4]);
      const result = setxor1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 4]);
    });

    it('handles identical arrays', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([1, 2, 3]);
      const result = setxor1d(ar1, ar2);
      expect(result.toArray()).toEqual([]);
    });

    it('handles disjoint arrays', () => {
      const ar1 = array([1, 2]);
      const ar2 = array([3, 4]);
      const result = setxor1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('returns sorted result', () => {
      const ar1 = array([5, 3, 1]);
      const ar2 = array([4, 2]);
      const result = setxor1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('union1d()', () => {
    it('finds union of two arrays', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([3, 4, 5]);
      const result = union1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('handles identical arrays', () => {
      const ar1 = array([1, 2, 3]);
      const ar2 = array([1, 2, 3]);
      const result = union1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles duplicate elements', () => {
      const ar1 = array([1, 1, 2, 2]);
      const ar2 = array([2, 2, 3, 3]);
      const result = union1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('returns sorted result', () => {
      const ar1 = array([5, 3, 1]);
      const ar2 = array([4, 2]);
      const result = union1d(ar1, ar2);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });
  });
});

describe('Edge Cases', () => {
  it('unique handles empty array', () => {
    const arr = array([]);
    const result = unique(arr);
    expect((result as any).toArray()).toEqual([]);
  });

  it('unique handles single element', () => {
    const arr = array([42]);
    const result = unique(arr);
    expect((result as any).toArray()).toEqual([42]);
  });

  it('intersect1d handles empty arrays', () => {
    const ar1 = array([]);
    const ar2 = array([1, 2, 3]);
    const result = intersect1d(ar1, ar2);
    expect(result.toArray()).toEqual([]);
  });

  it('union1d handles empty arrays', () => {
    const ar1 = array([]);
    const ar2 = array([1, 2, 3]);
    const result = union1d(ar1, ar2);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('setdiff1d handles empty ar2', () => {
    const ar1 = array([1, 2, 3]);
    const ar2 = array([]);
    const result = setdiff1d(ar1, ar2);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('isin handles 3D array', () => {
    const element = array([
      [
        [1, 2],
        [3, 4],
      ],
    ]);
    const testElements = array([2, 4]);
    const result = isin(element, testElements);
    expect(result.shape).toEqual([1, 2, 2]);
    expect(result.toArray()).toEqual([
      [
        [0, 1],
        [0, 1],
      ],
    ]);
  });
});
