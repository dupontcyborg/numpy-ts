/**
 * Unit tests for vindex — vectorized multi-dimensional indexing
 */

import { describe, it, expect } from 'vitest';
import { array, arange, vindex, Complex } from '../../src';

describe('vindex', () => {
  // ========================================
  // Scalar indexing
  // ========================================
  describe('scalar indices', () => {
    it('removes a dimension on 2D array', () => {
      const a = arange(15).reshape([5, 3]);
      const result = vindex(a, 1);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([3, 4, 5]);
    });

    it('all scalars yields 0-d result', () => {
      const a = arange(12).reshape([3, 4]);
      const result = vindex(a, 1, 2);
      expect(result.shape).toEqual([]);
      expect(result.item()).toBe(6);
    });

    it('negative scalar index', () => {
      const a = arange(15).reshape([5, 3]);
      const result = vindex(a, -1);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([12, 13, 14]);
    });

    it('string scalar index (e.g. "2") treated as number', () => {
      const a = arange(15).reshape([5, 3]);
      const r1 = vindex(a, 2);
      const r2 = vindex(a, '2');
      expect(r2.shape).toEqual(r1.shape);
      expect(r2.toArray()).toEqual(r1.toArray());
    });
  });

  // ========================================
  // Integer array indexing
  // ========================================
  describe('integer array indexing', () => {
    it('1-d index array — subspace first, trailing dims after', () => {
      const a = arange(15).reshape([5, 3]);
      const idx = array([1, 3, 0]);
      const result = vindex(a, idx);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [3, 4, 5],
        [9, 10, 11],
        [0, 1, 2],
      ]);
    });

    it('number[] is cast to index array', () => {
      const a = arange(15).reshape([5, 3]);
      const result = vindex(a, [1, 3, 0]);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [3, 4, 5],
        [9, 10, 11],
        [0, 1, 2],
      ]);
    });

    it('two index arrays: both dims consumed, subspace remains', () => {
      const a = arange(12).reshape([3, 4]);
      const row = array([0, 1, 2]);
      const col = array([1, 2, 3]);
      const result = vindex(a, row, col);
      expect(result.shape).toEqual([3]);
      // a[0,1]=1, a[1,2]=6, a[2,3]=11
      expect(result.toArray()).toEqual([1, 6, 11]);
    });

    it('index arrays broadcast together', () => {
      const a = arange(12).reshape([3, 4]);
      const row = array([[0], [1]]);   // shape (2,1)
      const col = array([[0, 1, 2]]); // shape (1,3)
      const result = vindex(a, row, col);
      expect(result.shape).toEqual([2, 3]);
      // a[0,0]=0, a[0,1]=1, a[0,2]=2
      // a[1,0]=4, a[1,1]=5, a[1,2]=6
      expect(result.toArray()).toEqual([
        [0, 1, 2],
        [4, 5, 6],
      ]);
    });

    it('negative integer array indices', () => {
      const a = arange(12).reshape([3, 4]);
      const result = vindex(a, array([-1]));
      expect(result.shape).toEqual([1, 4]);
      expect(result.toArray()).toEqual([[8, 9, 10, 11]]);
    });

    it('out-of-bounds index throws', () => {
      const a = arange(6).reshape([2, 3]);
      expect(() => vindex(a, array([5]))).toThrow(/out of bounds/);
    });
  });

  // ========================================
  // Slice indexing
  // ========================================
  describe('slice string indices', () => {
    it('":" keeps the dimension', () => {
      const a = arange(15).reshape([5, 3]);
      const result = vindex(a, ':', '1:3');
      expect(result.shape).toEqual([5, 2]);
    });

    it('range slice selects a subset', () => {
      const a = arange(12).reshape([3, 4]);
      const result = vindex(a, '1:3', ':');
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [4, 5, 6, 7],
        [8, 9, 10, 11],
      ]);
    });
  });

  // ========================================
  // Ellipsis
  // ========================================
  describe('ellipsis', () => {
    it('... expands to fill missing dims', () => {
      const a = arange(24).reshape([2, 3, 4]);
      const idx = array([0, 2]);
      // vindex(a, '...', idx): idx indexes last dim, '...' fills first two
      const result = vindex(a, '...', idx);
      expect(result.shape).toEqual([2, 2, 3]);
    });

    it('ellipsis at start', () => {
      const a = arange(24).reshape([2, 3, 4]);
      const idx = array([1, 2]);
      const result = vindex(a, '...', idx);
      // idx covers axis 2; '...' covers axes 0 and 1
      expect(result.shape).toEqual([2, 2, 3]);
    });

    it('more than one ellipsis throws', () => {
      const a = arange(6).reshape([2, 3]);
      expect(() => vindex(a, '...', '...')).toThrow(/ellipsis/);
    });
  });

  // ========================================
  // Mixed advanced + slice (dask ordering)
  // ========================================
  describe('mixed integer array + slice (dask ordering)', () => {
    it('subspace dims precede slice dims', () => {
      const a = arange(30).reshape([5, 6]);
      const idx = array([1, 3]);
      // vindex(a, idx, ':'): row selected by idx, all columns retained
      const result = vindex(a, idx, ':');
      // dask: subspace (2,) first, then slice dim (6,) → shape (2, 6)
      expect(result.shape).toEqual([2, 6]);
      expect(result.toArray()).toEqual([
        [6, 7, 8, 9, 10, 11],
        [18, 19, 20, 21, 22, 23],
      ]);
    });

    it('2D index + slice → subspace shape then slice shape', () => {
      const a = arange(24).reshape([4, 6]);
      const idx = array([[0, 2], [1, 3]]); // shape (2, 2)
      const result = vindex(a, idx, ':');
      // subspace (2,2), slice (6,) → (2, 2, 6)
      expect(result.shape).toEqual([2, 2, 6]);
    });

    it('slice + integer array: slice comes after subspace', () => {
      const a = arange(24).reshape([4, 6]);
      const idx = array([0, 2]);
      const result = vindex(a, '1:3', idx);
      // idx indexes axis 1 (advanced); '1:3' is a range slice for axis 0
      // dask ordering: subspace (2,) first, then slice dim (2,) → (2, 2)
      // result[advIdx, sliceIdx] = a[1+sliceIdx, idx[advIdx]]
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [a.get([1, 0]), a.get([2, 0])],  // idx[0]=0: col 0, rows 1 and 2
        [a.get([1, 2]), a.get([2, 2])],  // idx[1]=2: col 2, rows 1 and 2
      ]);
    });
  });

  // ========================================
  // Edge cases
  // ========================================
  describe('edge cases', () => {
    it('1D array, scalar → 0-d result', () => {
      const a = array([10, 20, 30, 40]);
      const result = vindex(a, 2);
      expect(result.shape).toEqual([]);
      expect(result.item()).toBe(30);
    });

    it('all-slice with ellipsis is identity', () => {
      const a = arange(6).reshape([2, 3]);
      const result = vindex(a, '...', ':');
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual(a.toArray());
    });

    it('preserves dtype', () => {
      const a = array([1.5, 2.5, 3.5], 'float32');
      const result = vindex(a, array([0, 2]));
      expect(result.dtype).toBe('float32');
      expect(result.toArray()).toEqual([1.5, 3.5]);
    });
  });

  // ========================================
  // Complex dtypes
  // ========================================
  describe('complex dtypes', () => {
    it('preserves complex128 dtype', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
      const result = vindex(a, array([2, 0]));
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const vals = result.toArray() as Complex[];
      expect(vals[0]!.re).toBe(5);
      expect(vals[0]!.im).toBe(6);
      expect(vals[1]!.re).toBe(1);
      expect(vals[1]!.im).toBe(2);
    });

    it('scalar index on complex128 2D array removes dimension', () => {
      const a = array([
        [new Complex(1, 0), new Complex(0, 1)],
        [new Complex(2, 0), new Complex(0, 2)],
        [new Complex(3, 0), new Complex(0, 3)],
      ]);
      const result = vindex(a, 1);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const vals = result.toArray() as Complex[];
      expect(vals[0]!.re).toBe(2);
      expect(vals[0]!.im).toBe(0);
      expect(vals[1]!.re).toBe(0);
      expect(vals[1]!.im).toBe(2);
    });

    it('integer array index on complex64 array', () => {
      const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)], 'complex64');
      const result = vindex(a, array([0, 2, 1]));
      expect(result.dtype).toBe('complex64');
      expect(result.shape).toEqual([3]);
      const vals = result.toArray() as Complex[];
      expect(vals[0]!.re).toBeCloseTo(1);
      expect(vals[1]!.re).toBeCloseTo(3);
      expect(vals[2]!.re).toBeCloseTo(2);
    });

    it('slice on complex128 array', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8)]);
      const result = vindex(a, '1:3');
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const vals = result.toArray() as Complex[];
      expect(vals[0]!.re).toBe(3);
      expect(vals[0]!.im).toBe(4);
      expect(vals[1]!.re).toBe(5);
      expect(vals[1]!.im).toBe(6);
    });
  });
});
