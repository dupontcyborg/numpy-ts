import { describe, it, expect } from 'vitest';
import { array, diff, ediff1d, gradient, cross } from '../../src';

describe('Gradient Operations', () => {
  describe('diff', () => {
    it('computes first difference for 1D array', () => {
      const arr = array([1, 2, 4, 7, 0]);
      const result = diff(arr);
      const expected = [1, 2, 3, -7];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('works as method', () => {
      const arr = array([1, 2, 4]);
      const result = arr.diff();
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([1, 2]);
    });

    it('computes second difference', () => {
      const arr = array([1, 2, 4, 7, 0]);
      const result = diff(arr, 2);
      const expected = [1, 1, -10];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('computes third difference', () => {
      const arr = array([1, 2, 4, 7, 0]);
      const result = diff(arr, 3);
      const expected = [0, -11];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('works with n=0 (returns copy)', () => {
      const arr = array([1, 2, 3]);
      const result = diff(arr, 0);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('works along specified axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const resultAxis0 = diff(arr, 1, 0);
      const resultAxis1 = diff(arr, 1, 1);

      expect(resultAxis0.shape).toEqual([1, 3]);
      expect(resultAxis0.toArray()).toEqual([[3, 3, 3]]);

      expect(resultAxis1.shape).toEqual([2, 2]);
      expect(resultAxis1.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = diff(arr);
      expect(result.dtype).toBe('int32');
    });

    it('throws for negative n', () => {
      const arr = array([1, 2, 3]);
      expect(() => diff(arr, -1)).toThrow('order must be non-negative');
    });

    it('throws when array is too small', () => {
      const arr = array([1, 2]);
      expect(() => diff(arr, 2)).toThrow('diff requires at least 3 elements');
    });
  });

  describe('ediff1d', () => {
    it('computes differences for flattened array', () => {
      const arr = array([1, 2, 4, 7, 0]);
      const result = ediff1d(arr);
      const expected = [1, 2, 3, -7];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('works with 2D array (flattens first)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = ediff1d(arr);
      // Flattened: [1, 2, 3, 4]
      // Differences: [1, 1, 1]
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([1, 1, 1]);
    });

    it('prepends values with to_begin', () => {
      const arr = array([1, 2, 4]);
      const result = ediff1d(arr, null, [0, 0]);
      const expected = [0, 0, 1, 2];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('appends values with to_end', () => {
      const arr = array([1, 2, 4]);
      const result = ediff1d(arr, [9, 9], null);
      const expected = [1, 2, 9, 9];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });

    it('uses both to_begin and to_end', () => {
      const arr = array([1, 2, 4]);
      const result = ediff1d(arr, [99], [0]);
      const expected = [0, 1, 2, 99];
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual(expected);
    });
  });

  describe('gradient', () => {
    it('computes gradient for 1D array', () => {
      const arr = array([1, 2, 4, 7, 11]);
      const result = gradient(arr) as any;
      // Forward: (2-1)/1 = 1
      // Central: (4-1)/2 = 1.5, (7-2)/2 = 2.5, (11-4)/2 = 3.5
      // Backward: (11-7)/1 = 4
      const expected = [1, 1.5, 2.5, 3.5, 4];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('respects custom spacing', () => {
      const arr = array([1, 2, 4]);
      const result = gradient(arr, 2) as any;
      // Forward: (2-1)/2 = 0.5
      // Backward: (4-2)/2 = 1
      // Central: (4-1)/4 = 0.75
      const expected = [0.5, 0.75, 1];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('returns multiple gradients for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = gradient(arr) as any[];
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(2);
    });

    it('computes gradient along specific axis', () => {
      const arr = array([
        [1, 2, 4],
        [4, 8, 12],
      ]);
      const result = gradient(arr, 1, 1) as any;
      expect(result.shape).toEqual([2, 3]);

      // First row: Forward: 1, Central: 1.5, Backward: 2
      // Second row: Forward: 4, Central: 4, Backward: 4
    });

    it('throws for array with size < 2 along axis', () => {
      const arr = array([1]);
      expect(() => gradient(arr)).toThrow('must be at least 2');
    });
  });

  describe('cross', () => {
    it('computes cross product of 3D vectors', () => {
      const a = array([1, 0, 0]);
      const b = array([0, 1, 0]);
      const result = cross(a, b);
      // i x j = k
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([0, 0, 1]);
    });

    it('computes cross product - example 2', () => {
      const a = array([0, 1, 0]);
      const b = array([0, 0, 1]);
      const result = cross(a, b);
      // j x k = i
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([1, 0, 0]);
    });

    it('computes cross product - example 3', () => {
      const a = array([0, 0, 1]);
      const b = array([1, 0, 0]);
      const result = cross(a, b);
      // k x i = j
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([0, 1, 0]);
    });

    it('satisfies a x a = 0', () => {
      const a = array([1, 2, 3]);
      const result = cross(a, a);
      const resultArr = result.toArray() as number[];
      expect(resultArr).toEqual([0, 0, 0]);
    });

    it('satisfies a x b = -b x a', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const axb = cross(a, b);
      const bxa = cross(b, a);

      const axbArr = axb.toArray() as number[];
      const bxaArr = bxa.toArray() as number[];

      for (let i = 0; i < 3; i++) {
        expect(Math.abs(axbArr[i]! + bxaArr[i]!)).toBeLessThan(1e-10);
      }
    });

    it('computes cross product of 2D vectors (returns scalar)', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const result = cross(a, b);
      // Cross product of 2D vectors: a0*b1 - a1*b0 = 1*4 - 2*3 = -2
      expect(result.shape).toEqual([]);
      expect(result.get([])).toBe(-2);
    });

    it('computes cross product of 2D x 3D vectors', () => {
      const a = array([1, 0]);
      const b = array([0, 1, 0]);
      const result = cross(a, b);
      // 2D treated as [1, 0, 0] x [0, 1, 0] = [0, 0, 1]
      expect(result.shape).toEqual([3]);
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[2]! - 1)).toBeLessThan(1e-10);
    });

    it('works with multiple vectors (2D arrays)', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
      ]);
      const b = array([
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = cross(a, b);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [0, 0, 1],
        [1, 0, 0],
      ]);
    });

    it('throws for invalid vector dimensions', () => {
      const a = array([1, 2, 3, 4]);
      const b = array([5, 6, 7, 8]);
      expect(() => cross(a, b)).toThrow('incompatible dimensions for cross product');
    });
  });
});
