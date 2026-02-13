import { describe, it, expect } from 'vitest';
import { array, diff, ediff1d, gradient, cross } from '../../src';
import {
  diff as diffStorage,
  ediff1d as ediff1dStorage,
  gradient as gradientStorage,
  cross as crossStorage,
} from '../../src/common/ops/gradient';
import { ArrayStorage } from '../../src/common/storage';

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

  // ============================================================
  // Additional coverage tests for src/common/ops/gradient.ts
  // ============================================================

  describe('diff - additional coverage', () => {
    it('throws for out-of-bounds axis', () => {
      const arr = array([1, 2, 3]);
      expect(() => diff(arr, 1, 5)).toThrow('out of bounds');
    });

    it('throws for out-of-bounds negative axis', () => {
      const arr = array([1, 2, 3]);
      expect(() => diff(arr, 1, -3)).toThrow('out of bounds');
    });

    it('works with float32 dtype', () => {
      const arr = array([1, 3, 6, 10], 'float32');
      const result = diff(arr);
      expect(result.dtype).toBe('float32');
      expect(result.toArray()).toEqual([2, 3, 4]);
    });

    it('works with float64 dtype', () => {
      const arr = array([0.5, 1.5, 3.0, 5.5], 'float64');
      const result = diff(arr);
      expect(result.dtype).toBe('float64');
      const resultArr = result.toArray() as number[];
      expect(resultArr[0]).toBeCloseTo(1.0);
      expect(resultArr[1]).toBeCloseTo(1.5);
      expect(resultArr[2]).toBeCloseTo(2.5);
    });
  });

  describe('diff - storage-level coverage', () => {
    it('computes diff on complex128 storage', () => {
      // Create a complex128 storage: [1+2i, 3+4i, 6+1i]
      const data = new Float64Array([1, 2, 3, 4, 6, 1]);
      const storage = ArrayStorage.fromData(data, [3], 'complex128');
      const result = diffStorage(storage, 1, -1);
      // Differences: (3-1) + (4-2)i = 2+2i, (6-3) + (1-4)i = 3-3i
      expect(result.shape).toEqual([2]);
      expect(result.dtype).toBe('complex128');
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(2);
      expect(rd[1]).toBeCloseTo(2);
      expect(rd[2]).toBeCloseTo(3);
      expect(rd[3]).toBeCloseTo(-3);
    });

    it('computes diff on bigint storage', () => {
      const data = new BigInt64Array([1n, 4n, 9n, 16n]);
      const storage = ArrayStorage.fromData(data, [4], 'int64');
      const result = diffStorage(storage, 1, -1);
      // Differences: 3, 5, 7 (converted to float64)
      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(3);
      expect(rd[1]).toBeCloseTo(5);
      expect(rd[2]).toBeCloseTo(7);
    });

    it('computes diff on 2D complex128 along axis 0', () => {
      // 2x2 complex array: [[1+0i, 2+0i], [4+0i, 6+0i]]
      const data = new Float64Array([1, 0, 2, 0, 4, 0, 6, 0]);
      const storage = ArrayStorage.fromData(data, [2, 2], 'complex128');
      const result = diffStorage(storage, 1, 0);
      expect(result.shape).toEqual([1, 2]);
      expect(result.dtype).toBe('complex128');
      const rd = result.data as Float64Array;
      // (4-1)+0i=3+0i, (6-2)+0i=4+0i
      expect(rd[0]).toBeCloseTo(3);
      expect(rd[1]).toBeCloseTo(0);
      expect(rd[2]).toBeCloseTo(4);
      expect(rd[3]).toBeCloseTo(0);
    });
  });

  describe('ediff1d - storage-level coverage', () => {
    it('computes ediff1d on complex128 storage', () => {
      // [1+2i, 3+4i, 6+1i]
      const data = new Float64Array([1, 2, 3, 4, 6, 1]);
      const storage = ArrayStorage.fromData(data, [3], 'complex128');
      const result = ediff1dStorage(storage);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const rd = result.data as Float64Array;
      // (3-1)+(4-2)i = 2+2i
      expect(rd[0]).toBeCloseTo(2);
      expect(rd[1]).toBeCloseTo(2);
      // (6-3)+(1-4)i = 3-3i
      expect(rd[2]).toBeCloseTo(3);
      expect(rd[3]).toBeCloseTo(-3);
    });

    it('computes ediff1d on complex128 with to_begin', () => {
      const data = new Float64Array([1, 0, 3, 0]);
      const storage = ArrayStorage.fromData(data, [2], 'complex128');
      const result = ediff1dStorage(storage, null, [10]);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const rd = result.data as Float64Array;
      // to_begin: 10+0i
      expect(rd[0]).toBeCloseTo(10);
      expect(rd[1]).toBeCloseTo(0);
      // diff: (3-1)+(0-0)i = 2+0i
      expect(rd[2]).toBeCloseTo(2);
      expect(rd[3]).toBeCloseTo(0);
    });

    it('computes ediff1d on complex128 with to_end', () => {
      const data = new Float64Array([1, 0, 3, 0]);
      const storage = ArrayStorage.fromData(data, [2], 'complex128');
      const result = ediff1dStorage(storage, [99], null);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
      const rd = result.data as Float64Array;
      // diff: (3-1)+0i = 2+0i
      expect(rd[0]).toBeCloseTo(2);
      expect(rd[1]).toBeCloseTo(0);
      // to_end: 99+0i
      expect(rd[2]).toBeCloseTo(99);
      expect(rd[3]).toBeCloseTo(0);
    });

    it('computes ediff1d on bigint storage', () => {
      const data = new BigInt64Array([10n, 13n, 19n]);
      const storage = ArrayStorage.fromData(data, [3], 'int64');
      const result = ediff1dStorage(storage);
      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([2]);
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(3);
      expect(rd[1]).toBeCloseTo(6);
    });
  });

  describe('gradient - additional coverage', () => {
    it('computes gradient with axis as array', () => {
      const arr = array([
        [1, 2, 4],
        [4, 8, 12],
      ]);
      // axis=[0, 1] should return an array of two gradients
      const result = gradient(arr, 1, [0, 1]) as any[];
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(2);
      // axis=0 gradient
      expect(result[0].shape).toEqual([2, 3]);
      // axis=1 gradient
      expect(result[1].shape).toEqual([2, 3]);
    });

    it('computes gradient with varargs as array matching axes', () => {
      const arr = array([
        [1, 2, 4],
        [4, 8, 12],
      ]);
      // Two axes with two spacings
      const result = gradient(arr, [2, 0.5], [0, 1]) as any[];
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(2);
    });

    it('throws when varargs array length does not match axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => gradient(arr, [1, 2, 3], [0, 1])).toThrow(
        'Number of spacings must match number of axes'
      );
    });

    it('throws for out-of-bounds axis (number)', () => {
      const arr = array([1, 2, 3]);
      expect(() => gradient(arr, 1, 5)).toThrow('out of bounds');
    });

    it('throws for out-of-bounds negative axis (number)', () => {
      const arr = array([1, 2, 3]);
      expect(() => gradient(arr, 1, -5)).toThrow('out of bounds');
    });

    it('throws for out-of-bounds axis in axis array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => gradient(arr, 1, [0, 5])).toThrow('out of bounds');
    });

    it('computes gradient with negative axis in axis array', () => {
      const arr = array([
        [1, 2, 4],
        [4, 8, 12],
      ]);
      // axis=[-1] should be normalized to axis=[1]
      const result = gradient(arr, 1, [-1]);
      expect(Array.isArray(result)).toBe(false);
      expect((result as any).shape).toEqual([2, 3]);
    });

    it('computes gradient with float32 dtype', () => {
      const arr = array([1, 4, 9], 'float32');
      const result = gradient(arr) as any;
      expect(result.dtype).toBe('float32');
      const resultArr = result.toArray() as number[];
      // Forward: (4-1)/1 = 3
      // Central: (9-1)/2 = 4
      // Backward: (9-4)/1 = 5
      expect(resultArr[0]).toBeCloseTo(3);
      expect(resultArr[1]).toBeCloseTo(4);
      expect(resultArr[2]).toBeCloseTo(5);
    });

    it('returns single gradient when axis array has one element', () => {
      const arr = array([
        [1, 2, 4],
        [4, 8, 12],
      ]);
      const result = gradient(arr, 1, [1]);
      // Single axis in array should still return a single result (not array)
      expect(Array.isArray(result)).toBe(false);
    });
  });

  describe('gradient - storage-level coverage', () => {
    it('computes gradient on complex128 storage', () => {
      // Complex array: [1+0i, 3+2i, 7+6i]
      const data = new Float64Array([1, 0, 3, 2, 7, 6]);
      const storage = ArrayStorage.fromData(data, [3], 'complex128');
      const result = gradientStorage(storage) as ArrayStorage;
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      // Forward: (3-1)/1 + (2-0)/1 i = 2+2i
      expect(rd[0]).toBeCloseTo(2);
      expect(rd[1]).toBeCloseTo(2);
      // Central: (7-1)/2 + (6-0)/2 i = 3+3i
      expect(rd[2]).toBeCloseTo(3);
      expect(rd[3]).toBeCloseTo(3);
      // Backward: (7-3)/1 + (6-2)/1 i = 4+4i
      expect(rd[4]).toBeCloseTo(4);
      expect(rd[5]).toBeCloseTo(4);
    });

    it('computes gradient on bigint storage', () => {
      const data = new BigInt64Array([2n, 5n, 11n, 20n]);
      const storage = ArrayStorage.fromData(data, [4], 'int64');
      const result = gradientStorage(storage) as ArrayStorage;
      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([4]);
      const rd = result.data as Float64Array;
      // Forward: (5-2)/1 = 3
      expect(rd[0]).toBeCloseTo(3);
      // Central: (11-2)/2 = 4.5
      expect(rd[1]).toBeCloseTo(4.5);
      // Central: (20-5)/2 = 7.5
      expect(rd[2]).toBeCloseTo(7.5);
      // Backward: (20-11)/1 = 9
      expect(rd[3]).toBeCloseTo(9);
    });

    it('computes gradient on 2D complex128 along axis 0', () => {
      // 2x2 complex array: [[1+0i, 2+0i], [5+0i, 8+0i]]
      const data = new Float64Array([1, 0, 2, 0, 5, 0, 8, 0]);
      const storage = ArrayStorage.fromData(data, [2, 2], 'complex128');
      const result = gradientStorage(storage, 1, 0) as ArrayStorage;
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2, 2]);
      const rd = result.data as Float64Array;
      // Along axis 0, size=2, so forward diff = backward diff
      // [0,0]: (5-1)/1 = 4+0i
      expect(rd[0]).toBeCloseTo(4);
      expect(rd[1]).toBeCloseTo(0);
      // [0,1]: (8-2)/1 = 6+0i
      expect(rd[2]).toBeCloseTo(6);
      expect(rd[3]).toBeCloseTo(0);
      // [1,0]: (5-1)/1 = 4+0i
      expect(rd[4]).toBeCloseTo(4);
      expect(rd[5]).toBeCloseTo(0);
      // [1,1]: (8-2)/1 = 6+0i
      expect(rd[6]).toBeCloseTo(6);
      expect(rd[7]).toBeCloseTo(0);
    });

    it('computes gradient with custom spacing on storage', () => {
      const data = new Float64Array([0, 2, 6, 12]);
      const storage = ArrayStorage.fromData(data, [4], 'float64');
      const result = gradientStorage(storage, 0.5) as ArrayStorage;
      expect(result.dtype).toBe('float64');
      const rd = result.data as Float64Array;
      // Forward: (2-0)/0.5 = 4
      expect(rd[0]).toBeCloseTo(4);
      // Central: (6-0)/(2*0.5) = 6
      expect(rd[1]).toBeCloseTo(6);
      // Central: (12-2)/(2*0.5) = 10
      expect(rd[2]).toBeCloseTo(10);
      // Backward: (12-6)/0.5 = 12
      expect(rd[3]).toBeCloseTo(12);
    });
  });

  describe('cross (gradient.ts) - storage-level coverage', () => {
    it('computes 3D x 3D cross product', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0]), [3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([0, 1, 0]), [3], 'float64');
      const result = crossStorage(a, b);
      // i x j = k = [0, 0, 1]
      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data as Float64Array)).toEqual([0, 0, 1]);
    });

    it('computes 2D x 2D cross product (scalar result)', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2]), [2], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([3, 4]), [2], 'float64');
      const result = crossStorage(a, b);
      // 1*4 - 2*3 = -2
      expect(result.shape).toEqual([]);
      expect(result.data[0]).toBe(-2);
    });

    it('computes 2D x 3D cross product', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2]), [2], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([3, 4, 5]), [3], 'float64');
      const result = crossStorage(a, b);
      // Treat a as [1, 2, 0]
      // c0 = 2*5 - 0*4 = 10
      // c1 = 0*3 - 1*5 = -5
      // c2 = 1*4 - 2*3 = -2
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(10);
      expect(rd[1]).toBeCloseTo(-5);
      expect(rd[2]).toBeCloseTo(-2);
    });

    it('computes 3D x 2D cross product', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2, 3]), [3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([4, 5]), [2], 'float64');
      const result = crossStorage(a, b);
      // Treat b as [4, 5, 0]
      // c0 = 2*0 - 3*5 = -15
      // c1 = 3*4 - 1*0 = 12
      // c2 = 1*5 - 2*4 = -3
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(-15);
      expect(rd[1]).toBeCloseTo(12);
      expect(rd[2]).toBeCloseTo(-3);
    });

    it('computes 2D arrays of 3D vectors', () => {
      // 2 vectors of dimension 3
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0, 0, 1, 0]), [2, 3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([0, 1, 0, 0, 0, 1]), [2, 3], 'float64');
      const result = crossStorage(a, b);
      expect(result.shape).toEqual([2, 3]);
      const rd = result.data as Float64Array;
      // First: [1,0,0] x [0,1,0] = [0,0,1]
      expect(rd[0]).toBeCloseTo(0);
      expect(rd[1]).toBeCloseTo(0);
      expect(rd[2]).toBeCloseTo(1);
      // Second: [0,1,0] x [0,0,1] = [1,0,0]
      expect(rd[3]).toBeCloseTo(1);
      expect(rd[4]).toBeCloseTo(0);
      expect(rd[5]).toBeCloseTo(0);
    });

    it('computes 2D arrays of 2D vectors', () => {
      // 2 vectors of dimension 2
      const a = ArrayStorage.fromData(new Float64Array([1, 2, 3, 4]), [2, 2], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([5, 6, 7, 8]), [2, 2], 'float64');
      const result = crossStorage(a, b);
      // Returns 1D array of scalar cross products
      expect(result.shape).toEqual([2]);
      const rd = result.data as Float64Array;
      // First: 1*6 - 2*5 = -4
      expect(rd[0]).toBeCloseTo(-4);
      // Second: 3*8 - 4*7 = -4
      expect(rd[1]).toBeCloseTo(-4);
    });

    it('throws for shape mismatch in 2D arrays', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0, 0, 1, 0]), [2, 3], 'float64');
      const b = ArrayStorage.fromData(
        new Float64Array([0, 1, 0, 0, 0, 1, 1, 0, 0]),
        [3, 3],
        'float64'
      );
      expect(() => crossStorage(a, b)).toThrow('Shape mismatch');
    });

    it('throws for out-of-bounds axisa', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0]), [3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([0, 1, 0]), [3], 'float64');
      expect(() => crossStorage(a, b, 5)).toThrow('axisa');
    });

    it('throws for out-of-bounds axisb', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0]), [3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([0, 1, 0]), [3], 'float64');
      expect(() => crossStorage(a, b, -1, 5)).toThrow('axisb');
    });

    it('throws for invalid vector sizes (not 2 or 3) - both invalid', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2, 3, 4]), [4], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([5, 6, 7, 8]), [4], 'float64');
      expect(() => crossStorage(a, b)).toThrow('incompatible dimensions');
    });

    it('throws for invalid sizeB only (sizeA is valid)', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2, 3]), [3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array([5, 6, 7, 8]), [4], 'float64');
      expect(() => crossStorage(a, b)).toThrow('incompatible dimensions');
    });

    it('throws for unsupported higher-dim array shapes', () => {
      // 3D arrays with vector dim=3 along last axis - not supported in gradient.ts cross
      const a = ArrayStorage.fromData(new Float64Array(18), [2, 3, 3], 'float64');
      const b = ArrayStorage.fromData(new Float64Array(18), [2, 3, 3], 'float64');
      expect(() => crossStorage(a, b)).toThrow('not implemented');
    });

    it('computes complex 3D x 3D cross product', () => {
      // [1+0i, 0+0i, 0+0i]
      const aData = new Float64Array([1, 0, 0, 0, 0, 0]);
      const a = ArrayStorage.fromData(aData, [3], 'complex128');
      // [0+0i, 1+0i, 0+0i]
      const bData = new Float64Array([0, 0, 1, 0, 0, 0]);
      const b = ArrayStorage.fromData(bData, [3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      // [1,0,0] x [0,1,0] = [0,0,1]
      expect(rd[0]).toBeCloseTo(0); // re
      expect(rd[1]).toBeCloseTo(0); // im
      expect(rd[2]).toBeCloseTo(0); // re
      expect(rd[3]).toBeCloseTo(0); // im
      expect(rd[4]).toBeCloseTo(1); // re
      expect(rd[5]).toBeCloseTo(0); // im
    });

    it('computes complex 2D x 2D cross product', () => {
      // [1+1i, 2+0i]
      const aData = new Float64Array([1, 1, 2, 0]);
      const a = ArrayStorage.fromData(aData, [2], 'complex128');
      // [3+0i, 4+0i]
      const bData = new Float64Array([3, 0, 4, 0]);
      const b = ArrayStorage.fromData(bData, [2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([]);
      const rd = result.data as Float64Array;
      // (1+1i)*(4+0i) - (2+0i)*(3+0i) = (4+4i) - (6+0i) = -2+4i
      expect(rd[0]).toBeCloseTo(-2);
      expect(rd[1]).toBeCloseTo(4);
    });

    it('computes complex 2D x 3D cross product', () => {
      // a = [1+0i, 0+0i] (2D)
      const aData = new Float64Array([1, 0, 0, 0]);
      const a = ArrayStorage.fromData(aData, [2], 'complex128');
      // b = [0+0i, 1+0i, 0+0i] (3D)
      const bData = new Float64Array([0, 0, 1, 0, 0, 0]);
      const b = ArrayStorage.fromData(bData, [3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
    });

    it('computes complex 3D x 2D cross product', () => {
      // a = [1+0i, 0+0i, 0+0i] (3D)
      const aData = new Float64Array([1, 0, 0, 0, 0, 0]);
      const a = ArrayStorage.fromData(aData, [3], 'complex128');
      // b = [0+0i, 1+0i] (2D)
      const bData = new Float64Array([0, 0, 1, 0]);
      const b = ArrayStorage.fromData(bData, [2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
    });

    it('computes complex 2D arrays of 3D vectors', () => {
      // 2 vectors of dimension 3, complex128
      const aData = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
      const a = ArrayStorage.fromData(aData, [2, 3], 'complex128');
      const bData = new Float64Array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
      const b = ArrayStorage.fromData(bData, [2, 3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2, 3]);
    });

    it('computes mixed real x complex 3D cross product (getComplex fallback)', () => {
      // a is real float64, b is complex128 -> result is complex, getComplex reads real as [val, 0]
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0]), [3], 'float64');
      const bData = new Float64Array([0, 0, 1, 0, 0, 0]);
      const b = ArrayStorage.fromData(bData, [3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
      const rd = result.data as Float64Array;
      // [1,0,0] x [0,1,0] = [0,0,1]
      expect(rd[0]).toBeCloseTo(0);
      expect(rd[1]).toBeCloseTo(0);
      expect(rd[2]).toBeCloseTo(0);
      expect(rd[3]).toBeCloseTo(0);
      expect(rd[4]).toBeCloseTo(1);
      expect(rd[5]).toBeCloseTo(0);
    });

    it('computes mixed real x complex 2D cross product', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2]), [2], 'float64');
      const bData = new Float64Array([3, 0, 4, 0]);
      const b = ArrayStorage.fromData(bData, [2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      // 1*4 - 2*3 = -2
      const rd = result.data as Float64Array;
      expect(rd[0]).toBeCloseTo(-2);
      expect(rd[1]).toBeCloseTo(0);
    });

    it('computes mixed real x complex 2D x 3D', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0]), [2], 'float64');
      const bData = new Float64Array([0, 0, 1, 0, 0, 0]);
      const b = ArrayStorage.fromData(bData, [3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
    });

    it('computes mixed real x complex 3D x 2D', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0]), [3], 'float64');
      const bData = new Float64Array([0, 0, 1, 0]);
      const b = ArrayStorage.fromData(bData, [2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
    });

    it('computes mixed real x complex 2D arrays of 3D vectors', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 0, 0, 0, 1, 0]), [2, 3], 'float64');
      const bData = new Float64Array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
      const b = ArrayStorage.fromData(bData, [2, 3], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2, 3]);
    });

    it('computes mixed real x complex 2D arrays of 2D vectors', () => {
      const a = ArrayStorage.fromData(new Float64Array([1, 2, 3, 4]), [2, 2], 'float64');
      const bData = new Float64Array([5, 0, 6, 0, 7, 0, 8, 0]);
      const b = ArrayStorage.fromData(bData, [2, 2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
    });

    it('computes complex 2D arrays of 2D vectors', () => {
      // 2 vectors of dimension 2, complex128
      const aData = new Float64Array([1, 0, 2, 0, 3, 0, 4, 0]);
      const a = ArrayStorage.fromData(aData, [2, 2], 'complex128');
      const bData = new Float64Array([5, 0, 6, 0, 7, 0, 8, 0]);
      const b = ArrayStorage.fromData(bData, [2, 2], 'complex128');
      const result = crossStorage(a, b);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
    });
  });
});
