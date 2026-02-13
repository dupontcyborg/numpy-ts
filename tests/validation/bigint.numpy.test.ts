/**
 * NumPy validation tests for BigInt (int64/uint64) operations
 * Tests that our BigInt handling matches Python NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: BigInt Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('int64 arithmetic operations', () => {
    it('should match NumPy for large int64 addition', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 9007199254740993n; // Larger than MAX_SAFE_INTEGER
      a.data[1] = 9007199254740994n;
      a.data[2] = 9007199254740995n;

      const jsResult = a.add(1000);
      const pyResult = runNumPy(`
a = np.array([9007199254740993, 9007199254740994, 9007199254740995], dtype=np.int64)
result = a + 1000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 array-to-array addition', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 9007199254740993n;
      a.data[1] = 9007199254740994n;
      a.data[2] = 9007199254740995n;

      const b = zeros([3], 'int64');
      b.data[0] = 1000000000000000n;
      b.data[1] = 2000000000000000n;
      b.data[2] = 3000000000000000n;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([9007199254740993, 9007199254740994, 9007199254740995], dtype=np.int64)
b = np.array([1000000000000000, 2000000000000000, 3000000000000000], dtype=np.int64)
result = a + b
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 subtraction', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 10007199254740993n;
      a.data[1] = 11007199254740994n;
      a.data[2] = 12007199254740995n;

      const jsResult = a.subtract(1000000000000000);
      const pyResult = runNumPy(`
a = np.array([10007199254740993, 11007199254740994, 12007199254740995], dtype=np.int64)
result = a - 1000000000000000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 multiplication', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000n;
      a.data[1] = 2000000000000n;
      a.data[2] = 3000000000000n;

      const jsResult = a.multiply(2);
      const pyResult = runNumPy(`
a = np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.int64)
result = a * 2
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 division (promotes to float64)', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 9007199254740992n;
      a.data[1] = 9007199254740994n;
      a.data[2] = 9007199254740996n;

      const jsResult = a.divide(2);
      const pyResult = runNumPy(`
a = np.array([9007199254740992, 9007199254740994, 9007199254740996], dtype=np.int64)
result = a / 2  # True division (promotes to float64)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('float64'); // NumPy promotes integer division to float64
      expect(pyResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for negative int64 values', () => {
      const a = zeros([3], 'int64');
      a.data[0] = -1000000000000000n;
      a.data[1] = -2000000000000000n;
      a.data[2] = -3000000000000000n;

      const jsResult = a.add(1000);
      const pyResult = runNumPy(`
a = np.array([-1000000000000000, -2000000000000000, -3000000000000000], dtype=np.int64)
result = a + 1000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('uint64 arithmetic operations', () => {
    it('should match NumPy for large uint64 addition', () => {
      const a = zeros([3], 'uint64');
      a.data[0] = 10000000000000000n;
      a.data[1] = 20000000000000000n;
      a.data[2] = 30000000000000000n;

      const jsResult = a.add(1);
      const pyResult = runNumPy(`
a = np.array([10000000000000000, 20000000000000000, 30000000000000000], dtype=np.uint64)
result = a + 1
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('uint64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for uint64 subtraction', () => {
      const a = zeros([3], 'uint64');
      a.data[0] = 10000000000000000n;
      a.data[1] = 20000000000000000n;
      a.data[2] = 30000000000000000n;

      const jsResult = a.subtract(100);
      const pyResult = runNumPy(`
a = np.array([10000000000000000, 20000000000000000, 30000000000000000], dtype=np.uint64)
result = a - 100
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('uint64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('int64 reduction operations', () => {
    it('should match NumPy for int64 sum reduction', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;

      const jsResult = a.sum();
      const pyResult = runNumPy(`
a = np.array([1000000000000000, 2000000000000000, 3000000000000000], dtype=np.int64)
result = a.sum()
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('should match NumPy for int64 sum with axis', () => {
      const a = zeros([2, 3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;
      a.data[3] = 4000000000000000n;
      a.data[4] = 5000000000000000n;
      a.data[5] = 6000000000000000n;

      const jsResult = a.sum(1) as any; // Type assertion needed for axis operations
      const pyResult = runNumPy(`
a = np.array([[1000000000000000, 2000000000000000, 3000000000000000],
              [4000000000000000, 5000000000000000, 6000000000000000]], dtype=np.int64)
result = a.sum(axis=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 max reduction', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 5000000000000000n;
      a.data[2] = 2000000000000000n;

      const jsResult = a.max();
      const pyResult = runNumPy(`
a = np.array([1000000000000000, 5000000000000000, 2000000000000000], dtype=np.int64)
result = a.max()
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('should match NumPy for int64 min reduction', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 5000000000000000n;
      a.data[1] = 1000000000000000n;
      a.data[2] = 3000000000000000n;

      const jsResult = a.min();
      const pyResult = runNumPy(`
a = np.array([5000000000000000, 1000000000000000, 3000000000000000], dtype=np.int64)
result = a.min()
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('should match NumPy for int64 mean reduction', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 9000000000000000n;
      a.data[1] = 9000000000000003n;
      a.data[2] = 9000000000000006n;

      const jsResult = a.mean();
      const pyResult = runNumPy(`
a = np.array([9000000000000000, 9000000000000003, 9000000000000006], dtype=np.int64)
result = a.mean()
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('int64 broadcasting', () => {
    it('should match NumPy for int64 broadcasting with scalar', () => {
      const a = zeros([2, 3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;
      a.data[3] = 4000000000000000n;
      a.data[4] = 5000000000000000n;
      a.data[5] = 6000000000000000n;

      const jsResult = a.add(100);
      const pyResult = runNumPy(`
a = np.array([[1000000000000000, 2000000000000000, 3000000000000000],
              [4000000000000000, 5000000000000000, 6000000000000000]], dtype=np.int64)
result = a + 100
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 broadcasting with 1D array', () => {
      const a = zeros([2, 3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;
      a.data[3] = 4000000000000000n;
      a.data[4] = 5000000000000000n;
      a.data[5] = 6000000000000000n;

      const b = zeros([3], 'int64');
      b.data[0] = 100n;
      b.data[1] = 200n;
      b.data[2] = 300n;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([[1000000000000000, 2000000000000000, 3000000000000000],
              [4000000000000000, 5000000000000000, 6000000000000000]], dtype=np.int64)
b = np.array([100, 200, 300], dtype=np.int64)
result = a + b
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('int64 comparison operations', () => {
    it('should match NumPy for int64 greater than comparison', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;

      const jsResult = a.greater(2000000000000000);
      const pyResult = runNumPy(`
a = np.array([1000000000000000, 2000000000000000, 3000000000000000], dtype=np.int64)
result = a > 2000000000000000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 less than comparison', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;

      const jsResult = a.less(2000000000000000);
      const pyResult = runNumPy(`
a = np.array([1000000000000000, 2000000000000000, 3000000000000000], dtype=np.int64)
result = a < 2000000000000000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 equal comparison', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000000n;
      a.data[1] = 2000000000000000n;
      a.data[2] = 3000000000000000n;

      const jsResult = a.equal(2000000000000000);
      const pyResult = runNumPy(`
a = np.array([1000000000000000, 2000000000000000, 3000000000000000], dtype=np.int64)
result = a == 2000000000000000
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('int64 edge cases', () => {
    it('should match NumPy at MAX_SAFE_INTEGER boundary', () => {
      const a = zeros([3], 'int64');
      a.data[0] = BigInt(Number.MAX_SAFE_INTEGER);
      a.data[1] = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
      a.data[2] = BigInt(Number.MAX_SAFE_INTEGER) + 2n;

      const jsResult = a.add(10);
      const pyResult = runNumPy(`
MAX_SAFE_INTEGER = 9007199254740991
a = np.array([MAX_SAFE_INTEGER, MAX_SAFE_INTEGER + 1, MAX_SAFE_INTEGER + 2], dtype=np.int64)
result = a + 10
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('int64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should match NumPy for int64 division truncation', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 9007199254740993n;
      a.data[1] = 9007199254740994n;
      a.data[2] = 9007199254740995n;

      const jsResult = a.divide(3);
      const pyResult = runNumPy(`
a = np.array([9007199254740993, 9007199254740994, 9007199254740995], dtype=np.int64)
result = a // 3  # Integer division truncates
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
