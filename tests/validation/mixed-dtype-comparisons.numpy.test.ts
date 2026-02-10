/**
 * NumPy validation tests for mixed-dtype comparison operations
 * Tests that mixed-dtype comparisons match Python NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Mixed DType Comparisons', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('Integer + Integer comparisons', () => {
    it('should compare int8 > uint8 correctly', () => {
      const a = zeros([3], 'int8');
      a.data[0] = 10;
      a.data[1] = 20;
      a.data[2] = 30;

      const b = zeros([3], 'uint8');
      b.data[0] = 15;
      b.data[1] = 10;
      b.data[2] = 25;

      const jsResult = a.greater(b);
      const pyResult = runNumPy(`
a = np.array([10, 20, 30], dtype=np.int8)
b = np.array([15, 10, 25], dtype=np.uint8)
result = a > b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should compare int64 == uint64 correctly', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000n;
      a.data[1] = 2000000000000n;
      a.data[2] = 3000000000000n;

      const b = zeros([3], 'uint64');
      b.data[0] = 1000000000000n;
      b.data[1] = 1500000000000n;
      b.data[2] = 3000000000000n;

      const jsResult = a.equal(b);
      const pyResult = runNumPy(`
a = np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.int64)
b = np.array([1000000000000, 1500000000000, 3000000000000], dtype=np.uint64)
result = a == b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Integer + Float comparisons', () => {
    it('should compare int32 < float64 correctly', () => {
      const a = zeros([3], 'int32');
      a.data[0] = 10;
      a.data[1] = 20;
      a.data[2] = 30;

      const b = zeros([3], 'float64');
      b.data[0] = 15.5;
      b.data[1] = 19.5;
      b.data[2] = 30.5;

      const jsResult = a.less(b);
      const pyResult = runNumPy(`
a = np.array([10, 20, 30], dtype=np.int32)
b = np.array([15.5, 19.5, 30.5], dtype=np.float64)
result = a < b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should compare int64 >= float32 correctly', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000n;
      a.data[1] = 2000n;
      a.data[2] = 3000n;

      const b = zeros([3], 'float32');
      b.data[0] = 999.5;
      b.data[1] = 2000.0;
      b.data[2] = 3000.5;

      const jsResult = a.greater_equal(b);
      const pyResult = runNumPy(`
a = np.array([1000, 2000, 3000], dtype=np.int64)
b = np.array([999.5, 2000.0, 3000.5], dtype=np.float32)
result = a >= b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Float + Float comparisons', () => {
    it('should compare float32 != float64 correctly', () => {
      const a = zeros([3], 'float32');
      a.data[0] = 1.5;
      a.data[1] = 2.5;
      a.data[2] = 3.5;

      const b = zeros([3], 'float64');
      b.data[0] = 1.5;
      b.data[1] = 2.6;
      b.data[2] = 3.5;

      const jsResult = a.not_equal(b);
      const pyResult = runNumPy(`
a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
b = np.array([1.5, 2.6, 3.5], dtype=np.float64)
result = a != b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Boolean comparisons', () => {
    it('should compare bool <= int32 correctly', () => {
      const a = zeros([3], 'bool');
      a.data[0] = 1; // true
      a.data[1] = 0; // false
      a.data[2] = 1; // true

      const b = zeros([3], 'int32');
      b.data[0] = 0;
      b.data[1] = 1;
      b.data[2] = 1;

      const jsResult = a.less_equal(b);
      const pyResult = runNumPy(`
a = np.array([True, False, True], dtype=np.bool_)
b = np.array([0, 1, 1], dtype=np.int32)
result = a <= b
      `);

      expect(jsResult.dtype).toBe('bool');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
