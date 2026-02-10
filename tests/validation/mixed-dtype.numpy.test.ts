/**
 * NumPy validation tests for mixed-dtype operations
 * Tests that mixed-dtype arithmetic matches Python NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Mixed DType Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('Integer + Integer promotions', () => {
    it('should promote int8 + uint8 → int16', () => {
      const a = zeros([3], 'int8');
      a.data[0] = 10;
      a.data[1] = 20;
      a.data[2] = 30;

      const b = zeros([3], 'uint8');
      b.data[0] = 5;
      b.data[1] = 10;
      b.data[2] = 15;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([10, 20, 30], dtype=np.int8)
b = np.array([5, 10, 15], dtype=np.uint8)
result = a + b
      `);

      expect(jsResult.dtype).toBe('int16');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should promote int32 + uint32 → int64', () => {
      const a = zeros([3], 'int32');
      a.data[0] = 1000;
      a.data[1] = 2000;
      a.data[2] = 3000;

      const b = zeros([3], 'uint32');
      b.data[0] = 500;
      b.data[1] = 1000;
      b.data[2] = 1500;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([1000, 2000, 3000], dtype=np.int32)
b = np.array([500, 1000, 1500], dtype=np.uint32)
result = a + b
      `);

      expect(jsResult.dtype).toBe('int64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should promote int64 + uint64 → float64', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000n;
      a.data[1] = 2000000000000n;
      a.data[2] = 3000000000000n;

      const b = zeros([3], 'uint64');
      b.data[0] = 500000000000n;
      b.data[1] = 1000000000000n;
      b.data[2] = 1500000000000n;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.int64)
b = np.array([500000000000, 1000000000000, 1500000000000], dtype=np.uint64)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Integer + Float promotions', () => {
    it('should promote int32 + float32 → float64', () => {
      const a = zeros([3], 'int32');
      a.data[0] = 10;
      a.data[1] = 20;
      a.data[2] = 30;

      const b = zeros([3], 'float32');
      b.data[0] = 1.5;
      b.data[1] = 2.5;
      b.data[2] = 3.5;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([10, 20, 30], dtype=np.int32)
b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should promote int64 + float64 → float64', () => {
      const a = zeros([3], 'int64');
      a.data[0] = 1000000000000n;
      a.data[1] = 2000000000000n;
      a.data[2] = 3000000000000n;

      const b = zeros([3], 'float64');
      b.data[0] = 1.5;
      b.data[1] = 2.5;
      b.data[2] = 3.5;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.int64)
b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Float + Float promotions', () => {
    it('should keep float32 + float32 → float32', () => {
      const a = zeros([3], 'float32');
      a.data[0] = 1.5;
      a.data[1] = 2.5;
      a.data[2] = 3.5;

      const b = zeros([3], 'float32');
      b.data[0] = 0.5;
      b.data[1] = 1.5;
      b.data[2] = 2.5;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
b = np.array([0.5, 1.5, 2.5], dtype=np.float32)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float32');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should promote float32 + float64 → float64', () => {
      const a = zeros([3], 'float32');
      a.data[0] = 1.5;
      a.data[1] = 2.5;
      a.data[2] = 3.5;

      const b = zeros([3], 'float64');
      b.data[0] = 0.5;
      b.data[1] = 1.5;
      b.data[2] = 2.5;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
b = np.array([0.5, 1.5, 2.5], dtype=np.float64)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Boolean promotions', () => {
    it('should promote bool + int32 → int32', () => {
      const a = zeros([3], 'bool');
      a.data[0] = 1; // true
      a.data[1] = 0; // false
      a.data[2] = 1; // true

      const b = zeros([3], 'int32');
      b.data[0] = 10;
      b.data[1] = 20;
      b.data[2] = 30;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([True, False, True], dtype=np.bool_)
b = np.array([10, 20, 30], dtype=np.int32)
result = a + b
      `);

      expect(jsResult.dtype).toBe('int32');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('should promote bool + float64 → float64', () => {
      const a = zeros([3], 'bool');
      a.data[0] = 1; // true
      a.data[1] = 0; // false
      a.data[2] = 1; // true

      const b = zeros([3], 'float64');
      b.data[0] = 1.5;
      b.data[1] = 2.5;
      b.data[2] = 3.5;

      const jsResult = a.add(b);
      const pyResult = runNumPy(`
a = np.array([True, False, True], dtype=np.bool_)
b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
result = a + b
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
