/**
 * NumPy validation tests for dtype edge cases
 * Tests overflow, underflow, special values, and precision limits
 * Validates against actual NumPy behavior
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, ones } from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: DType Edge Cases', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('Integer overflow behavior', () => {
    it('int8 overflow wraps like NumPy', () => {
      const arr = array([127], 'int8');
      const result = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([127], dtype=np.int8)
result = arr + 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // int8: 127 + 1 = -128 (wraps around)
      expect(result.get([0])).toBe(-128);
    });

    it('uint8 overflow wraps like NumPy', () => {
      const arr = array([255], 'uint8');
      const result = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([255], dtype=np.uint8)
result = arr + 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // uint8: 255 + 1 = 0 (wraps around)
      expect(result.get([0])).toBe(0);
    });

    it('int16 overflow wraps like NumPy', () => {
      const arr = array([32767], 'int16');
      const result = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([32767], dtype=np.int16)
result = arr + 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // int16: 32767 + 1 = -32768
      expect(result.get([0])).toBe(-32768);
    });

    it('int32 overflow wraps like NumPy', () => {
      const arr = array([2147483647], 'int32'); // INT32_MAX
      const result = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([2147483647], dtype=np.int32)
result = arr + 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // int32: 2^31-1 + 1 = -2^31
      expect(result.get([0])).toBe(-2147483648);
    });

    it('uint32 overflow wraps like NumPy', () => {
      const arr = array([4294967295], 'uint32'); // UINT32_MAX
      const result = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([4294967295], dtype=np.uint32)
result = arr + 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // uint32: 2^32-1 + 1 = 0
      expect(result.get([0])).toBe(0);
    });
  });

  describe('Integer underflow behavior', () => {
    it('int8 underflow wraps like NumPy', () => {
      const arr = array([-128], 'int8');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
arr = np.array([-128], dtype=np.int8)
result = arr - 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // int8: -128 - 1 = 127 (wraps around)
      expect(result.get([0])).toBe(127);
    });

    it('uint8 underflow wraps like NumPy', () => {
      const arr = array([0], 'uint8');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
arr = np.array([0], dtype=np.uint8)
result = arr - 1
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      // uint8: 0 - 1 = 255 (wraps around)
      expect(result.get([0])).toBe(255);
    });
  });

  describe('Float special values', () => {
    it('handles positive infinity like NumPy', () => {
      const arr = array([Infinity], 'float64');
      const doubled = arr.multiply(2);

      const npResult = runNumPy(`
arr = np.array([np.inf], dtype=np.float64)
result = arr * 2
      `);

      expect(doubled.get([0])).toBe(Infinity);
      expect(doubled.get([0])).toBe(npResult.value[0]);
    });

    it('handles negative infinity like NumPy', () => {
      const arr = array([-Infinity], 'float64');
      const doubled = arr.multiply(2);

      const npResult = runNumPy(`
arr = np.array([-np.inf], dtype=np.float64)
result = arr * 2
      `);

      expect(doubled.get([0])).toBe(-Infinity);
      expect(doubled.get([0])).toBe(npResult.value[0]);
    });

    it('handles NaN like NumPy', () => {
      const arr = array([NaN], 'float64');
      const added = arr.add(1);

      const npResult = runNumPy(`
arr = np.array([np.nan], dtype=np.float64)
result = arr + 1
      `);

      expect(Number.isNaN(added.get([0]))).toBe(true);
      expect(Number.isNaN(npResult.value[0])).toBe(true);
    });

    it('inf + inf = inf like NumPy', () => {
      const arr = array([Infinity], 'float64');
      const result = arr.add(Infinity);

      const npResult = runNumPy(`
arr = np.array([np.inf], dtype=np.float64)
result = arr + np.inf
      `);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('inf - inf = NaN like NumPy', () => {
      const arr = array([Infinity], 'float64');
      const result = arr.subtract(Infinity);

      const npResult = runNumPy(`
arr = np.array([np.inf], dtype=np.float64)
result = arr - np.inf
      `);

      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(npResult.value[0])).toBe(true);
    });

    it('0 * inf = NaN like NumPy', () => {
      const arr = array([0], 'float64');
      const result = arr.multiply(Infinity);

      const npResult = runNumPy(`
arr = np.array([0], dtype=np.float64)
result = arr * np.inf
      `);

      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(npResult.value[0])).toBe(true);
    });

    it('inf / inf = NaN like NumPy', () => {
      const arr = array([Infinity], 'float64');
      const result = arr.divide(Infinity);

      const npResult = runNumPy(`
arr = np.array([np.inf], dtype=np.float64)
result = arr / np.inf
      `);

      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(npResult.value[0])).toBe(true);
    });
  });

  describe('Division by zero', () => {
    it('float / 0 = inf like NumPy', () => {
      const arr = array([1.0], 'float64');
      const result = arr.divide(0);

      const npResult = runNumPy(`
import warnings
arr = np.array([1.0], dtype=np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr / 0
      `);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative float / 0 = -inf like NumPy', () => {
      const arr = array([-1.0], 'float64');
      const result = arr.divide(0);

      const npResult = runNumPy(`
import warnings
arr = np.array([-1.0], dtype=np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr / 0
      `);

      expect(result.get([0])).toBe(-Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('0.0 / 0.0 = NaN like NumPy', () => {
      const arr = array([0.0], 'float64');
      const result = arr.divide(0);

      const npResult = runNumPy(`
import warnings
arr = np.array([0.0], dtype=np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr / 0
      `);

      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(npResult.value[0])).toBe(true);
    });
  });

  describe('Float underflow', () => {
    it('float32 underflows to zero like NumPy', () => {
      // float32 min positive: ~1.2e-38
      const arr = array([1e-40], 'float32');
      const result = arr.multiply(1e-40);

      const npResult = runNumPy(`
arr = np.array([1e-40], dtype=np.float32)
result = arr * 1e-40
      `);

      expect(result.get([0])).toBe(0);
      expect(npResult.value[0]).toBe(0);
    });

    it('float64 preserves very small values', () => {
      // float64 min positive: ~2.2e-308
      const arr = array([1e-200], 'float64');
      const result = arr.multiply(1e-100);

      const npResult = runNumPy(`
arr = np.array([1e-200], dtype=np.float64)
result = arr * 1e-100
      `);

      // Should be 1e-300, not underflow to 0
      expect(result.get([0])).not.toBe(0);
      expect(npResult.value[0]).not.toBe(0);
      expect(result.get([0])).toBeCloseTo(npResult.value[0], 320);
    });
  });

  describe('Float precision differences', () => {
    it('float32 has less precision than float64', () => {
      // float32: ~7 decimal digits
      // float64: ~15 decimal digits
      const value = 1.0 + 1e-8;

      const f32 = array([value], 'float32');
      const f64 = array([value], 'float64');

      const npF32 = runNumPy(`
result = np.array([${value}], dtype=np.float32)
      `);

      const npF64 = runNumPy(`
result = np.array([${value}], dtype=np.float64)
      `);

      // float32 loses precision
      expect(f32.get([0])).toBeCloseTo(npF32.value[0], 6);

      // float64 preserves it
      expect(f64.get([0])).toBeCloseTo(npF64.value[0], 14);
    });

    it('float32 rounding errors accumulate faster', () => {
      const arr32 = ones([1000], 'float32');
      const sum32 = arr32.sum();

      const arr64 = ones([1000], 'float64');
      const sum64 = arr64.sum();

      const npSum32 = runNumPy(`
arr = np.ones(1000, dtype=np.float32)
result = np.array([arr.sum()])
      `);

      const npSum64 = runNumPy(`
arr = np.ones(1000, dtype=np.float64)
result = np.array([arr.sum()])
      `);

      // Both should get close to 1000, but with different precision
      expect(sum32).toBeCloseTo(npSum32.value[0], 5);
      expect(sum64).toBeCloseTo(npSum64.value[0], 14);
      expect(sum64).toBeCloseTo(1000, 14);
    });
  });

  describe('Type limits', () => {
    it('int8 range: -128 to 127', () => {
      const min = array([-128], 'int8');
      const max = array([127], 'int8');

      const npMin = runNumPy(`
result = np.array([-128], dtype=np.int8)
      `);

      const npMax = runNumPy(`
result = np.array([127], dtype=np.int8)
      `);

      expect(min.get([0])).toBe(npMin.value[0]);
      expect(max.get([0])).toBe(npMax.value[0]);
      expect(min.get([0])).toBe(-128);
      expect(max.get([0])).toBe(127);
    });

    it('uint8 range: 0 to 255', () => {
      const min = array([0], 'uint8');
      const max = array([255], 'uint8');

      const npMin = runNumPy(`
result = np.array([0], dtype=np.uint8)
      `);

      const npMax = runNumPy(`
result = np.array([255], dtype=np.uint8)
      `);

      expect(min.get([0])).toBe(npMin.value[0]);
      expect(max.get([0])).toBe(npMax.value[0]);
      expect(min.get([0])).toBe(0);
      expect(max.get([0])).toBe(255);
    });

    it('int32 range matches NumPy', () => {
      const min = array([-2147483648], 'int32');
      const max = array([2147483647], 'int32');

      const npMin = runNumPy(`
result = np.array([-2147483648], dtype=np.int32)
      `);

      const npMax = runNumPy(`
result = np.array([2147483647], dtype=np.int32)
      `);

      expect(min.get([0])).toBe(npMin.value[0]);
      expect(max.get([0])).toBe(npMax.value[0]);
    });
  });

  describe('Negative zero', () => {
    it('preserves negative zero in float64', () => {
      const arr = array([-0.0], 'float64');

      // Check sign bit - JavaScript preserves -0.0
      const value = arr.get([0]) as number;
      expect(1 / value).toBe(-Infinity);
      expect(Object.is(value, -0)).toBe(true);
    });

    it('negative zero equals positive zero', () => {
      const negZero = array([-0.0], 'float64');
      const posZero = array([0.0], 'float64');
      const result = negZero.equal(posZero);

      const npResult = runNumPy(`
neg_zero = np.array([-0.0], dtype=np.float64)
pos_zero = np.array([0.0], dtype=np.float64)
result = neg_zero == pos_zero
      `);

      // -0.0 == 0.0 is true in both JavaScript and NumPy
      expect(result.get([0])).toBe(1);
      expect(npResult.value[0]).toBe(true);
    });
  });
});
