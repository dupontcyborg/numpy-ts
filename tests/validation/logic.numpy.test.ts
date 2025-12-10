/**
 * Python NumPy validation tests for logic operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Logic Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('logical_and()', () => {
    it('validates logical_and() with scalar', () => {
      const arr = array([0, 1, 2, 3, 0]);
      const result = arr.logical_and(1);

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 0])
result = np.logical_and(arr, 1).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical_and() between arrays', () => {
      const a = array([0, 1, 0, 1]);
      const b = array([0, 0, 1, 1]);
      const result = a.logical_and(b);

      const npResult = runNumPy(`
a = np.array([0, 1, 0, 1])
b = np.array([0, 0, 1, 1])
result = np.logical_and(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical_and() with broadcasting', () => {
      const a = array([
        [1, 0, 1],
        [0, 1, 0],
      ]);
      const b = array([1, 1, 0]);
      const result = a.logical_and(b);

      const npResult = runNumPy(`
a = np.array([[1, 0, 1], [0, 1, 0]])
b = np.array([1, 1, 0])
result = np.logical_and(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('logical_or()', () => {
    it('validates logical_or() with scalar', () => {
      const arr = array([0, 1, 0, 3, 0]);
      const result = arr.logical_or(0);

      const npResult = runNumPy(`
arr = np.array([0, 1, 0, 3, 0])
result = np.logical_or(arr, 0).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical_or() between arrays', () => {
      const a = array([0, 1, 0, 1]);
      const b = array([0, 0, 1, 1]);
      const result = a.logical_or(b);

      const npResult = runNumPy(`
a = np.array([0, 1, 0, 1])
b = np.array([0, 0, 1, 1])
result = np.logical_or(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('logical_not()', () => {
    it('validates logical_not() on 1D array', () => {
      const arr = array([0, 1, 2, 0, -1]);
      const result = arr.logical_not();

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 0, -1])
result = np.logical_not(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical_not() on 2D array', () => {
      const arr = array([
        [0, 1],
        [2, 0],
      ]);
      const result = arr.logical_not();

      const npResult = runNumPy(`
arr = np.array([[0, 1], [2, 0]])
result = np.logical_not(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('logical_xor()', () => {
    it('validates logical_xor() with scalar', () => {
      const arr = array([0, 1, 2, 0]);
      const result = arr.logical_xor(1);

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 0])
result = np.logical_xor(arr, 1).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical_xor() between arrays', () => {
      const a = array([0, 1, 0, 1]);
      const b = array([0, 0, 1, 1]);
      const result = a.logical_xor(b);

      const npResult = runNumPy(`
a = np.array([0, 1, 0, 1])
b = np.array([0, 0, 1, 1])
result = np.logical_xor(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('isfinite()', () => {
    it('validates isfinite() with various values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isfinite();

      const npResult = runNumPy(`
arr = np.array([1, 2, np.inf, -np.inf, np.nan, 0])
result = np.isfinite(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isfinite() with 2D array', () => {
      const arr = array([
        [1.0, Infinity],
        [NaN, 2.0],
      ]);
      const result = arr.isfinite();

      const npResult = runNumPy(`
arr = np.array([[1.0, np.inf], [np.nan, 2.0]])
result = np.isfinite(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('isinf()', () => {
    it('validates isinf() with various values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isinf();

      const npResult = runNumPy(`
arr = np.array([1, 2, np.inf, -np.inf, np.nan, 0])
result = np.isinf(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('isnan()', () => {
    it('validates isnan() with various values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isnan();

      const npResult = runNumPy(`
arr = np.array([1, 2, np.inf, -np.inf, np.nan, 0])
result = np.isnan(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isnan() with multiple NaN values', () => {
      const arr = array([NaN, 1, NaN, 2, NaN]);
      const result = arr.isnan();

      const npResult = runNumPy(`
arr = np.array([np.nan, 1, np.nan, 2, np.nan])
result = np.isnan(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('copysign()', () => {
    it('validates copysign() with positive sign', () => {
      const a = array([-1, -2, 3, -4]);
      const result = a.copysign(1);

      const npResult = runNumPy(`
a = np.array([-1, -2, 3, -4], dtype=np.float64)
result = np.copysign(a, 1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates copysign() with negative sign', () => {
      const a = array([1, 2, -3, 4]);
      const result = a.copysign(-1);

      const npResult = runNumPy(`
a = np.array([1, 2, -3, 4], dtype=np.float64)
result = np.copysign(a, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates copysign() between arrays', () => {
      const a = array([1, -2, 3, -4]);
      const b = array([-1, 1, -1, 1]);
      const result = a.copysign(b);

      const npResult = runNumPy(`
a = np.array([1, -2, 3, -4], dtype=np.float64)
b = np.array([-1, 1, -1, 1], dtype=np.float64)
result = np.copysign(a, b)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('signbit()', () => {
    it('validates signbit() with various values', () => {
      const arr = array([1, -1, 0, 2, -3]);
      const result = arr.signbit();

      const npResult = runNumPy(`
arr = np.array([1, -1, 0, 2, -3], dtype=np.float64)
result = np.signbit(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates signbit() with negative zero', () => {
      const arr = array([-0.0, 0.0]);
      const result = arr.signbit();

      const npResult = runNumPy(`
arr = np.array([-0.0, 0.0], dtype=np.float64)
result = np.signbit(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates signbit() with infinity', () => {
      const arr = array([Infinity, -Infinity]);
      const result = arr.signbit();

      const npResult = runNumPy(`
arr = np.array([np.inf, -np.inf], dtype=np.float64)
result = np.signbit(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nextafter()', () => {
    it('validates nextafter() towards positive infinity', () => {
      const a = array([0, 1]);
      const result = a.nextafter(Infinity);

      const npResult = runNumPy(`
a = np.array([0, 1], dtype=np.float64)
result = np.nextafter(a, np.inf)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates nextafter() towards negative infinity', () => {
      const a = array([0, 1]);
      const result = a.nextafter(-Infinity);

      const npResult = runNumPy(`
a = np.array([0, 1], dtype=np.float64)
result = np.nextafter(a, -np.inf)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates nextafter() between arrays', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([0.0, 3.0, 2.0]);
      const result = a.nextafter(b);

      const npResult = runNumPy(`
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([0.0, 3.0, 2.0], dtype=np.float64)
result = np.nextafter(a, b)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates nextafter() when values are equal', () => {
      const a = array([1.0, 2.0, 3.0]);
      const result = a.nextafter(a);

      const npResult = runNumPy(`
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
result = np.nextafter(a, a)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('spacing()', () => {
    it('validates spacing() at various magnitudes', () => {
      const arr = array([1, 10, 100, 1000]);
      const result = arr.spacing();

      const npResult = runNumPy(`
arr = np.array([1, 10, 100, 1000], dtype=np.float64)
result = np.spacing(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates spacing() at zero', () => {
      const arr = array([0]);
      const result = arr.spacing();

      const npResult = runNumPy(`
arr = np.array([0], dtype=np.float64)
result = np.spacing(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates spacing() with negative values', () => {
      const arr = array([-1, -10, -100]);
      const result = arr.spacing();

      const npResult = runNumPy(`
arr = np.array([-1, -10, -100], dtype=np.float64)
result = np.spacing(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('complex scenarios', () => {
    it('validates chained logical operations', () => {
      const arr = array([0, 1, 2, 3, 4, 5]);

      // arr > 1 AND arr < 4
      const gt1 = arr.greater(1);
      const lt4 = arr.less(4);
      const result = gt1.logical_and(lt4);

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5])
result = np.logical_and(arr > 1, arr < 4).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates logical operations with floating point', () => {
      const arr = array([0.0, 0.1, 1.0, -0.1, 0.0]);
      const result = arr.logical_not();

      const npResult = runNumPy(`
arr = np.array([0.0, 0.1, 1.0, -0.1, 0.0])
result = np.logical_not(arr).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates broadcasting in 3D', () => {
      const a = array([[[1, 0]], [[0, 1]]]);
      const b = array([1, 0]);
      const result = a.logical_and(b);

      const npResult = runNumPy(`
a = np.array([[[1, 0]], [[0, 1]]])
b = np.array([1, 0])
result = np.logical_and(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
