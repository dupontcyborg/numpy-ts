/**
 * Python NumPy validation tests for comparison operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, array_equiv } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Comparison Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('greater()', () => {
    it('validates greater() with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.greater(3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = (arr > 3).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates greater() between arrays', () => {
      const a = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const b = array([
        [2, 4, 3],
        [8, 3, 5],
      ]);
      const result = a.greater(b);

      const npResult = runNumPy(`
a = np.array([[1, 5, 3], [9, 2, 6]])
b = np.array([[2, 4, 3], [8, 3, 5]])
result = (a > b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates greater() with broadcasting', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([2, 4, 5]);
      const result = a.greater(b);

      const npResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([2, 4, 5])
result = (a > b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('greater_equal()', () => {
    it('validates greater_equal() with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.greater_equal(3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = (arr >= 3).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates greater_equal() between arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.greater_equal(b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
result = (a >= b).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('less()', () => {
    it('validates less() with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less(3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = (arr < 3).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates less() between arrays', () => {
      const a = array([
        [1, 5],
        [3, 2],
      ]);
      const b = array([
        [2, 4],
        [3, 3],
      ]);
      const result = a.less(b);

      const npResult = runNumPy(`
a = np.array([[1, 5], [3, 2]])
b = np.array([[2, 4], [3, 3]])
result = (a < b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates less() with broadcasting', () => {
      const a = array([[1], [2], [3]]);
      const b = array([2]);
      const result = a.less(b);

      const npResult = runNumPy(`
a = np.array([[1], [2], [3]])
b = np.array([2])
result = (a < b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('less_equal()', () => {
    it('validates less_equal() with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less_equal(3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = (arr <= 3).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates less_equal() between arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.less_equal(b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
result = (a <= b).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('equal()', () => {
    it('validates equal() with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.equal(2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 2, 1])
result = (arr == 2).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates equal() between arrays', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([
        [1, 0, 3],
        [0, 5, 0],
      ]);
      const result = a.equal(b);

      const npResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 0, 3], [0, 5, 0]])
result = (a == b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates equal() with broadcasting', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2]);
      const result = a.equal(b);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])
result = (a == b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('not_equal()', () => {
    it('validates not_equal() with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.not_equal(2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 2, 1])
result = (arr != 2).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates not_equal() between arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 0, 3]);
      const result = a.not_equal(b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 0, 3])
result = (a != b).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('complex scenarios', () => {
    it('validates comparison of negative numbers', () => {
      const arr = array([-3, -1, 0, 1, 3]);
      const result = arr.greater(0);

      const npResult = runNumPy(`
arr = np.array([-3, -1, 0, 1, 3])
result = (arr > 0).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates comparison of floating point numbers', () => {
      const arr = array([1.1, 2.2, 3.3]);
      const result = arr.less(2.5);

      const npResult = runNumPy(`
arr = np.array([1.1, 2.2, 3.3])
result = (arr < 2.5).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates complex broadcasting scenario', () => {
      const a = array([[[1, 2]], [[3, 4]]]);
      const b = array([1, 3]);
      const result = a.greater(b);

      const npResult = runNumPy(`
a = np.array([[[1, 2]], [[3, 4]]])
b = np.array([1, 3])
result = (a > b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('isclose()', () => {
    it('validates isclose() with scalar', () => {
      const arr = array([1.0, 2.0, 3.0]);
      const result = arr.isclose(2.00001, 1e-4, 1e-8);

      const npResult = runNumPy(`
arr = np.array([1.0, 2.0, 3.0])
result = np.isclose(arr, 2.00001, rtol=1e-4, atol=1e-8).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isclose() between arrays', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.00001, 2.0, 2.99999]);
      const result = a.isclose(b, 1e-4, 1e-8);

      const npResult = runNumPy(`
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.00001, 2.0, 2.99999])
result = np.isclose(a, b, rtol=1e-4, atol=1e-8).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isclose() with broadcasting', () => {
      const a = array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = array([1.0, 2.0, 3.0]);
      const result = a.isclose(b);

      const npResult = runNumPy(`
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])
result = np.isclose(a, b).astype(np.uint8)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isclose() with floating point precision', () => {
      const a = array([0.1 + 0.2]);
      const b = array([0.3]);
      const result = a.isclose(b);

      const npResult = runNumPy(`
a = np.array([0.1 + 0.2])
b = np.array([0.3])
result = np.isclose(a, b).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isclose() with absolute tolerance', () => {
      const arr = array([1e-9, 2e-9]);
      const result = arr.isclose(0, 1e-5, 1e-8);

      const npResult = runNumPy(`
arr = np.array([1e-9, 2e-9])
result = np.isclose(arr, 0, rtol=1e-5, atol=1e-8).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates isclose() with relative tolerance', () => {
      const arr = array([1.0, 100.0]);
      const result = arr.isclose(1.001, 0.01, 0);

      const npResult = runNumPy(`
arr = np.array([1.0, 100.0])
result = np.isclose(arr, 1.001, rtol=0.01, atol=0).astype(np.uint8)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('allclose()', () => {
    it('validates allclose() returns true for close arrays', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.00001, 2.0, 2.99999]);
      const result = a.allclose(b, 1e-4, 1e-8);

      const npResult = runNumPy(`
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.00001, 2.0, 2.99999])
result = np.allclose(a, b, rtol=1e-4, atol=1e-8)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates allclose() returns false for different arrays', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.1, 3.0]);
      const result = a.allclose(b, 1e-5, 1e-8);

      const npResult = runNumPy(`
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.1, 3.0])
result = np.allclose(a, b, rtol=1e-5, atol=1e-8)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates allclose() with scalar', () => {
      const arr = array([1.0, 1.00001, 0.99999]);
      const result = arr.allclose(1.0, 1e-4, 1e-8);

      const npResult = runNumPy(`
arr = np.array([1.0, 1.00001, 0.99999])
result = np.allclose(arr, 1.0, rtol=1e-4, atol=1e-8)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates allclose() with broadcasting', () => {
      const a = array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
      ]);
      const b = array([1.0, 2.0, 3.0]);
      const result = a.allclose(b);

      const npResult = runNumPy(`
a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
b = np.array([1.0, 2.0, 3.0])
result = np.allclose(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates allclose() with floating point precision', () => {
      const a = array([0.1 + 0.2]);
      const b = array([0.3]);
      const result = a.allclose(b);

      const npResult = runNumPy(`
a = np.array([0.1 + 0.2])
b = np.array([0.3])
result = np.allclose(a, b)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates allclose() with 2D arrays', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const result = a.allclose(b);

      const npResult = runNumPy(`
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.allclose(a, b)
`);

      expect(result).toBe(npResult.value);
    });
  });

  describe('array_equiv()', () => {
    it('matches NumPy array_equiv with broadcasting', () => {
      const a = array([
        [1, 2],
        [1, 2],
      ]);
      const b = array([1, 2]);
      const result = array_equiv(a, b);

      const npResult = runNumPy(`
a = np.array([[1, 2], [1, 2]])
b = np.array([1, 2])
result = np.array_equiv(a, b)
      `);

      expect(result).toBe(npResult.value);
    });

    it('matches NumPy array_equiv for non-equivalent arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2]);
      const result = array_equiv(a, b);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])
result = np.array_equiv(a, b)
      `);

      expect(result).toBe(npResult.value);
    });

    it('matches NumPy array_equiv with scalars', () => {
      const a = array([5, 5, 5]);
      const b = array(5);
      const result = array_equiv(a, b);

      const npResult = runNumPy(`
a = np.array([5, 5, 5])
b = np.array(5)
result = np.array_equiv(a, b)
      `);

      expect(result).toBe(npResult.value);
    });

    it('matches NumPy array_equiv for incompatible shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3, 4]);
      const result = array_equiv(a, b);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([1, 2, 3, 4])
result = np.array_equiv(a, b)
      `);

      expect(result).toBe(npResult.value);
    });
  });
});
