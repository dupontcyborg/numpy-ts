/**
 * Python NumPy validation tests for rounding operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, around, ceil, fix, floor, rint, round, trunc } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Rounding Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('around()', () => {
    it('validates around() with default decimals', () => {
      const arr = array([1.4, 1.5, 1.6, 2.5, -1.5]);
      const result = around(arr);

      const npResult = runNumPy(`
arr = np.array([1.4, 1.5, 1.6, 2.5, -1.5])
result = np.around(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates around() with decimals=2', () => {
      const arr = array([1.234, 2.567, 3.891]);
      const result = around(arr, 2);

      const npResult = runNumPy(`
arr = np.array([1.234, 2.567, 3.891])
result = np.around(arr, decimals=2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates around() with negative decimals', () => {
      const arr = array([123, 456, 789]);
      const result = around(arr, -1);

      const npResult = runNumPy(`
arr = np.array([123, 456, 789])
result = np.around(arr, decimals=-1)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates around() on 2D array', () => {
      const arr = array([
        [1.4, 2.5],
        [3.6, 4.1],
      ]);
      const result = around(arr);

      const npResult = runNumPy(`
arr = np.array([[1.4, 2.5], [3.6, 4.1]])
result = np.around(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('ceil()', () => {
    it('validates ceil() on positive and negative values', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9]);
      const result = ceil(arr);

      const npResult = runNumPy(`
arr = np.array([1.1, 1.9, -1.1, -1.9])
result = np.ceil(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates ceil() on 2D array', () => {
      const arr = array([
        [1.1, 2.5],
        [3.9, 4.0],
      ]);
      const result = ceil(arr);

      const npResult = runNumPy(`
arr = np.array([[1.1, 2.5], [3.9, 4.0]])
result = np.ceil(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('fix()', () => {
    it('validates fix() rounds towards zero', () => {
      const arr = array([1.9, -1.9, 2.5, -2.5]);
      const result = fix(arr);

      const npResult = runNumPy(`
arr = np.array([1.9, -1.9, 2.5, -2.5])
result = np.fix(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates fix() on 2D array', () => {
      const arr = array([
        [1.7, -1.7],
        [2.3, -2.3],
      ]);
      const result = fix(arr);

      const npResult = runNumPy(`
arr = np.array([[1.7, -1.7], [2.3, -2.3]])
result = np.fix(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('floor()', () => {
    it('validates floor() on positive and negative values', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9]);
      const result = floor(arr);

      const npResult = runNumPy(`
arr = np.array([1.1, 1.9, -1.1, -1.9])
result = np.floor(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates floor() on 2D array', () => {
      const arr = array([
        [1.1, 2.5],
        [3.9, 4.0],
      ]);
      const result = floor(arr);

      const npResult = runNumPy(`
arr = np.array([[1.1, 2.5], [3.9, 4.0]])
result = np.floor(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('rint()', () => {
    it('validates rint() rounds to nearest integer', () => {
      const arr = array([1.4, 1.5, 1.6, 2.5, -1.5]);
      const result = rint(arr);

      const npResult = runNumPy(`
arr = np.array([1.4, 1.5, 1.6, 2.5, -1.5])
result = np.rint(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates rint() on 2D array', () => {
      const arr = array([
        [1.4, 2.5],
        [3.6, 4.1],
      ]);
      const result = rint(arr);

      const npResult = runNumPy(`
arr = np.array([[1.4, 2.5], [3.6, 4.1]])
result = np.rint(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('round()', () => {
    it('validates round() with default decimals', () => {
      const arr = array([1.4, 1.5, 1.6]);
      const result = round(arr);

      const npResult = runNumPy(`
arr = np.array([1.4, 1.5, 1.6])
result = np.round(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates round() with decimals=2', () => {
      const arr = array([1.234, 2.567]);
      const result = round(arr, 2);

      const npResult = runNumPy(`
arr = np.array([1.234, 2.567])
result = np.round(arr, decimals=2)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('trunc()', () => {
    it('validates trunc() truncates to integer part', () => {
      const arr = array([1.7, -1.7, 2.9, -2.9]);
      const result = trunc(arr);

      const npResult = runNumPy(`
arr = np.array([1.7, -1.7, 2.9, -2.9])
result = np.trunc(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates trunc() on 2D array', () => {
      const arr = array([
        [1.9, -1.9],
        [2.1, -2.1],
      ]);
      const result = trunc(arr);

      const npResult = runNumPy(`
arr = np.array([[1.9, -1.9], [2.1, -2.1]])
result = np.trunc(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('validates around() with integer input', () => {
      const arr = array([1, 2, 3]);
      const result = around(arr);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.around(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates floor() with very small values', () => {
      const arr = array([0.001, -0.001]);
      const result = floor(arr);

      const npResult = runNumPy(`
arr = np.array([0.001, -0.001])
result = np.floor(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates ceil() with very small values', () => {
      const arr = array([0.001, -0.001]);
      const result = ceil(arr);

      const npResult = runNumPy(`
arr = np.array([0.001, -0.001])
result = np.ceil(arr)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
