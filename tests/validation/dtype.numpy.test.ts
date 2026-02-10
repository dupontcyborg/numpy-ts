/**
 * Python NumPy validation tests for dtype support
 *
 * Validates that our dtype implementation matches NumPy's behavior
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros, ones, array, arange, linspace, eye } from '../../src';
import { checkNumPyAvailable, runNumPy, arraysClose } from './numpy-oracle';

describe('NumPy Validation: DType Support', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('Array creation with dtype', () => {
    it('zeros with float32 dtype', () => {
      const jsArr = zeros([2, 3], 'float32');
      const pyResult = runNumPy("result = np.zeros((2, 3), dtype='float32')");

      expect(jsArr.dtype).toBe('float32');
      expect(jsArr.shape).toEqual(pyResult.shape);
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('zeros with int32 dtype', () => {
      const jsArr = zeros([2, 3], 'int32');
      const pyResult = runNumPy("result = np.zeros((2, 3), dtype='int32')");

      expect(jsArr.dtype).toBe('int32');
      expect(jsArr.shape).toEqual(pyResult.shape);
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('zeros with uint8 dtype', () => {
      const jsArr = zeros([2, 3], 'uint8');
      const pyResult = runNumPy("result = np.zeros((2, 3), dtype='uint8')");

      expect(jsArr.dtype).toBe('uint8');
      expect(jsArr.shape).toEqual(pyResult.shape);
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('ones with float64 dtype', () => {
      const jsArr = ones([2, 2], 'float64');
      const pyResult = runNumPy("result = np.ones((2, 2), dtype='float64')");

      expect(jsArr.dtype).toBe('float64');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('ones with int32 dtype', () => {
      const jsArr = ones([2, 2], 'int32');
      const pyResult = runNumPy("result = np.ones((2, 2), dtype='int32')");

      expect(jsArr.dtype).toBe('int32');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('array with explicit float32 dtype', () => {
      const jsArr = array([1, 2, 3], 'float32');
      const pyResult = runNumPy("result = np.array([1, 2, 3], dtype='float32')");

      expect(jsArr.dtype).toBe('float32');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('arange with int32 dtype', () => {
      const jsArr = arange(0, 5, 1, 'int32');
      const pyResult = runNumPy("result = np.arange(0, 5, 1, dtype='int32')");

      expect(jsArr.dtype).toBe('int32');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('linspace with float32 dtype', () => {
      const jsArr = linspace(0, 1, 5, 'float32');
      const pyResult = runNumPy("result = np.linspace(0, 1, 5, dtype='float32')");

      expect(jsArr.dtype).toBe('float32');
      expect(arraysClose(jsArr.toArray(), pyResult.value)).toBe(true);
    });

    it('eye with int32 dtype', () => {
      const jsArr = eye(3, undefined, 0, 'int32');
      const pyResult = runNumPy("result = np.eye(3, dtype='int32')");

      expect(jsArr.dtype).toBe('int32');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });
  });

  describe('astype() conversions', () => {
    it('float64 to float32', () => {
      const jsArr = array([1.5, 2.5, 3.5]);
      const jsConverted = jsArr.astype('float32');
      const pyResult = runNumPy(
        "result = np.array([1.5, 2.5, 3.5], dtype='float64').astype('float32')"
      );

      expect(jsConverted.dtype).toBe('float32');
      expect(jsConverted.toArray()).toEqual(pyResult.value);
    });

    it('float to int32 (truncates)', () => {
      const jsArr = array([1.7, 2.3, 3.9]);
      const jsConverted = jsArr.astype('int32');
      const pyResult = runNumPy("result = np.array([1.7, 2.3, 3.9]).astype('int32')");

      expect(jsConverted.dtype).toBe('int32');
      expect(jsConverted.toArray()).toEqual(pyResult.value);
    });

    it('int32 to float64', () => {
      const jsArr = array([1, 2, 3], 'int32');
      const jsConverted = jsArr.astype('float64');
      const pyResult = runNumPy("result = np.array([1, 2, 3], dtype='int32').astype('float64')");

      expect(jsConverted.dtype).toBe('float64');
      expect(jsConverted.toArray()).toEqual(pyResult.value);
    });

    it('to bool dtype', () => {
      const jsArr = array([0, 1, 2, 0, -1]);
      const jsConverted = jsArr.astype('bool');
      const pyResult = runNumPy("result = np.array([0, 1, 2, 0, -1]).astype('bool')");

      expect(jsConverted.dtype).toBe('bool');
      // NumPy bool values come back as boolean, but we store as uint8
      const expectedValues = pyResult.value.map((v: boolean) => (v ? 1 : 0));
      expect(jsConverted.toArray()).toEqual(expectedValues);
    });

    it('bool to int32', () => {
      const jsArr = array([0, 1, 1, 0], 'bool');
      const jsConverted = jsArr.astype('int32');
      const pyResult = runNumPy("result = np.array([0, 1, 1, 0], dtype='bool').astype('int32')");

      expect(jsConverted.dtype).toBe('int32');
      expect(jsConverted.toArray()).toEqual(pyResult.value);
    });

    it('returns same array with copy=false when dtype matches', () => {
      const jsArr = array([1, 2, 3]);
      const jsSame = jsArr.astype('float64', false);

      // In JavaScript, check they're the same object
      expect(jsSame).toBe(jsArr);
    });

    it('creates copy with copy=true when dtype matches', () => {
      const jsArr = array([1, 2, 3]);
      const jsCopy = jsArr.astype('float64', true);

      // Different objects
      expect(jsCopy).not.toBe(jsArr);
      expect(jsCopy.toArray()).toEqual(jsArr.toArray());
    });
  });

  describe('Dtype property', () => {
    it('returns float64 dtype for default arrays', () => {
      const jsArr = array([1, 2, 3]);

      // Note: NumPy defaults to int64 for integers, but we default to float64
      // (following @stdlib's behavior for JavaScript number compatibility)
      expect(jsArr.dtype).toBe('float64');
    });

    it('returns correct dtype for explicitly typed arrays', () => {
      expect(zeros([2, 3], 'float32').dtype).toBe('float32');
      expect(zeros([2, 3], 'int32').dtype).toBe('int32');
      expect(zeros([2, 3], 'uint8').dtype).toBe('uint8');
      expect(zeros([2, 3], 'bool').dtype).toBe('bool');
    });
  });

  describe('Arithmetic with different dtypes', () => {
    it('addition with int32 dtype', () => {
      const jsArr = ones([2, 2], 'int32');
      const jsResult = jsArr.add(5);
      const pyResult = runNumPy("result = np.ones((2, 2), dtype='int32') + 5");

      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('multiplication with float32 dtype', () => {
      const jsArr = array([1, 2, 3], 'float32');
      const jsResult = jsArr.multiply(2.5);
      const pyResult = runNumPy("result = np.array([1, 2, 3], dtype='float32') * 2.5");

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Edge cases', () => {
    it('handles empty arrays with dtype', () => {
      const jsArr = zeros([0], 'float32');
      const pyResult = runNumPy("result = np.zeros(0, dtype='float32')");

      expect(jsArr.dtype).toBe('float32');
      expect(jsArr.size).toBe(0);
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('handles 1D arrays with dtype', () => {
      const jsArr = ones([5], 'int32');
      const pyResult = runNumPy("result = np.ones(5, dtype='int32')");

      expect(jsArr.dtype).toBe('int32');
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('handles multidimensional arrays with dtype', () => {
      const jsArr = zeros([2, 3, 4], 'float32');
      const pyResult = runNumPy("result = np.zeros((2, 3, 4), dtype='float32')");

      expect(jsArr.dtype).toBe('float32');
      expect(jsArr.shape).toEqual(pyResult.shape);
      expect(jsArr.toArray()).toEqual(pyResult.value);
    });
  });
});
