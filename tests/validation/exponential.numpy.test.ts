/**
 * Python NumPy validation tests for exponential operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  exp,
  exp2,
  expm1,
  log,
  log2,
  log10,
  log1p,
  logaddexp,
  logaddexp2,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Exponential Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n'
      );
    }
  });

  describe('exp', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = exp(array([-2, -1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.exp(np.array([-2, -1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = exp(
        array([
          [-1, 0],
          [1, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.exp(np.array([[-1, 0], [1, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = exp(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.exp(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('exp2', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = exp2(array([0, 1, 2, 3]));
      const pyResult = runNumPy(`
result = np.exp2(np.array([0, 1, 2, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative exponents', () => {
      const jsResult = exp2(array([-1, -2, -3]));
      const pyResult = runNumPy(`
result = np.exp2(np.array([-1, -2, -3]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('expm1', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = expm1(array([-1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.expm1(np.array([-1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for small values (high accuracy)', () => {
      const jsResult = expm1(array([1e-10, 1e-15]));
      const pyResult = runNumPy(`
result = np.expm1(np.array([1e-10, 1e-15]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = log(array([1, 2, 10, 100]));
      const pyResult = runNumPy(`
result = np.log(np.array([1, 2, 10, 100]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = log(array([1, 2, 3], 'int32'));
      const pyResult = runNumPy(`
result = np.log(np.array([1, 2, 3], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log2', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = log2(array([1, 2, 4, 8, 16]));
      const pyResult = runNumPy(`
result = np.log2(np.array([1, 2, 4, 8, 16]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log10', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = log10(array([1, 10, 100, 1000]));
      const pyResult = runNumPy(`
result = np.log10(np.array([1, 10, 100, 1000]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log1p', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = log1p(array([0, 1, 2, 10]));
      const pyResult = runNumPy(`
result = np.log1p(np.array([0, 1, 2, 10]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for small values', () => {
      const jsResult = log1p(array([1e-10, 1e-15]));
      const pyResult = runNumPy(`
result = np.log1p(np.array([1e-10, 1e-15]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('logaddexp', () => {
    it('matches NumPy for equal values', () => {
      const jsResult = logaddexp(array([1, 2, 3]), array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.logaddexp(np.array([1, 2, 3]), np.array([1, 2, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for different values', () => {
      const jsResult = logaddexp(array([0, 1, 2]), array([3, 4, 5]));
      const pyResult = runNumPy(`
result = np.logaddexp(np.array([0, 1, 2]), np.array([3, 4, 5]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for large values (numerical stability)', () => {
      const jsResult = logaddexp(array([1000, 500]), array([1000, 600]));
      const pyResult = runNumPy(`
result = np.logaddexp(np.array([1000, 500]), np.array([1000, 600]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('logaddexp2', () => {
    it('matches NumPy for equal values', () => {
      const jsResult = logaddexp2(array([1, 2, 3]), array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.logaddexp2(np.array([1, 2, 3]), np.array([1, 2, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for different values', () => {
      const jsResult = logaddexp2(array([0, 1, 2]), array([3, 4, 5]));
      const pyResult = runNumPy(`
result = np.logaddexp2(np.array([0, 1, 2]), np.array([3, 4, 5]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for large values (numerical stability)', () => {
      const jsResult = logaddexp2(array([1000, 500]), array([1000, 600]));
      const pyResult = runNumPy(`
result = np.logaddexp2(np.array([1000, 500]), np.array([1000, 600]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('roundtrip', () => {
    it('exp(log(x)) matches NumPy', () => {
      const x = array([1, 2, 10, 100]);
      const jsResult = exp(log(x));
      const pyResult = runNumPy(`
result = np.exp(np.log(np.array([1, 2, 10, 100])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
