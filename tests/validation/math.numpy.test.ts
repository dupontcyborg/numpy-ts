/**
 * Python NumPy validation tests for mathematical operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  sqrt,
  power,
  absolute,
  negative,
  sign,
  mod,
  floor_divide,
  positive,
  reciprocal,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Mathematical Operations', () => {
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

  describe('sqrt', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = sqrt(array([4, 9, 16, 25]));
      const pyResult = runNumPy(`
result = np.sqrt(np.array([4, 9, 16, 25]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = sqrt(
        array([
          [1, 4, 9],
          [16, 25, 36],
        ])
      );
      const pyResult = runNumPy(`
result = np.sqrt(np.array([[1, 4, 9], [16, 25, 36]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = sqrt(array([4, 9, 16], 'int32'));
      const pyResult = runNumPy(`
result = np.sqrt(np.array([4, 9, 16], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float32', () => {
      const jsResult = sqrt(array([1, 4, 9], 'float32'));
      const pyResult = runNumPy(`
result = np.sqrt(np.array([1, 4, 9], dtype=np.float32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('power', () => {
    it('matches NumPy for scalar exponent', () => {
      const jsResult = power(array([2, 3, 4]), 2);
      const pyResult = runNumPy(`
result = np.power(np.array([2, 3, 4]), 2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = power(
        array([
          [2, 3],
          [4, 5],
        ]),
        3
      );
      const pyResult = runNumPy(`
result = np.power(np.array([[2, 3], [4, 5]]), 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative exponent', () => {
      const jsResult = power(array([2.0, 4.0, 8.0]), -1);
      const pyResult = runNumPy(`
result = np.power(np.array([2.0, 4.0, 8.0]), -1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for fractional exponent', () => {
      const jsResult = power(array([4, 9, 16]), 0.5);
      const pyResult = runNumPy(`
result = np.power(np.array([4, 9, 16]), 0.5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for negative exponent', () => {
      // NumPy doesn't allow integer arrays with negative exponents
      // Test that our implementation promotes to float64 correctly
      const jsResult = power(array([2, 4, 8], 'int32'), -1);

      expect(jsResult.dtype).toBe('float64'); // Should promote to float64
      expect(arraysClose(jsResult.toArray(), [0.5, 0.25, 0.125])).toBe(true);
    });

    it('matches NumPy for zero exponent', () => {
      const jsResult = power(array([1, 2, 3, 0]), 0);
      const pyResult = runNumPy(`
result = np.power(np.array([1, 2, 3, 0]), 0)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('absolute', () => {
    it('matches NumPy for mixed signs', () => {
      const jsResult = absolute(array([-3, -1, 0, 1, 3]));
      const pyResult = runNumPy(`
result = np.absolute(np.array([-3, -1, 0, 1, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = absolute(
        array([
          [-1, 2, -3],
          [4, -5, 6],
        ])
      );
      const pyResult = runNumPy(`
result = np.absolute(np.array([[-1, 2, -3], [4, -5, 6]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation for int32', () => {
      const jsResult = absolute(array([-5, 10, -15], 'int32'));
      const pyResult = runNumPy(`
result = np.absolute(np.array([-5, 10, -15], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float64', () => {
      const jsResult = absolute(array([-3.5, 2.1, -0.5], 'float64'));
      const pyResult = runNumPy(`
result = np.absolute(np.array([-3.5, 2.1, -0.5], dtype=np.float64))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('negative', () => {
    it('matches NumPy for mixed signs', () => {
      const jsResult = negative(array([1, -2, 3, -4, 0]));
      const pyResult = runNumPy(`
result = np.negative(np.array([1, -2, 3, -4, 0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = negative(
        array([
          [1, -2],
          [-3, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.negative(np.array([[1, -2], [-3, 4]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation for int32', () => {
      const jsResult = negative(array([5, -10, 15], 'int32'));
      const pyResult = runNumPy(`
result = np.negative(np.array([5, -10, 15], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float32', () => {
      const jsResult = negative(array([3.5, -2.1, 0.5], 'float32'));
      const pyResult = runNumPy(`
result = np.negative(np.array([3.5, -2.1, 0.5], dtype=np.float32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('sign', () => {
    it('matches NumPy for mixed signs', () => {
      const jsResult = sign(array([-5, -0.5, 0, 0.5, 5]));
      const pyResult = runNumPy(`
result = np.sign(np.array([-5, -0.5, 0, 0.5, 5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = sign(
        array([
          [-1, 0, 1],
          [2, -2, 0],
        ])
      );
      const pyResult = runNumPy(`
result = np.sign(np.array([[-1, 0, 1], [2, -2, 0]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation for int32', () => {
      const jsResult = sign(array([-5, 0, 5], 'int32'));
      const pyResult = runNumPy(`
result = np.sign(np.array([-5, 0, 5], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float64', () => {
      const jsResult = sign(array([-3.5, 0, 2.1], 'float64'));
      const pyResult = runNumPy(`
result = np.sign(np.array([-3.5, 0, 2.1], dtype=np.float64))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for all zeros', () => {
      const jsResult = sign(array([0, 0, 0]));
      const pyResult = runNumPy(`
result = np.sign(np.array([0, 0, 0]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('mod', () => {
    it('matches NumPy for scalar modulo', () => {
      const jsResult = mod(array([10, 11, 12, 13]), 3);
      const pyResult = runNumPy(`
result = np.mod(np.array([10, 11, 12, 13]), 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = mod(
        array([
          [10, 11, 12],
          [13, 14, 15],
        ]),
        4
      );
      const pyResult = runNumPy(`
result = np.mod(np.array([[10, 11, 12], [13, 14, 15]]), 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation', () => {
      const jsResult = mod(array([10, 11, 12], 'int32'), 3);
      const pyResult = runNumPy(`
result = np.mod(np.array([10, 11, 12], dtype=np.int32), 3)
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = mod(array([-10, -7, -3, 0, 3, 7, 10]), 3);
      const pyResult = runNumPy(`
result = np.mod(np.array([-10, -7, -3, 0, 3, 7, 10]), 3)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('floor_divide', () => {
    it('matches NumPy for scalar floor division', () => {
      const jsResult = floor_divide(array([10, 11, 12, 13]), 3);
      const pyResult = runNumPy(`
result = np.floor_divide(np.array([10, 11, 12, 13]), 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = floor_divide(
        array([
          [10, 15],
          [20, 25],
        ]),
        4
      );
      const pyResult = runNumPy(`
result = np.floor_divide(np.array([[10, 15], [20, 25]]), 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation for integers', () => {
      const jsResult = floor_divide(array([10, 11, 12], 'int32'), 3);
      const pyResult = runNumPy(`
result = np.floor_divide(np.array([10, 11, 12], dtype=np.int32), 3)
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = floor_divide(array([-10, -7, -3, 0, 3, 7, 10]), 3);
      const pyResult = runNumPy(`
result = np.floor_divide(np.array([-10, -7, -3, 0, 3, 7, 10]), 3)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('positive', () => {
    it('matches NumPy for mixed values', () => {
      const jsResult = positive(array([1, -2, 3, -4, 0]));
      const pyResult = runNumPy(`
result = np.positive(np.array([1, -2, 3, -4, 0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = positive(
        array([
          [1, -2],
          [-3, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.positive(np.array([[1, -2], [-3, 4]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype preservation', () => {
      const jsResult = positive(array([5, -10, 15], 'int32'));
      const pyResult = runNumPy(`
result = np.positive(np.array([5, -10, 15], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('reciprocal', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = reciprocal(array([2, 4, 8]));
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([2.0, 4.0, 8.0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = reciprocal(
        array([
          [1, 2],
          [4, 5],
        ])
      );
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([[1.0, 2.0], [4.0, 5.0]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = reciprocal(array([2, 4, 8], 'int32'));
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([2.0, 4.0, 8.0]))
      `);

      expect(jsResult.dtype).toBe('float64'); // Promoted to float64
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = reciprocal(array([-2, -4, 2, 4]));
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([-2.0, -4.0, 2.0, 4.0]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
