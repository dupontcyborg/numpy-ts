/**
 * Python NumPy validation tests for hyperbolic operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, sinh, cosh, tanh, arcsinh, arccosh, arctanh } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Hyperbolic Operations', () => {
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

  describe('sinh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = sinh(array([-2, -1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.sinh(np.array([-2, -1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = sinh(
        array([
          [-1, 0],
          [1, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.sinh(np.array([[-1, 0], [1, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = sinh(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.sinh(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float32', () => {
      const jsResult = sinh(array([0, 1, 2], 'float32'));
      const pyResult = runNumPy(`
result = np.sinh(np.array([0, 1, 2], dtype=np.float32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cosh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = cosh(array([-2, -1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.cosh(np.array([-2, -1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = cosh(
        array([
          [-1, 0],
          [1, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.cosh(np.array([[-1, 0], [1, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = cosh(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.cosh(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('is even function: cosh(-x) = cosh(x) matches NumPy', () => {
      const jsPos = cosh(array([1, 2, 3]));
      const jsNeg = cosh(array([-1, -2, -3]));

      const pyResult = runNumPy(`
pos = np.cosh(np.array([1, 2, 3]))
neg = np.cosh(np.array([-1, -2, -3]))
result = pos - neg  # Should be zeros
      `);

      // Both should be equal
      expect(arraysClose(jsPos.toArray(), jsNeg.toArray())).toBe(true);
      // And the difference from NumPy should be zero
      expect(arraysClose(pyResult.value, [0, 0, 0])).toBe(true);
    });
  });

  describe('tanh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = tanh(array([-2, -1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.tanh(np.array([-2, -1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = tanh(
        array([
          [-1, 0],
          [1, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.tanh(np.array([[-1, 0], [1, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = tanh(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.tanh(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for large values approaching limits', () => {
      const jsResult = tanh(array([-100, 100]));
      const pyResult = runNumPy(`
result = np.tanh(np.array([-100, 100]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arcsinh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arcsinh(array([-10, -1, 0, 1, 10]));
      const pyResult = runNumPy(`
result = np.arcsinh(np.array([-10, -1, 0, 1, 10]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = arcsinh(
        array([
          [-1, 0],
          [1, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.arcsinh(np.array([[-1, 0], [1, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arcsinh(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.arcsinh(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('roundtrip with sinh matches NumPy', () => {
      const original = array([-2, -1, 0, 1, 2]);
      const jsSinh = sinh(original);
      const jsBack = arcsinh(jsSinh);

      const pyResult = runNumPy(`
original = np.array([-2, -1, 0, 1, 2])
result = np.arcsinh(np.sinh(original))
      `);

      expect(arraysClose(jsBack.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arccosh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arccosh(array([1, 2, 5, 10]));
      const pyResult = runNumPy(`
result = np.arccosh(np.array([1, 2, 5, 10]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = arccosh(
        array([
          [1, 2],
          [3, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.arccosh(np.array([[1, 2], [3, 4]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arccosh(array([1, 2, 3], 'int32'));
      const pyResult = runNumPy(`
result = np.arccosh(np.array([1, 2, 3], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for NaN on values < 1', () => {
      const jsResult = arccosh(array([0, 0.5]));
      const pyResult = runNumPy(`
result = np.arccosh(np.array([0, 0.5]))
      `);

      // Both should be NaN
      const jsArr = jsResult.toArray() as number[];
      expect(Number.isNaN(jsArr[0])).toBe(true);
      expect(Number.isNaN(jsArr[1])).toBe(true);
      expect(Number.isNaN(pyResult.value[0])).toBe(true);
      expect(Number.isNaN(pyResult.value[1])).toBe(true);
    });
  });

  describe('arctanh', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arctanh(array([-0.9, -0.5, 0, 0.5, 0.9]));
      const pyResult = runNumPy(`
result = np.arctanh(np.array([-0.9, -0.5, 0, 0.5, 0.9]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = arctanh(
        array([
          [-0.5, 0],
          [0.5, 0.9],
        ])
      );
      const pyResult = runNumPy(`
result = np.arctanh(np.array([[-0.5, 0], [0.5, 0.9]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arctanh(array([0], 'int32'));
      const pyResult = runNumPy(`
result = np.arctanh(np.array([0], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for infinity at boundaries', () => {
      const jsResult = arctanh(array([1, -1]));
      const pyResult = runNumPy(`
result = np.arctanh(np.array([1.0, -1.0]))
      `);

      // Both should be ±Infinity
      const jsArr = jsResult.toArray() as number[];
      expect(jsArr[0]).toBe(Infinity);
      expect(jsArr[1]).toBe(-Infinity);
      expect(pyResult.value[0]).toBe(Infinity);
      expect(pyResult.value[1]).toBe(-Infinity);
    });

    it('roundtrip with tanh matches NumPy', () => {
      const original = array([-2, -1, 0, 1, 2]);
      const jsTanh = tanh(original);
      const jsBack = arctanh(jsTanh);

      const pyResult = runNumPy(`
original = np.array([-2, -1, 0, 1, 2])
result = np.arctanh(np.tanh(original))
      `);

      expect(arraysClose(jsBack.toArray(), pyResult.value)).toBe(true);
    });
  });
});
