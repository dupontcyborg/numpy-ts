/**
 * Python NumPy validation tests for Utility functions
 * (apply_along_axis, apply_over_axes, shares_memory, ndim, shape, size, geterr, seterr)
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import {
  array,
  zeros,
  apply_along_axis,
  apply_over_axes,
  may_share_memory,
  shares_memory,
  ndim,
  shape,
  size,
  geterr,
  seterr,
  NDArray,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Utility Functions', () => {
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

  describe('apply_along_axis', () => {
    it('matches NumPy for sum along axis 0', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      // Apply sum function along axis 0 (columns)
      const sumFunc = (a: NDArray) => {
        let total = 0;
        const data = a.toArray() as number[];
        for (const v of data) {
          total += v;
        }
        return total;
      };
      const jsResult = apply_along_axis(sumFunc, 0, arr);
      const pyResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.apply_along_axis(np.sum, 0, arr)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for sum along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      // Apply sum function along axis 1 (rows)
      const sumFunc = (a: NDArray) => {
        let total = 0;
        const data = a.toArray() as number[];
        for (const v of data) {
          total += v;
        }
        return total;
      };
      const jsResult = apply_along_axis(sumFunc, 1, arr);
      const pyResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.apply_along_axis(np.sum, 1, arr)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('apply_over_axes', () => {
    it('matches NumPy for sum over axis 0', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      // Apply sum function over axis 0 - func must take (array, axis) and return array
      // Use NDArray's sum method with keepdims=true
      const sumFunc = (a: NDArray, axis: number) => a.sum(axis, undefined, true);
      const jsResult = apply_over_axes(sumFunc, arr, [0]);
      const pyResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.apply_over_axes(np.sum, arr, [0])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('may_share_memory / shares_memory', () => {
    it('returns true for arrays sharing the same buffer', () => {
      const a = zeros([5]);
      // Create a view that shares the same buffer
      const b = a;

      expect(may_share_memory(a, b)).toBe(true);
      expect(shares_memory(a, b)).toBe(true);
    });

    it('returns false for independent arrays', () => {
      const a = zeros([5]);
      const b = zeros([5]);

      // Independent arrays shouldn't share memory
      expect(may_share_memory(a, b)).toBe(false);
      expect(shares_memory(a, b)).toBe(false);
    });
  });

  describe('ndim', () => {
    it('matches NumPy for 0D (scalar)', () => {
      const jsResult = ndim(5);
      const pyResult = runNumPy(`
result = np.ndim(5)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 1D array', () => {
      const jsResult = ndim(array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.ndim(np.array([1, 2, 3]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = ndim(
        array([
          [1, 2],
          [3, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.ndim(np.array([[1, 2], [3, 4]]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 3D array', () => {
      const jsResult = ndim(
        array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ])
      );
      const pyResult = runNumPy(`
result = np.ndim(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for nested list input', () => {
      const jsResult = ndim([
        [1, 2],
        [3, 4],
      ]);
      const pyResult = runNumPy(`
result = np.ndim([[1, 2], [3, 4]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('shape', () => {
    it('matches NumPy for scalar', () => {
      const jsResult = shape(5);
      const pyResult = runNumPy(`
result = list(np.shape(5))
      `);

      expect(jsResult).toEqual(pyResult.value);
    });

    it('matches NumPy for 1D array', () => {
      const jsResult = shape(array([1, 2, 3, 4, 5]));
      const pyResult = runNumPy(`
result = list(np.shape(np.array([1, 2, 3, 4, 5])))
      `);

      expect(jsResult).toEqual(pyResult.value);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = shape(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ])
      );
      const pyResult = runNumPy(`
result = list(np.shape(np.array([[1, 2, 3], [4, 5, 6]])))
      `);

      expect(jsResult).toEqual(pyResult.value);
    });

    it('matches NumPy for nested list', () => {
      const jsResult = shape([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const pyResult = runNumPy(`
result = list(np.shape([[1, 2], [3, 4], [5, 6]]))
      `);

      expect(jsResult).toEqual(pyResult.value);
    });
  });

  describe('size', () => {
    it('matches NumPy for scalar', () => {
      const jsResult = size(5);
      const pyResult = runNumPy(`
result = np.size(5)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 1D array', () => {
      const jsResult = size(array([1, 2, 3, 4, 5]));
      const pyResult = runNumPy(`
result = np.size(np.array([1, 2, 3, 4, 5]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = size(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ])
      );
      const pyResult = runNumPy(`
result = np.size(np.array([[1, 2, 3], [4, 5, 6]]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 3D array', () => {
      const jsResult = size(
        array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ])
      );
      const pyResult = runNumPy(`
result = np.size(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('geterr / seterr', () => {
    afterEach(() => {
      // Reset to default state
      seterr('warn', 'warn', 'ignore', 'warn');
    });

    it('returns current error state', () => {
      const state = geterr();

      expect(state).toHaveProperty('divide');
      expect(state).toHaveProperty('over');
      expect(state).toHaveProperty('under');
      expect(state).toHaveProperty('invalid');
    });

    it('seterr updates error state', () => {
      const oldState = seterr('ignore', 'ignore', 'ignore', 'ignore', 'ignore');
      const newState = geterr();

      expect(newState.divide).toBe('ignore');
      expect(newState.over).toBe('ignore');
      expect(newState.under).toBe('ignore');
      expect(newState.invalid).toBe('ignore');

      // Restore old state
      seterr(undefined, oldState.divide, oldState.over, oldState.under, oldState.invalid);
    });

    it('seterr returns old state', () => {
      const initialState = geterr();
      const oldState = seterr('raise');

      expect(oldState).toEqual(initialState);
    });

    it('seterr with individual parameters', () => {
      seterr(undefined, 'raise', undefined, undefined, undefined);
      const state = geterr();

      expect(state.divide).toBe('raise');
    });
  });
});
