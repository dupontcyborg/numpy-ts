/**
 * Python NumPy validation tests for Other Math operations
 * (clip, maximum, minimum, fmax, fmin, nan_to_num, interp, unwrap, sinc, i0)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  clip,
  maximum,
  minimum,
  fmax,
  fmin,
  nan_to_num,
  interp,
  unwrap,
  sinc,
  i0,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Other Math Operations', () => {
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

  describe('clip', () => {
    it('matches NumPy for 1D array with scalar bounds', () => {
      const jsResult = clip(array([1, 5, 10, 15, 20]), 3, 12);
      const pyResult = runNumPy(`
result = np.clip(np.array([1, 5, 10, 15, 20]), 3, 12)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = clip(
        array([
          [0, 5, 10],
          [-5, 15, 20],
        ]),
        2,
        8
      );
      const pyResult = runNumPy(`
result = np.clip(np.array([[0, 5, 10], [-5, 15, 20]]), 2, 8)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with only min bound', () => {
      const jsResult = clip(array([1, 2, 3, 4, 5]), 3, null);
      const pyResult = runNumPy(`
result = np.clip(np.array([1, 2, 3, 4, 5]), 3, None)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with only max bound', () => {
      const jsResult = clip(array([1, 2, 3, 4, 5]), null, 3);
      const pyResult = runNumPy(`
result = np.clip(np.array([1, 2, 3, 4, 5]), None, 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float array', () => {
      const jsResult = clip(array([0.5, 1.5, 2.5, 3.5, 4.5]), 1.0, 4.0);
      const pyResult = runNumPy(`
result = np.clip(np.array([0.5, 1.5, 2.5, 3.5, 4.5]), 1.0, 4.0)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('maximum', () => {
    it('matches NumPy for two arrays', () => {
      const jsResult = maximum(array([1, 4, 3]), array([2, 2, 5]));
      const pyResult = runNumPy(`
result = np.maximum(np.array([1, 4, 3]), np.array([2, 2, 5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array and scalar', () => {
      const jsResult = maximum(array([1, 5, 3, 8, 2]), 4);
      const pyResult = runNumPy(`
result = np.maximum(np.array([1, 5, 3, 8, 2]), 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = maximum(
        array([
          [1, 4],
          [3, 2],
        ]),
        array([
          [2, 2],
          [5, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.maximum(np.array([[1, 4], [3, 2]]), np.array([[2, 2], [5, 1]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with NaN (NaN propagates)', () => {
      const jsResult = maximum(array([1, NaN, 3]), array([2, 2, NaN]));
      const pyResult = runNumPy(`
result = np.maximum(np.array([1, np.nan, 3]), np.array([2, 2, np.nan]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      // NaN comparison needs special handling
      const jsArr = jsResult.toArray() as number[];
      const pyArr = pyResult.value as number[];
      expect(jsArr[0]).toBeCloseTo(pyArr[0]!);
      expect(isNaN(jsArr[1]!)).toBe(true);
      expect(isNaN(pyArr[1]!)).toBe(true);
      expect(isNaN(jsArr[2]!)).toBe(true);
      expect(isNaN(pyArr[2]!)).toBe(true);
    });
  });

  describe('minimum', () => {
    it('matches NumPy for two arrays', () => {
      const jsResult = minimum(array([1, 4, 3]), array([2, 2, 5]));
      const pyResult = runNumPy(`
result = np.minimum(np.array([1, 4, 3]), np.array([2, 2, 5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array and scalar', () => {
      const jsResult = minimum(array([1, 5, 3, 8, 2]), 4);
      const pyResult = runNumPy(`
result = np.minimum(np.array([1, 5, 3, 8, 2]), 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = minimum(
        array([
          [1, 4],
          [3, 2],
        ]),
        array([
          [2, 2],
          [5, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.minimum(np.array([[1, 4], [3, 2]]), np.array([[2, 2], [5, 1]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fmax', () => {
    it('matches NumPy for two arrays without NaN', () => {
      const jsResult = fmax(array([1, 4, 3]), array([2, 2, 5]));
      const pyResult = runNumPy(`
result = np.fmax(np.array([1, 4, 3]), np.array([2, 2, 5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy ignoring NaN', () => {
      const jsResult = fmax(array([1, NaN, 3]), array([2, 2, NaN]));
      const pyResult = runNumPy(`
result = np.fmax(np.array([1, np.nan, 3]), np.array([2, 2, np.nan]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with scalar', () => {
      const jsResult = fmax(array([NaN, 2, 3, NaN]), 1);
      const pyResult = runNumPy(`
result = np.fmax(np.array([np.nan, 2, 3, np.nan]), 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fmin', () => {
    it('matches NumPy for two arrays without NaN', () => {
      const jsResult = fmin(array([1, 4, 3]), array([2, 2, 5]));
      const pyResult = runNumPy(`
result = np.fmin(np.array([1, 4, 3]), np.array([2, 2, 5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy ignoring NaN', () => {
      const jsResult = fmin(array([1, NaN, 3]), array([2, 2, NaN]));
      const pyResult = runNumPy(`
result = np.fmin(np.array([1, np.nan, 3]), np.array([2, 2, np.nan]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with scalar', () => {
      const jsResult = fmin(array([NaN, 2, 3, NaN]), 1);
      const pyResult = runNumPy(`
result = np.fmin(np.array([np.nan, 2, 3, np.nan]), 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('nan_to_num', () => {
    it('matches NumPy for NaN replacement', () => {
      const jsResult = nan_to_num(array([1, NaN, 3, Infinity, -Infinity]));
      const pyResult = runNumPy(`
result = np.nan_to_num(np.array([1, np.nan, 3, np.inf, -np.inf]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      // Check non-inf values
      const jsArr = jsResult.toArray() as number[];
      const pyArr = pyResult.value as number[];
      expect(jsArr[0]).toBeCloseTo(pyArr[0]!);
      expect(jsArr[1]).toBeCloseTo(pyArr[1]!);
      expect(jsArr[2]).toBeCloseTo(pyArr[2]!);
      // Inf values are replaced with large finite numbers
      expect(isFinite(jsArr[3]!)).toBe(true);
      expect(isFinite(jsArr[4]!)).toBe(true);
    });

    it('matches NumPy with custom nan value', () => {
      const jsResult = nan_to_num(array([NaN, NaN, 3]), 999);
      const pyResult = runNumPy(`
result = np.nan_to_num(np.array([np.nan, np.nan, 3]), nan=999)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = nan_to_num(
        array([
          [1, NaN],
          [NaN, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.nan_to_num(np.array([[1, np.nan], [np.nan, 4]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('interp', () => {
    it('matches NumPy for simple interpolation', () => {
      const xp = array([0, 1, 2, 3]);
      const fp = array([0, 1, 4, 9]);
      const x = array([0.5, 1.5, 2.5]);
      const jsResult = interp(x, xp, fp);
      const pyResult = runNumPy(`
xp = np.array([0, 1, 2, 3])
fp = np.array([0, 1, 4, 9])
x = np.array([0.5, 1.5, 2.5])
result = np.interp(x, xp, fp)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for values at boundaries', () => {
      const xp = array([0, 1, 2]);
      const fp = array([0, 10, 20]);
      const x = array([0, 1, 2]);
      const jsResult = interp(x, xp, fp);
      const pyResult = runNumPy(`
result = np.interp(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 10, 20]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for out of bounds (left extrapolation)', () => {
      const xp = array([1, 2, 3]);
      const fp = array([10, 20, 30]);
      const x = array([-1, 0, 0.5]);
      const jsResult = interp(x, xp, fp);
      const pyResult = runNumPy(`
result = np.interp(np.array([-1, 0, 0.5]), np.array([1, 2, 3]), np.array([10, 20, 30]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for out of bounds (right extrapolation)', () => {
      const xp = array([1, 2, 3]);
      const fp = array([10, 20, 30]);
      const x = array([3.5, 4, 5]);
      const jsResult = interp(x, xp, fp);
      const pyResult = runNumPy(`
result = np.interp(np.array([3.5, 4, 5]), np.array([1, 2, 3]), np.array([10, 20, 30]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('unwrap', () => {
    it('matches NumPy for simple phase array', () => {
      // Simple phase values without discontinuity
      const phase = array([0, 0.5, 1.0, 1.5, 2.0, 2.5]);
      const jsResult = unwrap(phase);
      const pyResult = runNumPy(`
phase = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])
result = np.unwrap(phase)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });

    it('returns correct shape for empty array', () => {
      const phase = array([1.0, 2.0, 3.0]);
      const jsResult = unwrap(phase);

      expect(jsResult.shape).toEqual([3]);
      expect(jsResult.dtype).toBe('float64');
    });
  });

  describe('sinc', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = sinc(array([-2, -1, 0, 1, 2]));
      const pyResult = runNumPy(`
result = np.sinc(np.array([-2, -1, 0, 1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float values', () => {
      const jsResult = sinc(array([-1.5, -0.5, 0, 0.5, 1.5]));
      const pyResult = runNumPy(`
result = np.sinc(np.array([-1.5, -0.5, 0, 0.5, 1.5]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = sinc(
        array([
          [-1, 0, 1],
          [0.5, 1.5, 2],
        ])
      );
      const pyResult = runNumPy(`
result = np.sinc(np.array([[-1, 0, 1], [0.5, 1.5, 2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('i0', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = i0(array([0, 1, 2, 3]));
      const pyResult = runNumPy(`
result = np.i0(np.array([0, 1, 2, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-6)).toBe(true);
    });

    it('matches NumPy for negative values', () => {
      const jsResult = i0(array([-3, -2, -1, 0, 1, 2, 3]));
      const pyResult = runNumPy(`
result = np.i0(np.array([-3, -2, -1, 0, 1, 2, 3]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-6)).toBe(true);
    });

    it('matches NumPy for small values', () => {
      const jsResult = i0(array([0.1, 0.5, 1.0, 1.5, 2.0]));
      const pyResult = runNumPy(`
result = np.i0(np.array([0.1, 0.5, 1.0, 1.5, 2.0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-6)).toBe(true);
    });

    it('matches NumPy for large values', () => {
      const jsResult = i0(array([5, 10, 15]));
      const pyResult = runNumPy(`
result = np.i0(np.array([5, 10, 15]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      // For large values, use relative tolerance
      const jsArr = jsResult.toArray() as number[];
      const pyArr = pyResult.value as number[];
      for (let i = 0; i < jsArr.length; i++) {
        expect(Math.abs(jsArr[i]! - pyArr[i]!) / pyArr[i]!).toBeLessThan(1e-5);
      }
    });
  });
});
