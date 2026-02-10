/**
 * Python NumPy validation tests for slicing operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Slicing', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          `   Current Python command: ${process.env.NUMPY_PYTHON || 'python3'}\n`
      );
    }

    const info = getPythonInfo();
    console.log(`\n  ✓ Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('1D slicing', () => {
    it('validates basic slice start:stop', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('2:7');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[2:7]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with colon ":"', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.slice(':');

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = arr[:]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice from start "5:"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('5:');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[5:]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice to stop ":5"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice(':5');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[:5]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with negative start "-3:"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('-3:');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[-3:]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with negative stop ":-3"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice(':-3');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[:-3]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with negative start and stop "-7:-2"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('-7:-2');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[-7:-2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates single index "2"', () => {
      const arr = array([0, 1, 2, 3, 4, 5]);
      const result = arr.slice('2');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5])
result = arr[2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(result.toArray()).toEqual(npResult.value);
    });

    it('validates negative index "-1"', () => {
      const arr = array([0, 1, 2, 3, 4, 5]);
      const result = arr.slice('-1');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5])
result = arr[-1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(result.toArray()).toEqual(npResult.value);
    });

    it('validates slice with step "::2"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('::2');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[::2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with start:stop:step "1:8:2"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('1:8:2');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[1:8:2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reverse slice "::-1"', () => {
      const arr = array([0, 1, 2, 3, 4]);
      const result = arr.slice('::-1');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4])
result = arr[::-1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reverse slice with bounds "7:2:-1"', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('7:2:-1');

      const npResult = runNumPy(`
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = arr[7:2:-1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('2D slicing', () => {
    it('validates single row slice', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('0', ':');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[0, :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates single column slice', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice(':', '1');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[:, 1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates row range slice', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('0:2', ':');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[0:2, :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates column range slice', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice(':', '1:3');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[:, 1:3]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates submatrix slice', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);
      const result = arr.slice('1:3', '1:3');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
result = arr[1:3, 1:3]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates negative indices slice', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('-2:', '-2:');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[-2:, -2:]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates scalar element access', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('-1', '-1');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[-1, -1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(result.toArray()).toEqual(npResult.value);
    });

    it('validates slice with step on rows', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const result = arr.slice('::2', ':');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
result = arr[::2, :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with step on columns', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = arr.slice(':', '::2');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = arr[:, ::2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates slice with step on both dimensions', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);
      const result = arr.slice('::2', '::2');

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
result = arr[::2, ::2]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('convenience methods', () => {
    it('validates row() method', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.row(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[1, :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates col() method', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.col(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr[:, 1]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates rows() method', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const result = arr.rows(1, 3);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
result = arr[1:3, :]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates cols() method', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const result = arr.cols(1, 3);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = arr[:, 1:3]
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
