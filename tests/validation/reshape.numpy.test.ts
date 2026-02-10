/**
 * Python NumPy validation tests for reshape operations
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Reshape Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('reshape()', () => {
    it('validates reshape 1D to 2D', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = arr.reshape(2, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = arr.reshape(2, 3)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reshape 2D to 1D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.reshape(6);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.reshape(6)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reshape with -1 (infer dimension)', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = arr.reshape(2, -1);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = arr.reshape(2, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reshape 2D to 3D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.reshape(2, 1, 3);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.reshape(2, 1, 3)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('flatten()', () => {
    it('validates flatten 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.flatten();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.flatten()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates flatten 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.flatten();

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.flatten()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('ravel()', () => {
    it('validates ravel 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.ravel();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.ravel()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('transpose()', () => {
    it('validates transpose 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.transpose();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.transpose()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates transpose 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.transpose();

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.transpose()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates transpose with custom axes', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = arr.transpose([1, 0, 2]);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.transpose([1, 0, 2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('squeeze()', () => {
    it('validates squeeze all axes', () => {
      const arr = array([[[1, 2, 3]]]);
      const result = arr.squeeze();

      const npResult = runNumPy(`
arr = np.array([[[1, 2, 3]]])
result = np.squeeze(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates squeeze specific axis', () => {
      const arr = array([[[1, 2, 3]]]);
      const result = arr.squeeze(0);

      const npResult = runNumPy(`
arr = np.array([[[1, 2, 3]]])
result = np.squeeze(arr, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates squeeze with negative axis', () => {
      const arr = array([[[1, 2, 3]]]);
      const result = arr.squeeze(-3);

      const npResult = runNumPy(`
arr = np.array([[[1, 2, 3]]])
result = np.squeeze(arr, axis=-3)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates squeeze with multiple size-1 axes', () => {
      const arr = array([[[[1], [2]]]]);
      const result = arr.squeeze();

      const npResult = runNumPy(`
arr = np.array([[[[1], [2]]]])
result = np.squeeze(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('expand_dims()', () => {
    it('validates expand_dims at beginning', () => {
      const arr = array([1, 2, 3]);
      const result = arr.expand_dims(0);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.expand_dims(arr, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates expand_dims at end', () => {
      const arr = array([1, 2, 3]);
      const result = arr.expand_dims(1);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.expand_dims(arr, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates expand_dims in middle', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.expand_dims(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.expand_dims(arr, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates expand_dims with negative axis', () => {
      const arr = array([1, 2, 3]);
      const result = arr.expand_dims(-1);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.expand_dims(arr, axis=-1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('combined operations', () => {
    it('validates reshape then transpose', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = arr.reshape(2, 3).transpose();

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = arr.reshape(2, 3).transpose()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates transpose then flatten', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.transpose().flatten();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.transpose().flatten()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
