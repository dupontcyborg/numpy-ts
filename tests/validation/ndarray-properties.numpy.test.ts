/**
 * Python NumPy validation tests for NDArray properties and methods
 * Tests: .T, .itemsize, .nbytes, .fill(), iterator protocol, copyto()
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros, array, arange, copyto } from '../../src/core/ndarray';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: NDArray Properties', () => {
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

  describe('.T (transpose)', () => {
    it('validates .T for 2D array', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]).T;

      const pyResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.T
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('validates .T for 1D array (no-op)', () => {
      const jsResult = array([1, 2, 3]).T;

      const pyResult = runNumPy(`
arr = np.array([1, 2, 3])
result = arr.T
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('validates .T for 3D array', () => {
      const jsResult = arange(24).reshape(2, 3, 4).T;

      const pyResult = runNumPy(`
arr = np.arange(24).reshape(2, 3, 4)
result = arr.T
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('validates .T is equivalent to .transpose()', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.T.toArray()).toEqual(arr.transpose().toArray());

      const pyResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.T.tolist() == arr.transpose().tolist()
      `);

      expect(pyResult.value).toBe(true);
    });
  });

  describe('.itemsize', () => {
    it('validates itemsize for float64', () => {
      const jsResult = zeros([3], 'float64').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.float64).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for float32', () => {
      const jsResult = zeros([3], 'float32').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.float32).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for int64', () => {
      const jsResult = zeros([3], 'int64').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.int64).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for int32', () => {
      const jsResult = zeros([3], 'int32').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.int32).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for int16', () => {
      const jsResult = zeros([3], 'int16').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.int16).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for int8', () => {
      const jsResult = zeros([3], 'int8').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.int8).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for uint8', () => {
      const jsResult = zeros([3], 'uint8').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.uint8).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates itemsize for bool', () => {
      const jsResult = zeros([3], 'bool').itemsize;

      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.bool_).itemsize
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('.nbytes', () => {
    it('validates nbytes for float64', () => {
      const jsResult = zeros([10], 'float64').nbytes;

      const pyResult = runNumPy(`
result = np.zeros(10, dtype=np.float64).nbytes
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates nbytes for 2D int32 array', () => {
      const jsResult = zeros([5, 4], 'int32').nbytes;

      const pyResult = runNumPy(`
result = np.zeros((5, 4), dtype=np.int32).nbytes
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates nbytes for uint8', () => {
      const jsResult = zeros([100], 'uint8').nbytes;

      const pyResult = runNumPy(`
result = np.zeros(100, dtype=np.uint8).nbytes
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('validates nbytes equals size * itemsize', () => {
      const arr = zeros([3, 4, 5], 'float64');
      expect(arr.nbytes).toBe(arr.size * arr.itemsize);

      const pyResult = runNumPy(`
arr = np.zeros((3, 4, 5), dtype=np.float64)
result = arr.nbytes == arr.size * arr.itemsize
      `);

      expect(pyResult.value).toBe(true);
    });
  });

  describe('.fill()', () => {
    it('validates fill for float64', () => {
      const jsArr = zeros([5]);
      jsArr.fill(7);

      const pyResult = runNumPy(`
arr = np.zeros(5)
arr.fill(7)
result = arr
      `);

      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('validates fill for int32', () => {
      const jsArr = zeros([3], 'int32');
      jsArr.fill(42);

      const pyResult = runNumPy(`
arr = np.zeros(3, dtype=np.int32)
arr.fill(42)
result = arr
      `);

      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('validates fill for 2D array', () => {
      const jsArr = zeros([2, 3]);
      jsArr.fill(9);

      const pyResult = runNumPy(`
arr = np.zeros((2, 3))
arr.fill(9)
result = arr
      `);

      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('validates fill truncates floats for int dtype', () => {
      const jsArr = zeros([3], 'int32');
      jsArr.fill(3.7);

      const pyResult = runNumPy(`
arr = np.zeros(3, dtype=np.int32)
arr.fill(3.7)
result = arr
      `);

      expect(jsArr.toArray()).toEqual(pyResult.value);
    });
  });

  describe('Iterator protocol', () => {
    it('validates iteration over 1D array matches NumPy list()', () => {
      const jsArr = array([1, 2, 3, 4, 5]);
      const jsValues = [...jsArr];

      const pyResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = [int(x) for x in arr]
      `);

      expect(jsValues).toEqual(pyResult.value);
    });

    it('validates iteration over 2D array yields rows', () => {
      const jsArr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsRows = [...jsArr].map((row) => (row as ReturnType<typeof array>).toArray());

      const pyResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = [row.tolist() for row in arr]
      `);

      expect(jsRows).toEqual(pyResult.value);
    });

    it('validates iteration count matches first dimension', () => {
      const jsArr = zeros([5, 3, 2]);
      let count = 0;
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for (const _ of jsArr) count++;

      const pyResult = runNumPy(`
arr = np.zeros((5, 3, 2))
result = len(list(arr))
      `);

      expect(count).toBe(pyResult.value);
    });
  });

  describe('copyto()', () => {
    it('validates copyto with scalar', () => {
      const jsArr = zeros([3]);
      copyto(jsArr, 5);

      const pyResult = runNumPy(`
arr = np.zeros(3)
np.copyto(arr, 5)
result = arr
      `);

      expect(jsArr.toArray()).toEqual(pyResult.value);
    });

    it('validates copyto with same-shape array', () => {
      const jsDst = zeros([3]);
      const jsSrc = array([1, 2, 3]);
      copyto(jsDst, jsSrc);

      const pyResult = runNumPy(`
dst = np.zeros(3)
src = np.array([1, 2, 3])
np.copyto(dst, src)
result = dst
      `);

      expect(jsDst.toArray()).toEqual(pyResult.value);
    });

    it('validates copyto broadcasts 1D to 2D', () => {
      const jsDst = zeros([3, 4]);
      const jsSrc = array([1, 2, 3, 4]);
      copyto(jsDst, jsSrc);

      const pyResult = runNumPy(`
dst = np.zeros((3, 4))
src = np.array([1, 2, 3, 4])
np.copyto(dst, src)
result = dst
      `);

      expect(jsDst.toArray()).toEqual(pyResult.value);
    });

    it('validates copyto broadcasts column vector', () => {
      const jsDst = zeros([3, 4]);
      const jsSrc = array([[1], [2], [3]]);
      copyto(jsDst, jsSrc);

      const pyResult = runNumPy(`
dst = np.zeros((3, 4))
src = np.array([[1], [2], [3]])
np.copyto(dst, src)
result = dst
      `);

      expect(jsDst.toArray()).toEqual(pyResult.value);
    });

    it('validates copyto modifies in place', () => {
      const jsDst = array([1, 2, 3]);
      const jsOriginal = jsDst;
      copyto(jsDst, array([9, 9, 9]));

      expect(jsDst).toBe(jsOriginal);
      expect(jsDst.toArray()).toEqual([9, 9, 9]);
    });
  });
});
