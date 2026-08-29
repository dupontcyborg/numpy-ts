/**
 * NumPy validation for operations on C-contiguous views with a non-zero offset.
 *
 * `isCContiguous` is computed from strides alone, so a sliced view — `a[1:]`, a
 * row range, a trailing window — is C-contiguous *and* starts partway into its
 * buffer. Every "fast path: contiguous, so hand back a view" branch has to carry
 * that offset, and `reshape`/`ravel` did not: they rebased on element 0 and
 * silently returned the wrong elements.
 *
 * `tensordot` made this reachable from the public API when it started reshaping
 * its operands to 2-D to delegate to `matmul`. The identity-permutation case —
 * the `axes: number` form, which is the common one — reshapes the caller's
 * storage directly, so any sliced operand produced wrong numbers with no error.
 *
 * The rest of the suite builds its operands with `array(...)` and `arange(...)`,
 * which are always offset 0, so nothing here was covered. These tests exist to
 * keep that blind spot closed.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, dot, matmul, ravel, reshape, tensordot } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: offset (sliced) views', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" pnpm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n',
      );
    }
  });

  describe('the views themselves are contiguous and offset', () => {
    it('a row slice is C-contiguous with a non-zero offset', () => {
      const a = arange(12).reshape([4, 3]).astype('float64');
      const sliced = a.slice('1:4');

      // If this ever stops holding, the tests below stop testing anything:
      // they would be exercising the copying path, not the view fast path.
      expect(sliced.storage.isCContiguous).toBe(true);
      expect(sliced.storage.offset).toBeGreaterThan(0);
    });
  });

  describe('reshape', () => {
    it('reshapes a 1-D slice without rebasing on element 0', () => {
      const jsResult = reshape(arange(8).astype('float64').slice('2:8'), [3, 2]);
      const pyResult = runNumPy(`
result = np.arange(8).astype(np.float64)[2:8].reshape(3, 2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('reshapes a row slice of a 2-D array', () => {
      const jsResult = reshape(arange(12).reshape([4, 3]).astype('float64').slice('1:4'), [9]);
      const pyResult = runNumPy(`
result = np.arange(12).reshape(4, 3).astype(np.float64)[1:4].reshape(9)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('reshapes an int64 slice', () => {
      const jsResult = reshape(arange(10).astype('int64').slice('4:10'), [2, 3]);
      const pyResult = runNumPy(`
result = np.arange(10).astype(np.int64)[4:10].reshape(2, 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.tolist()).toEqual([
        [4n, 5n, 6n],
        [7n, 8n, 9n],
      ]);
      expect(pyResult.value).toEqual([
        [4, 5, 6],
        [7, 8, 9],
      ]);
    });
  });

  describe('ravel', () => {
    it('ravels a row slice without rebasing on element 0', () => {
      const jsResult = ravel(arange(12).reshape([4, 3]).astype('float64').slice('1:4'));
      const pyResult = runNumPy(`
result = np.ravel(np.arange(12).reshape(4, 3).astype(np.float64)[1:4])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tensordot', () => {
    it('contracts a sliced operand (axes as a count, identity permutation)', () => {
      const a = arange(12).reshape([4, 3]).astype('float64').slice('1:4');
      const b = arange(6).reshape([3, 2]).astype('float64');

      const jsResult = tensordot(a, b, 1);
      const pyResult = runNumPy(`
a = np.arange(12).reshape(4, 3).astype(np.float64)[1:4]
b = np.arange(6).reshape(3, 2).astype(np.float64)
result = np.tensordot(a, b, 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      // The same contraction through matmul, which never had the bug — the two
      // must not disagree.
      expect(jsResult.toArray()).toEqual(matmul(a, b).toArray());
    });

    it('contracts when the *second* operand is sliced', () => {
      const jsResult = tensordot(
        arange(6).reshape([2, 3]).astype('float64'),
        arange(8).reshape([4, 2]).astype('float64').slice('1:4'),
        1,
      );
      const pyResult = runNumPy(`
a = np.arange(6).reshape(2, 3).astype(np.float64)
b = np.arange(8).reshape(4, 2).astype(np.float64)[1:4]
result = np.tensordot(a, b, 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('contracts with an explicit axes list on sliced operands', () => {
      const jsResult = tensordot(
        arange(24).reshape([4, 3, 2]).astype('float64').slice('1:4'),
        arange(24).reshape([4, 3, 2]).astype('float64').slice('1:4'),
        [
          [1, 2],
          [1, 2],
        ],
      );
      const pyResult = runNumPy(`
a = np.arange(24).reshape(4, 3, 2).astype(np.float64)[1:4]
result = np.tensordot(a, a, axes=([1, 2], [1, 2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('contracts to a scalar from sliced operands', () => {
      // Two 1-D operands fully contracted: this takes tensordot's scalar branch,
      // which walks with get() and was never affected — it is here so the two
      // branches are pinned to the same NumPy answer.
      const jsResult = tensordot(
        arange(8).astype('float64').slice('2:8'),
        arange(10).astype('float64').slice('4:10'),
        1,
      );
      const pyResult = runNumPy(`
a = np.arange(8).astype(np.float64)[2:8]
b = np.arange(10).astype(np.float64)[4:10]
result = float(np.tensordot(a, b, 1))
      `);

      // 2*4 + 3*5 + 4*6 + 5*7 + 6*8 + 7*9 = 193
      expect(Number(jsResult)).toBeCloseTo(pyResult.value as number, 10);
      expect(Number(jsResult)).toBeCloseTo(193, 10);
    });

    it('contracts sliced int64 operands exactly', () => {
      const jsResult = tensordot(
        arange(12).reshape([4, 3]).astype('int64').slice('1:4'),
        arange(6).reshape([3, 2]).astype('int64'),
        1,
      );
      const pyResult = runNumPy(`
a = np.arange(12).reshape(4, 3).astype(np.int64)[1:4]
b = np.arange(6).reshape(3, 2).astype(np.int64)
result = np.tensordot(a, b, 1)
      `);

      expect(jsResult.dtype).toBe('int64');
      expect(pyResult.dtype).toBe('int64');
      expect(jsResult.tolist()).toEqual(
        (pyResult.value as number[][]).map((row) => row.map((v) => BigInt(v))),
      );
    });
  });

  describe('dot / matmul stay correct on sliced operands', () => {
    it('matmul', () => {
      const jsResult = matmul(
        arange(12).reshape([4, 3]).astype('float64').slice('1:4'),
        arange(6).reshape([3, 2]).astype('float64'),
      );
      const pyResult = runNumPy(`
a = np.arange(12).reshape(4, 3).astype(np.float64)[1:4]
b = np.arange(6).reshape(3, 2).astype(np.float64)
result = a @ b
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('dot', () => {
      const jsResult = dot(
        arange(12).reshape([4, 3]).astype('float64').slice('1:4'),
        arange(6).reshape([3, 2]).astype('float64'),
      );
      const pyResult = runNumPy(`
a = np.arange(12).reshape(4, 3).astype(np.float64)[1:4]
b = np.arange(6).reshape(3, 2).astype(np.float64)
result = np.dot(a, b)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
