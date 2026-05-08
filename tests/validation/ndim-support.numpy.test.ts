/**
 * Comprehensive NDim support validation tests
 *
 * Validates that numpy-ts matches NumPy behavior for all implemented
 * functions across various input dimensionalities (0D through 4D+).
 *
 * Generated from scripts/ndim_requirements.py which systematically
 * discovers what NDims NumPy supports for each function.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import {
  add,
  all,
  any,
  argsort,
  array,
  average,
  clip,
  concatenate,
  cos,
  cross,
  cumprod,
  cumsum,
  diagonal,
  diff,
  dot,
  exp,
  fft,
  flip,
  inner,
  linalg,
  matmul,
  mean,
  median,
  moveaxis,
  multiply,
  reshape,
  roll,
  sin,
  sort,
  sqrt,
  stack,
  std,
  sum,
  swapaxes,
  tensordot,
  trace,
  transpose,
  where,
  zeros,
} from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

// Helper: create ascending float array of given shape
function arange(shape: number[]): ReturnType<typeof array> {
  const total = shape.reduce((a, b) => a * b, 1);
  const data: number[] = Array.from({ length: total }, (_, i) => i + 1);
  return array(data).reshape(shape);
}

describe('NumPy Validation: NDim Support', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   Activate conda: source ~/.zshrc && conda activate py313\n',
      );
    }
  });

  // ============================================================
  // matmul - THE KEY ISSUE: only 2D was supported
  // NumPy supports: 1D@1D=scalar, 2D@1D=1D, 1D@2D=1D, ND@ND=ND (batched)
  // ============================================================
  describe('matmul - NDim support', () => {
    it('1D @ 1D → scalar (inner product)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const jsResult = matmul(a, b);

      const pyResult = runNumPy(`
result = np.matmul(np.array([1, 2, 3], dtype=float), np.array([4, 5, 6], dtype=float))
      `);
      // 1D @ 1D = scalar in NumPy: 1*4 + 2*5 + 3*6 = 4+10+18 = 32
      const val = typeof jsResult === 'number' ? jsResult : (jsResult as any).toArray();
      expect(arraysClose(val, pyResult.value)).toBe(true);
    });

    it('2D @ 1D → 1D (matrix-vector product)', () => {
      const A = arange([2, 3]);
      const v = array([1, 2, 3]);
      const jsResult = matmul(A, v);

      const pyResult = runNumPy(`
A = np.arange(1, 7, dtype=float).reshape(2, 3)
v = np.array([1, 2, 3], dtype=float)
result = np.matmul(A, v)
      `);
      expect((jsResult as any).shape).toEqual([2]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('1D @ 2D → 1D (vector-matrix product)', () => {
      const v = array([1, 2]);
      const B = arange([2, 3]);
      const jsResult = matmul(v, B);

      const pyResult = runNumPy(`
v = np.array([1, 2], dtype=float)
B = np.arange(1, 7, dtype=float).reshape(2, 3)
result = np.matmul(v, B)
      `);
      expect((jsResult as any).shape).toEqual([3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('2D @ 2D → 2D (standard matrix multiply)', () => {
      const A = arange([2, 3]);
      const B = arange([3, 4]);
      const jsResult = matmul(A, B);

      const pyResult = runNumPy(`
A = np.arange(1, 7, dtype=float).reshape(2, 3)
B = np.arange(1, 13, dtype=float).reshape(3, 4)
result = np.matmul(A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('3D @ 3D → 3D (batched matrix multiply)', () => {
      const A = arange([2, 3, 4]);
      const B = arange([2, 4, 5]);
      const jsResult = matmul(A, B);

      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
B = np.arange(1, 41, dtype=float).reshape(2, 4, 5)
result = np.matmul(A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 5]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('4D @ 4D → 4D (batched matrix multiply)', () => {
      const A = arange([2, 2, 3, 4]);
      const B = arange([2, 2, 4, 5]);
      const jsResult = matmul(A, B);

      const pyResult = runNumPy(`
A = np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4)
B = np.arange(1, 81, dtype=float).reshape(2, 2, 4, 5)
result = np.matmul(A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 5]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('3D @ 2D → 3D (batch broadcast against single matrix)', () => {
      const A = arange([2, 3, 4]);
      const B = arange([4, 5]);
      const jsResult = matmul(A, B);

      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
B = np.arange(1, 21, dtype=float).reshape(4, 5)
result = np.matmul(A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 5]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('0D fails with error', () => {
      const s = array(5.0);
      expect(() => matmul(s, s)).toThrow();
    });
  });

  // ============================================================
  // trace - NumPy supports ND (sums along axis1/axis2)
  // ============================================================
  describe('trace - NDim support', () => {
    it('2D trace (base case)', () => {
      const A = arange([3, 3]);
      const jsResult = trace(A);
      const pyResult = runNumPy(`
result = np.trace(np.arange(1, 10, dtype=float).reshape(3, 3))
      `);
      expect(arraysClose(jsResult as number, pyResult.value)).toBe(true);
    });

    it('3D trace → 1D array (trace along last two axes)', () => {
      const A = arange([2, 3, 3]);
      const jsResult = trace(A);
      const pyResult = runNumPy(`
result = np.trace(np.arange(1, 19, dtype=float).reshape(2, 3, 3))
      `);
      expect((jsResult as any).shape ?? []).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray?.() ?? jsResult, pyResult.value)).toBe(true);
    });

    it('4D trace → 2D array', () => {
      const A = arange([2, 2, 3, 3]);
      const jsResult = trace(A);
      const pyResult = runNumPy(`
result = np.trace(np.arange(1, 37, dtype=float).reshape(2, 2, 3, 3))
      `);
      expect((jsResult as any).shape ?? []).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray?.() ?? jsResult, pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // linalg.det - NumPy supports batch (ND) operations
  // ============================================================
  describe('linalg.det - NDim support', () => {
    it('2D det (base case)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.det(A);
      const pyResult = runNumPy(`
result = np.linalg.det(np.array([[1, 2], [3, 4]], dtype=float))
      `);
      expect(arraysClose(jsResult as number, pyResult.value)).toBe(true);
    });

    it('3D det → 1D batch (det of each 2D slice)', () => {
      const A = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const jsResult = linalg.det(A);
      const pyResult = runNumPy(`
A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
result = np.linalg.det(A)
      `);
      expect((jsResult as any).shape ?? []).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray?.() ?? jsResult, pyResult.value)).toBe(true);
    });

    it('4D det → 2D batch', () => {
      const A = array([
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [2, 1],
            [5, 3],
          ],
        ],
        [
          [
            [4, 3],
            [2, 1],
          ],
          [
            [3, 2],
            [1, 4],
          ],
        ],
      ]);
      const jsResult = linalg.det(A);
      const pyResult = runNumPy(`
A = np.array([[[[1, 2], [3, 4]], [[2, 1], [5, 3]]], [[[4, 3], [2, 1]], [[3, 2], [1, 4]]]], dtype=float)
result = np.linalg.det(A)
      `);
      expect((jsResult as any).shape ?? []).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray?.() ?? jsResult, pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // linalg.inv - NumPy supports batch (ND) operations
  // ============================================================
  describe('linalg.inv - NDim support', () => {
    it('2D inv (base case)', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.inv(A);
      const pyResult = runNumPy(`
result = np.linalg.inv(np.array([[1, 2], [3, 4]], dtype=float))
      `);
      expect((jsResult as any).shape).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('3D inv → 3D batch (invert each 2D slice)', () => {
      const A = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [2, 1],
          [5, 3],
        ],
      ]);
      const jsResult = linalg.inv(A);
      const pyResult = runNumPy(`
A = np.array([[[1, 2], [3, 4]], [[2, 1], [5, 3]]], dtype=float)
result = np.linalg.inv(A)
      `);
      expect((jsResult as any).shape).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('4D inv → 4D batch', () => {
      const A = array([
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [2, 1],
            [5, 3],
          ],
        ],
        [
          [
            [4, 3],
            [2, 1],
          ],
          [
            [3, 2],
            [1, 4],
          ],
        ],
      ]);
      const jsResult = linalg.inv(A);
      const pyResult = runNumPy(`
A = np.array([[[[1, 2], [3, 4]], [[2, 1], [5, 3]]], [[[4, 3], [2, 1]], [[3, 2], [1, 4]]]], dtype=float)
result = np.linalg.inv(A)
      `);
      expect((jsResult as any).shape).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // Elementwise functions: all ndims 0D-4D (these should already work)
  // ============================================================
  describe('elementwise ops - NDim support', () => {
    for (const [name, fn] of [
      ['sin', sin],
      ['cos', cos],
      ['exp', exp],
      ['sqrt', sqrt],
    ] as const) {
      it(`${name}: 3D input`, () => {
        const A = arange([2, 3, 4]).divide(25);
        const jsResult = fn(A);
        const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4) / 25
result = np.${name}(A)
        `);
        expect((jsResult as any).shape).toEqual([2, 3, 4]);
        expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
      });

      it(`${name}: 4D input`, () => {
        const A = arange([2, 2, 3, 4]).divide(50);
        const jsResult = fn(A);
        const pyResult = runNumPy(`
A = np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4) / 50
result = np.${name}(A)
        `);
        expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
        expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
      });
    }
  });

  // ============================================================
  // Binary ops with broadcasting
  // ============================================================
  describe('binary ops - NDim broadcasting', () => {
    it('add: 1D + 3D broadcasts correctly', () => {
      const a = array([1, 2, 3, 4]);
      const b = arange([2, 3, 4]);
      const jsResult = add(a, b);
      const pyResult = runNumPy(`
a = np.array([1, 2, 3, 4], dtype=float)
b = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
result = np.add(a, b)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('multiply: 1D * 4D broadcasts correctly', () => {
      const a = array([1, 2, 3, 4]);
      const b = arange([2, 2, 3, 4]);
      const jsResult = multiply(a, b);
      const pyResult = runNumPy(`
a = np.array([1, 2, 3, 4], dtype=float)
b = np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4)
result = np.multiply(a, b)
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // Reduction ops: sum, mean, etc. with various axes on 3D/4D
  // ============================================================
  describe('reductions - NDim support', () => {
    it('sum: 3D no axis', () => {
      const A = arange([2, 3, 4]);
      // 1+2+...+24 = 24*25/2 = 300
      expect(arraysClose(sum(A) as number, 300)).toBe(true);
    });

    it('sum: 3D axis=1', () => {
      const A = arange([2, 3, 4]);
      const jsResult = sum(A, 1);
      const pyResult = runNumPy(`
result = np.sum(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=1)
      `);
      expect((jsResult as any).shape).toEqual([2, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('sum: 4D axis=2', () => {
      const A = arange([2, 2, 3, 4]);
      const jsResult = sum(A, 2);
      const pyResult = runNumPy(`
result = np.sum(np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4), axis=2)
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('mean: 3D axis=0', () => {
      const A = arange([2, 3, 4]);
      const jsResult = mean(A, 0);
      const pyResult = runNumPy(`
result = np.mean(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=0)
      `);
      expect((jsResult as any).shape).toEqual([3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('std: 3D axis=-1', () => {
      const A = arange([2, 3, 4]);
      const jsResult = std(A, -1);
      const pyResult = runNumPy(`
result = np.std(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=-1)
      `);
      expect((jsResult as any).shape).toEqual([2, 3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('all: 3D', () => {
      const A = array([
        [
          [true, false],
          [true, true],
        ],
        [
          [true, true],
          [false, true],
        ],
      ]);
      const jsResult = all(A);
      const pyResult = runNumPy(`
A = np.array([[[True, False], [True, True]], [[True, True], [False, True]]])
result = np.all(A)
      `);
      expect(arraysClose(jsResult as any, pyResult.value)).toBe(true);
    });

    it('any: 4D zero array', () => {
      const A = zeros([2, 2, 3, 4]);
      const jsResult = any(A);
      expect(jsResult).toBe(false);
    });
  });

  // ============================================================
  // Shape operations with 3D/4D arrays
  // ============================================================
  describe('shape ops - NDim support', () => {
    it('flip: 4D', () => {
      const A = arange([2, 2, 3, 4]);
      const jsResult = flip(A);
      const pyResult = runNumPy(`
result = np.flip(np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4))
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('swapaxes: 4D', () => {
      const A = arange([2, 3, 4, 5]);
      const jsResult = swapaxes(A, 1, 3);
      const pyResult = runNumPy(`
result = np.swapaxes(np.arange(1, 121, dtype=float).reshape(2, 3, 4, 5), 1, 3)
      `);
      expect((jsResult as any).shape).toEqual([2, 5, 4, 3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('transpose: 4D', () => {
      const A = arange([2, 3, 4, 5]);
      const jsResult = transpose(A);
      const pyResult = runNumPy(`
result = np.transpose(np.arange(1, 121, dtype=float).reshape(2, 3, 4, 5))
      `);
      expect((jsResult as any).shape).toEqual([5, 4, 3, 2]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('roll: 3D with axis', () => {
      const A = arange([2, 3, 4]);
      const jsResult = roll(A, 2, 1); // axis is 3rd arg
      const pyResult = runNumPy(`
result = np.roll(np.arange(1, 25, dtype=float).reshape(2, 3, 4), 2, axis=1)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('concatenate: 4D', () => {
      const A = arange([2, 2, 3, 4]);
      const jsResult = concatenate([A, A], 0); // axis is 2nd arg
      const pyResult = runNumPy(`
A = np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4)
result = np.concatenate([A, A], axis=0)
      `);
      expect((jsResult as any).shape).toEqual([4, 2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('stack: 3D → 4D', () => {
      const A = arange([2, 3, 4]);
      const jsResult = stack([A, A]);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('reshape: 3D → 4D', () => {
      const A = arange([2, 3, 4]);
      const jsResult = reshape(A, [2, 3, 2, 2]);
      const pyResult = runNumPy(`
result = np.arange(1, 25, dtype=float).reshape(2, 3, 4).reshape(2, 3, 2, 2)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 2, 2]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('moveaxis: 4D', () => {
      const A = arange([2, 3, 4, 5]);
      const jsResult = moveaxis(A, 0, -1);
      const pyResult = runNumPy(`
result = np.moveaxis(np.arange(1, 121, dtype=float).reshape(2, 3, 4, 5), 0, -1)
      `);
      expect((jsResult as any).shape).toEqual([3, 4, 5, 2]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // diagonal - 3D and 4D
  // ============================================================
  describe('diagonal - NDim support', () => {
    it('3D diagonal → 2D', () => {
      const A = arange([2, 3, 4]);
      const jsResult = diagonal(A, 0, 1, 2);
      const pyResult = runNumPy(`
result = np.diagonal(np.arange(1, 25, dtype=float).reshape(2, 3, 4), 0, 1, 2)
      `);
      expect((jsResult as any).shape).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('4D diagonal → 3D', () => {
      const A = arange([2, 3, 4, 4]);
      const jsResult = diagonal(A, 0, 2, 3);
      const pyResult = runNumPy(`
result = np.diagonal(np.arange(1, 97, dtype=float).reshape(2, 3, 4, 4), 0, 2, 3)
      `);
      expect((jsResult as any).shape).toEqual(pyResult.shape);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // sort, argsort: 3D/4D
  // ============================================================
  describe('sort/argsort - NDim support', () => {
    it('sort: 3D (sorts along last axis by default)', () => {
      const A = arange([2, 3, 4]).multiply(-1).add(25);
      const jsResult = sort(A);
      const pyResult = runNumPy(`
A = (np.arange(1, 25, dtype=float) * -1 + 25).reshape(2, 3, 4)
result = np.sort(A)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('argsort: 4D', () => {
      const A = arange([2, 2, 3, 4]).multiply(-1).add(49);
      const jsResult = argsort(A);
      const pyResult = runNumPy(`
A = (np.arange(1, 49, dtype=float) * -1 + 49).reshape(2, 2, 3, 4)
result = np.argsort(A)
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // cumsum/cumprod: 3D with axis
  // ============================================================
  describe('cumsum/cumprod - NDim support', () => {
    it('cumsum: 3D axis=1', () => {
      const A = arange([2, 3, 4]);
      const jsResult = cumsum(A, 1); // axis is 2nd arg
      const pyResult = runNumPy(`
result = np.cumsum(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=1)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('cumprod: 3D axis=0', () => {
      const A = arange([2, 3, 4]);
      const jsResult = cumprod(A, 0); // axis is 2nd arg
      const pyResult = runNumPy(`
result = np.cumprod(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=0)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // diff: 3D
  // ============================================================
  describe('diff - NDim support', () => {
    it('diff: 3D axis=0', () => {
      const A = arange([3, 3, 4]);
      const jsResult = diff(A, 1, 0); // n=1, axis=0
      const pyResult = runNumPy(`
result = np.diff(np.arange(1, 37, dtype=float).reshape(3, 3, 4), axis=0)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('diff: 3D axis=2', () => {
      const A = arange([2, 3, 4]);
      const jsResult = diff(A, 1, 2); // n=1, axis=2
      const pyResult = runNumPy(`
result = np.diff(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=2)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // clip: 3D/4D
  // ============================================================
  describe('clip - NDim support', () => {
    it('clip: 3D', () => {
      const A = arange([2, 3, 4]);
      const jsResult = clip(A, 5, 15);
      const pyResult = runNumPy(`
result = np.clip(np.arange(1, 25, dtype=float).reshape(2, 3, 4), 5, 15)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('clip: 4D', () => {
      const A = arange([2, 2, 3, 4]);
      const jsResult = clip(A, 10, 30);
      const pyResult = runNumPy(`
result = np.clip(np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4), 10, 30)
      `);
      expect((jsResult as any).shape).toEqual([2, 2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // where: 3D
  // ============================================================
  describe('where - NDim support', () => {
    it('where: 3D', () => {
      const A = arange([2, 3, 4]);
      const B = arange([2, 3, 4]).multiply(-1);
      // create boolean condition: even elements
      const cond = array([
        [
          [false, true, false, true],
          [false, true, false, true],
          [false, true, false, true],
        ],
        [
          [false, true, false, true],
          [false, true, false, true],
          [false, true, false, true],
        ],
      ]);
      const jsResult = where(cond, A, B);
      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
B = A * -1
cond = np.arange(1, 25).reshape(2, 3, 4) % 2 == 0
result = np.where(cond, A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // median/average: 3D/4D
  // ============================================================
  describe('median/average - NDim support', () => {
    it('median: 3D axis=1', () => {
      const A = arange([2, 3, 4]);
      const jsResult = median(A, 1); // axis is 2nd arg
      const pyResult = runNumPy(`
result = np.median(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=1)
      `);
      expect((jsResult as any).shape).toEqual([2, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('average: 3D axis=0', () => {
      const A = arange([2, 3, 4]);
      const jsResult = average(A, 0); // axis is 2nd arg
      const pyResult = runNumPy(`
result = np.average(np.arange(1, 25, dtype=float).reshape(2, 3, 4), axis=0)
      `);
      expect((jsResult as any).shape).toEqual([3, 4]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // cross: batched 2D/3D
  // ============================================================
  describe('cross - NDim support', () => {
    it('cross: 2D batched (3-vectors)', () => {
      const A = arange([3, 3]);
      const B = arange([3, 3]).add(1);
      const jsResult = cross(A, B);
      const pyResult = runNumPy(`
A = np.arange(1, 10, dtype=float).reshape(3, 3)
B = np.arange(2, 11, dtype=float).reshape(3, 3)
result = np.cross(A, B)
      `);
      expect((jsResult as any).shape).toEqual([3, 3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });

    it('cross: 3D batched', () => {
      const A = arange([2, 3, 3]);
      const B = arange([2, 3, 3]).add(1);
      const jsResult = cross(A, B);
      const pyResult = runNumPy(`
A = np.arange(1, 19, dtype=float).reshape(2, 3, 3)
B = np.arange(2, 20, dtype=float).reshape(2, 3, 3)
result = np.cross(A, B)
      `);
      expect((jsResult as any).shape).toEqual([2, 3, 3]);
      expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // dot: general ND cases
  // ============================================================
  describe('dot - NDim support', () => {
    it('dot: 3D · 2D', () => {
      const A = arange([2, 3, 4]);
      const B = arange([4, 5]);
      const jsResult = dot(A, B) as any;
      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
B = np.arange(1, 21, dtype=float).reshape(4, 5)
result = np.dot(A, B)
      `);
      expect(jsResult.shape).toEqual([2, 3, 5]);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('dot: 4D · 2D', () => {
      const A = arange([2, 2, 3, 4]);
      const B = arange([4, 5]);
      const jsResult = dot(A, B) as any;
      const pyResult = runNumPy(`
A = np.arange(1, 49, dtype=float).reshape(2, 2, 3, 4)
B = np.arange(1, 21, dtype=float).reshape(4, 5)
result = np.dot(A, B)
      `);
      expect(jsResult.shape).toEqual([2, 2, 3, 5]);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // inner: ND cases
  // ============================================================
  describe('inner - NDim support', () => {
    it('inner: 3D · 1D', () => {
      const A = arange([2, 3, 4]);
      const b = array([1, 2, 3, 4]);
      const jsResult = inner(A, b) as any;
      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
b = np.array([1, 2, 3, 4], dtype=float)
result = np.inner(A, b)
      `);
      expect(jsResult.shape).toEqual([2, 3]);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('inner: 2D · 2D', () => {
      const A = arange([2, 3]);
      const B = arange([4, 3]);
      const jsResult = inner(A, B) as any;
      const pyResult = runNumPy(`
A = np.arange(1, 7, dtype=float).reshape(2, 3)
B = np.arange(1, 13, dtype=float).reshape(4, 3)
result = np.inner(A, B)
      `);
      expect(jsResult.shape).toEqual([2, 4]);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // tensordot: ND
  // ============================================================
  describe('tensordot - NDim support', () => {
    it('tensordot: 3D · 2D (axes=1)', () => {
      const A = arange([2, 3, 4]);
      const B = arange([4, 5]);
      const jsResult = tensordot(A, B, 1) as any;
      const pyResult = runNumPy(`
A = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
B = np.arange(1, 21, dtype=float).reshape(4, 5)
result = np.tensordot(A, B, 1)
      `);
      expect(jsResult.shape).toEqual([2, 3, 5]);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  // ============================================================
  // FFT functions: 2D/3D
  // ============================================================
  describe('fft - NDim support', () => {
    it('fft.fft: 2D (operates on last axis)', () => {
      const A = arange([3, 4]);
      const jsResult = fft.fft(A);
      // FFT returns complex output - just check shape
      expect((jsResult as any).shape).toEqual([3, 4]);
    });

    it('fft.fft: 3D (operates on last axis)', () => {
      const A = arange([2, 3, 4]);
      const jsResult = fft.fft(A);
      expect((jsResult as any).shape).toEqual([2, 3, 4]);
    });
  });

  // ============================================================
  // MAX_NDIM enforcement - NumPy supports max 64 dims
  // ============================================================
  describe('MAX_NDIM enforcement', () => {
    it('accepts 64-dimensional arrays (NumPy maximum)', () => {
      const shape = Array(64).fill(1) as number[];
      expect(() => zeros(shape)).not.toThrow();
    });

    it('rejects arrays with more than 64 dimensions', () => {
      const shape = Array(65).fill(1) as number[];
      expect(() => zeros(shape)).toThrow();
    });

    it('64-dim array has correct ndim', () => {
      const shape = Array(64).fill(1) as number[];
      const a = zeros(shape);
      expect(a.ndim).toBe(64);
    });
  });
});
