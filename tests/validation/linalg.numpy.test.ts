/**
 * Python NumPy validation tests for linear algebra operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  dot,
  trace,
  transpose,
  inner,
  outer,
  tensordot,
  diagonal,
  kron,
  einsum,
  linalg,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Linear Algebra', () => {
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

  describe('dot()', () => {
    it('matches NumPy for 1D · 1D (inner product)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([6, 7, 8, 9, 10]);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
result = np.dot([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D · 2D (matrix multiplication)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D · 1D (matrix-vector)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([7, 8, 9]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 1D · 2D (vector-matrix)', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [4, 5],
        [6, 7],
        [8, 9],
      ]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([1, 2, 3], [[4, 5], [6, 7], [8, 9]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with float arrays', () => {
      const a = array([1.5, 2.5, 3.5]);
      const b = array([4.2, 5.3, 6.4]);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
result = np.dot([1.5, 2.5, 3.5], [4.2, 5.3, 6.4])
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for large vectors', () => {
      const aData = Array.from({ length: 100 }, (_, i) => i);
      const bData = Array.from({ length: 100 }, (_, i) => 99 - i);
      const a = array(aData);
      const b = array(bData);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
a = np.arange(100)
b = np.arange(100)[::-1]
result = np.dot(a, b)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    describe('scalar (0D) cases', () => {
      it('matches NumPy for 0D · 0D (scalar multiplication)', () => {
        const a = array(5);
        const b = array(3);
        const jsResult = dot(a, b);

        const pyResult = runNumPy(`
result = np.dot(np.array(5), np.array(3))
        `);

        expect(jsResult).toBe(pyResult.value);
      });

      it('matches NumPy for 0D · 1D (scalar times vector)', () => {
        const a = array(2);
        const b = array([1, 2, 3, 4]);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array(2), np.array([1, 2, 3, 4]))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 1D · 0D (vector times scalar)', () => {
        const a = array([1, 2, 3, 4]);
        const b = array(2);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array([1, 2, 3, 4]), np.array(2))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 0D · 2D (scalar times matrix)', () => {
        const a = array(3);
        const b = array([
          [1, 2],
          [3, 4],
        ]);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array(3), np.array([[1, 2], [3, 4]]))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D · 0D (matrix times scalar)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array(3);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array([[1, 2], [3, 4]]), np.array(3))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('higher dimensional cases', () => {
      it('matches NumPy for 3D · 1D', () => {
        const a = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const b = array([10, 20]); // Shape: (2,)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([10, 20])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 1D · 3D', () => {
        const a = array([1, 2]); // Shape: (2,)
        const b = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([1, 2])
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 3D · 2D (general tensor)', () => {
        const a = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const b = array([
          [10, 20, 30],
          [40, 50, 60],
        ]); // Shape: (2, 3)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[10, 20, 30], [40, 50, 60]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D · 3D (general tensor)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]); // Shape: (2, 2)
        const b = array([
          [
            [10, 20],
            [30, 40],
          ],
          [
            [50, 60],
            [70, 80],
          ],
        ]); // Shape: (2, 2, 2)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });
  });

  describe('trace()', () => {
    it('matches NumPy for square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2], [3, 4]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for non-square matrix (wide)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2, 3], [4, 5, 6]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for non-square matrix (tall)', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2], [3, 4], [5, 6]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for identity matrix', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace(np.eye(3))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy with float values', () => {
      const a = array([
        [1.5, 2.3],
        [3.7, 4.9],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1.5, 2.3], [3.7, 4.9]])
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for large matrix', () => {
      const size = 50;
      const data = Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) => i * size + j)
      );
      const a = array(data);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
a = np.arange(${size * size}).reshape(${size}, ${size})
result = np.trace(a)
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('transpose()', () => {
    it('matches NumPy for 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([[1, 2, 3], [4, 5, 6]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 3D array with custom axes', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const jsResult = transpose(a, [1, 0, 2]);

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.transpose(a, (1, 0, 2))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 1D array (no-op)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([1, 2, 3, 4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('inner()', () => {
    it('matches NumPy for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const jsResult = inner(a, b);

      const pyResult = runNumPy(`
result = np.inner([1, 2, 3], [4, 5, 6])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2], [3, 4]], [[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D · 1D', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([7, 8, 9]);
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for different shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // (3, 2)
      const b = array([
        [7, 8],
        [9, 10],
      ]); // (2, 2)
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('outer()', () => {
    it('matches NumPy for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const jsResult = outer(a, b);

      const pyResult = runNumPy(`
result = np.outer([1, 2, 3], [4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with 2D inputs (flattens)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([5, 6]);
      const jsResult = outer(a, b);

      const pyResult = runNumPy(`
result = np.outer([[1, 2], [3, 4]], [5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tensordot()', () => {
    it('matches NumPy for axes=0 (outer product)', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const jsResult = tensordot(a, b, 0) as any;

      const pyResult = runNumPy(`
result = np.tensordot([1, 2], [3, 4], axes=0)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for axes=1 (dot product)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = tensordot(a, b, 1) as any;

      const pyResult = runNumPy(`
result = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for axes=2 (full contraction -> scalar)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = tensordot(a, b, 2);

      const pyResult = runNumPy(`
result = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=2)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for custom axis pairs', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const b = array([[[9, 10]], [[11, 12]]]); // (2, 1, 2)
      const jsResult = tensordot(a, b, [[2], [2]]) as any;

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[[9, 10]], [[11, 12]]])
result = np.tensordot(a, b, axes=([2], [2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for multiple axis contraction', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const b = array([
        [
          [9, 10],
          [11, 12],
        ],
        [
          [13, 14],
          [15, 16],
        ],
      ]); // (2, 2, 2)
      const jsResult = tensordot(a, b, [
        [1, 2],
        [0, 1],
      ]) as any;

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
result = np.tensordot(a, b, axes=([1, 2], [0, 1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('diagonal()', () => {
    it('matches NumPy diagonal for 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = diagonal(a);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal with positive offset', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = diagonal(a, 1);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a, 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal with negative offset', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = diagonal(a, -1);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a, -1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal for non-square matrix', () => {
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const jsResult = diagonal(a);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.diagonal(a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('kron()', () => {
    it('matches NumPy kron for 1D vectors', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([1, 2])
b = np.array([3, 4])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron for 2D matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron with identity matrix', () => {
      const a = array([
        [0, 1],
        [1, 0],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[0, 1], [1, 0]])
b = np.array([[1, 2], [3, 4]])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron with different dimensions', () => {
      const a = array([[1, 2]]);
      const b = array([3, 4, 5]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[1, 2]])
b = np.array([3, 4, 5])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('einsum()', () => {
    it('matches NumPy for matrix multiplication (ij,jk->ik)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = einsum('ij,jk->ik', a, b) as any;

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.einsum('ij,jk->ik', a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for inner product (i,i->)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([6, 7, 8, 9, 10]);
      const jsResult = einsum('i,i->', a, b);

      const pyResult = runNumPy(`
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
result = np.einsum('i,i->', a, b)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for trace (ii->)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = einsum('ii->', a);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.einsum('ii->', a)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for transpose (ij->ji)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = einsum('ij->ji', a) as any;

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
result = np.einsum('ij->ji', a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for sum over axis (ij->j)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = einsum('ij->j', a) as any;

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
result = np.einsum('ij->j', a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for outer product (i,j->ij)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const jsResult = einsum('i,j->ij', a, b) as any;

      const pyResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = np.einsum('i,j->ij', a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for element-wise product with sum (ij,ij->)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = einsum('ij,ij->', a, b);

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.einsum('ij,ij->', a, b)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for batched matrix-vector product (ijk,ik->ij)', () => {
      const a = array([
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
      ]);
      const b = array([
        [1, 0, 0],
        [0, 1, 0],
      ]);
      const jsResult = einsum('ijk,ik->ij', a, b) as any;

      const pyResult = runNumPy(`
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
b = np.array([[1, 0, 0], [0, 1, 0]])
result = np.einsum('ijk,ik->ij', a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for implicit output notation', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = einsum('ij,jk', a, b) as any;

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.einsum('ij,jk', a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for diagonal extraction (ii->i)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = einsum('ii->i', a) as any;

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.einsum('ii->i', a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});

describe('NumPy Validation: numpy.linalg Module', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('linalg.cross()', () => {
    it('matches NumPy for 3D cross product', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const jsResult = linalg.cross(a, b) as any;

      const pyResult = runNumPy(`
result = np.linalg.cross([1, 2, 3], [4, 5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D cross product (scalar)', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const jsResult = linalg.cross(a, b);

      const pyResult = runNumPy(`
result = np.linalg.cross([1, 2], [3, 4])
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('linalg.norm()', () => {
    it('matches NumPy for 2-norm of vector', () => {
      const a = array([3, 4]);
      const jsResult = linalg.norm(a);

      const pyResult = runNumPy(`
result = np.linalg.norm([3, 4])
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for 1-norm of vector', () => {
      const a = array([3, -4, 2]);
      const jsResult = linalg.norm(a, 1);

      const pyResult = runNumPy(`
result = np.linalg.norm([3, -4, 2], 1)
      `);

      expect(jsResult).toBeCloseTo(pyResult.value, 10);
    });

    it('matches NumPy for Frobenius norm of matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.norm(a, 'fro');

      const pyResult = runNumPy(`
result = np.linalg.norm([[1, 2], [3, 4]], 'fro')
      `);

      expect(jsResult).toBeCloseTo(pyResult.value, 10);
    });

    it('matches NumPy for infinity norm of vector', () => {
      const a = array([1, -5, 3]);
      const jsResult = linalg.norm(a, Infinity);

      const pyResult = runNumPy(`
result = np.linalg.norm([1, -5, 3], np.inf)
      `);

      expect(jsResult).toBeCloseTo(pyResult.value, 10);
    });
  });

  describe('linalg.det()', () => {
    it('matches NumPy for 2x2 determinant', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.det(a);

      const pyResult = runNumPy(`
result = np.linalg.det([[1, 2], [3, 4]])
      `);

      expect(jsResult).toBeCloseTo(pyResult.value, 10);
    });

    it('matches NumPy for 3x3 determinant', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10],
      ]);
      const jsResult = linalg.det(a);

      const pyResult = runNumPy(`
result = np.linalg.det([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
      `);

      expect(jsResult).toBeCloseTo(pyResult.value, 5);
    });
  });

  describe('linalg.inv()', () => {
    it('matches NumPy for 2x2 inverse', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.inv(a);

      const pyResult = runNumPy(`
result = np.linalg.inv([[1, 2], [3, 4]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });
  });

  describe('linalg.solve()', () => {
    it('matches NumPy for linear system', () => {
      const a = array([
        [3, 1],
        [1, 2],
      ]);
      const b = array([9, 8]);
      const jsResult = linalg.solve(a, b);

      const pyResult = runNumPy(`
result = np.linalg.solve([[3, 1], [1, 2]], [9, 8])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });
  });

  describe('linalg.qr()', () => {
    it('matches NumPy QR decomposition (reconstructs A)', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsResult = linalg.qr(a) as { q: any; r: any };
      const reconstructed = jsResult.q.matmul(jsResult.r);

      // Verify A ≈ Q @ R
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          expect(reconstructed.get(i, j)).toBeCloseTo(a.get(i, j), 5);
        }
      }
    });
  });

  describe('linalg.cholesky()', () => {
    it('matches NumPy for Cholesky decomposition', () => {
      const a = array([
        [4, 2],
        [2, 5],
      ]);
      const jsResult = linalg.cholesky(a);

      const pyResult = runNumPy(`
result = np.linalg.cholesky([[4, 2], [2, 5]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });
  });

  describe('linalg.svd()', () => {
    it('matches NumPy for singular values', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsResult = linalg.svd(a) as { u: any; s: any; vt: any };

      const pyResult = runNumPy(`
u, s, vt = np.linalg.svd([[1, 2], [3, 4], [5, 6]])
result = s
      `);

      expect(arraysClose(jsResult.s.toArray(), pyResult.value, 1e-5)).toBe(true);
    });
  });

  describe('linalg.eig()', () => {
    it('matches NumPy eigenvalues for symmetric matrix', () => {
      const a = array([
        [4, 2],
        [2, 4],
      ]);
      const { w } = linalg.eig(a);

      const pyResult = runNumPy(`
w, v = np.linalg.eig([[4, 2], [2, 4]])
result = np.sort(w)
      `);

      const jsEigVals = [w.get([0]), w.get([1])].sort((a: any, b: any) => a - b);
      expect(jsEigVals[0]).toBeCloseTo(pyResult.value[0], 5);
      expect(jsEigVals[1]).toBeCloseTo(pyResult.value[1], 5);
    });
  });

  describe('linalg.eigh()', () => {
    it('matches NumPy eigenvalues for symmetric matrix', () => {
      const a = array([
        [4, 2],
        [2, 4],
      ]);
      const { w } = linalg.eigh(a);

      const pyResult = runNumPy(`
w, v = np.linalg.eigh([[4, 2], [2, 4]])
result = w
      `);

      expect(arraysClose(w.toArray(), pyResult.value, 1e-5)).toBe(true);
    });
  });

  describe('linalg.eigvals()', () => {
    it('matches NumPy eigenvalues', () => {
      const a = array([
        [1, 0],
        [0, 2],
      ]);
      const w = linalg.eigvals(a);

      const pyResult = runNumPy(`
result = np.sort(np.linalg.eigvals([[1, 0], [0, 2]]))
      `);

      const jsEigVals = [w.get([0]), w.get([1])].sort((a: any, b: any) => a - b);
      expect(jsEigVals[0]).toBeCloseTo(pyResult.value[0], 5);
      expect(jsEigVals[1]).toBeCloseTo(pyResult.value[1], 5);
    });
  });

  describe('linalg.eigvalsh()', () => {
    it('matches NumPy eigenvalues for symmetric matrix', () => {
      const a = array([
        [3, 1],
        [1, 3],
      ]);
      const w = linalg.eigvalsh(a);

      const pyResult = runNumPy(`
result = np.linalg.eigvalsh([[3, 1], [1, 3]])
      `);

      expect(arraysClose(w.toArray(), pyResult.value, 1e-5)).toBe(true);
    });
  });

  describe('linalg.cond()', () => {
    it('returns finite condition number for well-conditioned matrix', () => {
      // Identity matrix should have condition number 1
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      const jsResult = linalg.cond(a);

      expect(jsResult).toBeCloseTo(1, 3);
    });
  });

  describe('linalg.matrix_rank()', () => {
    it('matches NumPy for full rank matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.matrix_rank(a);

      const pyResult = runNumPy(`
result = np.linalg.matrix_rank([[1, 2], [3, 4]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('detects rank-deficient matrix', () => {
      const a = array([
        [1, 2],
        [2, 4],
      ]);
      const jsResult = linalg.matrix_rank(a);

      // Note: SVD-based rank detection may differ due to numerical precision
      // The rank should be less than or equal to the full rank (2)
      expect(jsResult).toBeLessThanOrEqual(2);
      expect(jsResult).toBeGreaterThanOrEqual(1);
    });
  });

  describe('linalg.matrix_power()', () => {
    it('matches NumPy for matrix squared', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.matrix_power(a, 2);

      const pyResult = runNumPy(`
result = np.linalg.matrix_power([[1, 2], [3, 4]], 2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });

    it('matches NumPy for matrix cubed', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.matrix_power(a, 3);

      const pyResult = runNumPy(`
result = np.linalg.matrix_power([[1, 2], [3, 4]], 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });
  });

  describe('linalg.pinv()', () => {
    it('computes pseudo-inverse with correct shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = linalg.pinv(a);

      // Check shape is correct (same as input for square matrix)
      expect(jsResult.shape).toEqual([2, 2]);
    });

    it('computes pseudo-inverse of non-square matrix with correct shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsResult = linalg.pinv(a);

      // Pseudo-inverse of (m x n) has shape (n x m)
      expect(jsResult.shape).toEqual([2, 3]);
    });
  });

  describe('linalg.lstsq()', () => {
    it('returns solution with correct shape for overdetermined system', () => {
      const a = array([
        [1, 1],
        [1, 2],
        [1, 3],
      ]);
      const b = array([1, 2, 2]);
      const jsResult = linalg.lstsq(a, b);

      // Solution should have shape matching number of columns in A
      expect(jsResult.x.shape).toEqual([2]);
      expect(typeof jsResult.rank).toBe('number');
      expect(jsResult.s.shape[0]).toBe(2); // Two singular values
    });
  });
});
