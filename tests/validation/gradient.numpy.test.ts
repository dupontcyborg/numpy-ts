/**
 * Python NumPy validation tests for gradient operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, diff, ediff1d, gradient, cross } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Gradient Operations', () => {
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

  describe('diff', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = diff(array([1, 2, 4, 7, 0]));
      const pyResult = runNumPy(`
result = np.diff(np.array([1, 2, 4, 7, 0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for n=2', () => {
      const jsResult = diff(array([1, 2, 4, 7, 0]), 2);
      const pyResult = runNumPy(`
result = np.diff(np.array([1, 2, 4, 7, 0]), n=2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for n=3', () => {
      const jsResult = diff(array([1, 2, 4, 7, 0]), 3);
      const pyResult = runNumPy(`
result = np.diff(np.array([1, 2, 4, 7, 0]), n=3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array along axis 0', () => {
      const jsResult = diff(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1,
        0
      );
      const pyResult = runNumPy(`
result = np.diff(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array along axis 1', () => {
      const jsResult = diff(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1,
        1
      );
      const pyResult = runNumPy(`
result = np.diff(np.array([[1, 2, 3], [4, 5, 6]]), axis=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative axis', () => {
      const jsResult = diff(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1,
        -1
      );
      const pyResult = runNumPy(`
result = np.diff(np.array([[1, 2, 3], [4, 5, 6]]), axis=-1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('ediff1d', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = ediff1d(array([1, 2, 4, 7, 0]));
      const pyResult = runNumPy(`
result = np.ediff1d(np.array([1, 2, 4, 7, 0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array (flattened)', () => {
      const jsResult = ediff1d(
        array([
          [1, 2],
          [3, 4],
        ])
      );
      const pyResult = runNumPy(`
result = np.ediff1d(np.array([[1, 2], [3, 4]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with to_begin', () => {
      const jsResult = ediff1d(array([1, 2, 4]), null, [0, 0]);
      const pyResult = runNumPy(`
result = np.ediff1d(np.array([1, 2, 4]), to_begin=[0, 0])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with to_end', () => {
      const jsResult = ediff1d(array([1, 2, 4]), [9, 9]);
      const pyResult = runNumPy(`
result = np.ediff1d(np.array([1, 2, 4]), to_end=[9, 9])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('gradient', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = gradient(array([1, 2, 4, 7, 11])) as any;
      const pyResult = runNumPy(`
result = np.gradient(np.array([1, 2, 4, 7, 11]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with custom spacing', () => {
      const jsResult = gradient(array([1, 2, 4, 7, 11]), 2) as any;
      const pyResult = runNumPy(`
result = np.gradient(np.array([1, 2, 4, 7, 11]), 2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array axis 0', () => {
      const jsResult = gradient(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1,
        0
      ) as any;
      const pyResult = runNumPy(`
result = np.gradient(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array axis 1', () => {
      const jsResult = gradient(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1,
        1
      ) as any;
      const pyResult = runNumPy(`
result = np.gradient(np.array([[1, 2, 3], [4, 5, 6]]), axis=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for specific axis', () => {
      const jsResult = gradient(
        array([
          [1, 2, 4],
          [4, 8, 12],
        ]),
        1,
        1
      ) as any;
      const pyResult = runNumPy(`
result = np.gradient(np.array([[1, 2, 4], [4, 8, 12]]), axis=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cross', () => {
    it('matches NumPy for 3D vectors', () => {
      const jsResult = cross(array([1, 0, 0]), array([0, 1, 0]));
      const pyResult = runNumPy(`
result = np.cross(np.array([1, 0, 0]), np.array([0, 1, 0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 3D vectors - example 2', () => {
      const jsResult = cross(array([1, 2, 3]), array([4, 5, 6]));
      const pyResult = runNumPy(`
result = np.cross(np.array([1, 2, 3]), np.array([4, 5, 6]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D vectors (scalar result)', () => {
      const jsResult = cross(array([1, 2]), array([3, 4]));
      const pyResult = runNumPy(`
result = np.cross(np.array([1, 2]), np.array([3, 4]))
      `);

      // NumPy returns a scalar, we return a 0-d array
      const jsValue = jsResult.shape.length === 0 ? jsResult.get([]) : jsResult.toArray();
      expect(arraysClose([jsValue], [pyResult.value])).toBe(true);
    });

    it('matches NumPy for a x a = 0', () => {
      const jsResult = cross(array([1, 2, 3]), array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.cross(np.array([1, 2, 3]), np.array([1, 2, 3]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for multiple vectors (2D arrays)', () => {
      const jsResult = cross(
        array([
          [1, 0, 0],
          [0, 1, 0],
        ]),
        array([
          [0, 1, 0],
          [0, 0, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.cross(np.array([[1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 0], [0, 0, 1]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
