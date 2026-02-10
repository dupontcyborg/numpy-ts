/**
 * Python NumPy validation tests for broadcasting
 *
 * These tests validate that our broadcasting implementation matches Python NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, broadcast_shapes } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Broadcasting', () => {
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
  });

  describe('add with broadcasting', () => {
    it('broadcasts (3, 4) + (4,) correctly', () => {
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const b = array([10, 20, 30, 40]);
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.array([10, 20, 30, 40])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (3, 1) + (1, 4) correctly', () => {
      const a = array([[1], [2], [3]]);
      const b = array([[10, 20, 30, 40]]);
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([[1], [2], [3]])
b = np.array([[10, 20, 30, 40]])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (3, 4) + (1,) correctly', () => {
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const b = array([100]);
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.array([100])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (1,) + (3, 4) correctly', () => {
      const a = array([5]);
      const b = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([5])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('subtract with broadcasting', () => {
    it('broadcasts (3, 4) - (4,) correctly', () => {
      const a = array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
      ]);
      const b = array([1, 2, 3, 4]);
      const result = a.subtract(b);

      const pyResult = runNumPy(`
a = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
b = np.array([1, 2, 3, 4])
result = a - b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (3, 1) - (1, 4) correctly', () => {
      const a = array([[10], [20], [30]]);
      const b = array([[1, 2, 3, 4]]);
      const result = a.subtract(b);

      const pyResult = runNumPy(`
a = np.array([[10], [20], [30]])
b = np.array([[1, 2, 3, 4]])
result = a - b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('multiply with broadcasting', () => {
    it('broadcasts (3, 4) * (4,) correctly', () => {
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const b = array([2, 2, 2, 2]);
      const result = a.multiply(b);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.array([2, 2, 2, 2])
result = a * b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (3, 1) * (1, 4) correctly', () => {
      const a = array([[2], [3], [4]]);
      const b = array([[10, 20, 30, 40]]);
      const result = a.multiply(b);

      const pyResult = runNumPy(`
a = np.array([[2], [3], [4]])
b = np.array([[10, 20, 30, 40]])
result = a * b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('divide with broadcasting', () => {
    it('broadcasts (3, 4) / (4,) correctly', () => {
      const a = array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
      ]);
      const b = array([2, 4, 5, 8]);
      const result = a.divide(b);

      const pyResult = runNumPy(`
a = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
b = np.array([2, 4, 5, 8])
result = a / b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts (3, 1) / (1, 4) correctly', () => {
      const a = array([[100], [200], [300]]);
      const b = array([[2, 4, 5, 10]]);
      const result = a.divide(b);

      const pyResult = runNumPy(`
a = np.array([[100], [200], [300]])
b = np.array([[2, 4, 5, 10]])
result = a / b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Complex broadcasting scenarios', () => {
    it('handles 3D broadcasting correctly', () => {
      const a = array([[[1, 2, 3]], [[4, 5, 6]]]); // (2, 1, 3)
      const b = array([[10], [20]]); // (2, 1)
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([[[1, 2, 3]], [[4, 5, 6]]])
b = np.array([[10], [20]])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts with larger dimensions', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // (3, 2)
      const b = array([100, 200]); // (2,)
      const result = a.multiply(b);

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([100, 200])
result = a * b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('maintains commutativity', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([10, 20]);

      const result1 = a.add(b);
      const result2 = b.add(a);

      const pyResult1 = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
result = a + b
      `);

      const pyResult2 = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
result = b + a
      `);

      expect(result1.shape).toEqual(pyResult1.shape);
      expect(result2.shape).toEqual(pyResult2.shape);
      expect(arraysClose(result1.toArray(), pyResult1.value)).toBe(true);
      expect(arraysClose(result2.toArray(), pyResult2.value)).toBe(true);
    });
  });

  describe('Edge cases', () => {
    it('broadcasts with shape (1, 1)', () => {
      const a = array([[5]]); // (1, 1)
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]); // (2, 3)
      const result = a.add(b);

      const pyResult = runNumPy(`
a = np.array([[5]])
b = np.array([[1, 2, 3], [4, 5, 6]])
result = a + b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });

    it('broadcasts with multiple singleton dimensions', () => {
      const a = array([[[1]], [[2]]]); // (2, 1, 1)
      const b = array([[10, 20, 30]]); // (1, 3)
      const result = a.multiply(b);

      const pyResult = runNumPy(`
a = np.array([[[1]], [[2]]])
b = np.array([[10, 20, 30]])
result = a * b
      `);

      expect(result.shape).toEqual(pyResult.shape);
      expect(arraysClose(result.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('broadcast_shapes', () => {
    it('matches NumPy broadcast_shapes for compatible shapes', () => {
      const result1 = broadcast_shapes([3, 4], [4]);
      expect(result1).toEqual([3, 4]);

      const pyResult1 = runNumPy(`
result = np.broadcast_shapes((3, 4), (4,))
      `);
      expect(result1).toEqual(Array.from(pyResult1.value));
    });

    it('matches NumPy for complex broadcasting', () => {
      const result = broadcast_shapes([8, 1, 6, 1], [7, 1, 5]);
      expect(result).toEqual([8, 7, 6, 5]);

      const pyResult = runNumPy(`
result = np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5))
      `);
      expect(result).toEqual(Array.from(pyResult.value));
    });

    it('matches NumPy for multiple shapes', () => {
      const result = broadcast_shapes([6, 7], [5, 6, 1], [7], [5, 1, 7]);
      expect(result).toEqual([5, 6, 7]);

      const pyResult = runNumPy(`
result = np.broadcast_shapes((6, 7), (5, 6, 1), (7,), (5, 1, 7))
      `);
      expect(result).toEqual(Array.from(pyResult.value));
    });

    it('throws error for incompatible shapes like NumPy', () => {
      expect(() => broadcast_shapes([3], [4])).toThrow();

      const pyCode = `
try:
    np.broadcast_shapes((3,), (4,))
    result = False
except ValueError:
    result = True
      `;
      const pyResult = runNumPy(pyCode);
      expect(pyResult.value).toBe(true);
    });
  });
});
