/**
 * Python NumPy validation tests for matrix operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Matrix Operations', () => {
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

  describe('matmul', () => {
    it('matches NumPy for 2x2 @ 2x2', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).matmul(
        array([
          [5, 6],
          [7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2x3 @ 3x2', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]).matmul(
        array([
          [7, 8],
          [9, 10],
          [11, 12],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[7, 8], [9, 10], [11, 12]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 3x3 @ 3x3', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]).matmul(
        array([
          [9, 8, 7],
          [6, 5, 4],
          [3, 2, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) @ np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for identity matrix multiplication', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const I = array([
        [1, 0],
        [0, 1],
      ]);
      const jsResult = A.matmul(I);

      const pyResult = runNumPy(`
A = np.array([[1, 2], [3, 4]])
I = np.array([[1, 0], [0, 1]])
result = A @ I
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for matrix with zeros', () => {
      const jsResult = array([
        [1, 0],
        [0, 1],
      ]).matmul(
        array([
          [2, 3],
          [4, 5],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 0], [0, 1]]) @ np.array([[2, 3], [4, 5]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([
        [-1, 2],
        [3, -4],
      ]).matmul(
        array([
          [5, -6],
          [-7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[-1, 2], [3, -4]]) @ np.array([[5, -6], [-7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for larger matrices (5x5)', () => {
      const A = array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
      ]);
      const B = array([
        [25, 24, 23, 22, 21],
        [20, 19, 18, 17, 16],
        [15, 14, 13, 12, 11],
        [10, 9, 8, 7, 6],
        [5, 4, 3, 2, 1],
      ]);
      const jsResult = A.matmul(B);

      const pyResult = runNumPy(`
A = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
B = np.array([[25,24,23,22,21],[20,19,18,17,16],[15,14,13,12,11],[10,9,8,7,6],[5,4,3,2,1]])
result = A @ B
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('matmul with transpose', () => {
    it('matches NumPy for A.T @ B (case 1: transpose A only)', () => {
      // A is 3x2, A.T is 2x3, B is 2x3
      // A.T @ B should give (2x3) @ (2x3) -> Error OR need B to be (3xN)
      // Let's fix: A is 3x2, A.T is 2x3, B is 3x2 -> result is 2x2
      const A = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const B = array([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const jsResult = A.transpose().matmul(B);

      const pyResult = runNumPy(`
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
result = A.T @ B
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for A @ B.T (case 2: transpose B only)', () => {
      // A is 2x3, B is 2x3, B.T is 3x2, result should be 2x2
      const A = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const B = array([
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const jsResult = A.matmul(B.transpose());

      const pyResult = runNumPy(`
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])
result = A @ B.T
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for A.T @ B.T (case 3: transpose both)', () => {
      // A is 3x2, A.T is 2x3, B is 3x2, B.T is 2x3
      // A.T @ B.T: (2x3) @ (2x3) -> incompatible!
      // Let's fix: A is 3x2, A.T is 2x3, B is 2x3, B.T is 3x2
      // A.T @ B.T: (2x3) @ (3x2) -> (2x2)
      const A = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const B = array([
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const jsResult = A.transpose().matmul(B.transpose());

      const pyResult = runNumPy(`
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])
result = A.T @ B.T
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for transposed square matrices (all combinations)', () => {
      const A = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const B = array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
      ]);

      // Test A.T @ B
      const jsResult1 = A.transpose().matmul(B);
      const pyResult1 = runNumPy(`
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[3,2,1]])
result = A.T @ B
      `);
      expect(jsResult1.shape).toEqual(pyResult1.shape);
      expect(arraysClose(jsResult1.toArray(), pyResult1.value)).toBe(true);

      // Test A @ B.T
      const jsResult2 = A.matmul(B.transpose());
      const pyResult2 = runNumPy(`
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[3,2,1]])
result = A @ B.T
      `);
      expect(jsResult2.shape).toEqual(pyResult2.shape);
      expect(arraysClose(jsResult2.toArray(), pyResult2.value)).toBe(true);

      // Test A.T @ B.T
      const jsResult3 = A.transpose().matmul(B.transpose());
      const pyResult3 = runNumPy(`
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[3,2,1]])
result = A.T @ B.T
      `);
      expect(jsResult3.shape).toEqual(pyResult3.shape);
      expect(arraysClose(jsResult3.toArray(), pyResult3.value)).toBe(true);
    });

    it('matches NumPy for rectangular transposed matrices', () => {
      const A = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const B = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);

      // A is 2x4, B is 4x2, normal: A @ B = 2x2
      const jsResult1 = A.matmul(B);
      const pyResult1 = runNumPy(`
A = np.array([[1,2,3,4],[5,6,7,8]])
B = np.array([[1,2],[3,4],[5,6],[7,8]])
result = A @ B
      `);
      expect(jsResult1.shape).toEqual(pyResult1.shape);
      expect(arraysClose(jsResult1.toArray(), pyResult1.value)).toBe(true);

      // A.T is 4x2, B is 4x2, B.T is 2x4
      // A.T @ B.T: (4x2) @ (2x4) = 4x4
      const jsResult2 = A.transpose().matmul(B.transpose());
      const pyResult2 = runNumPy(`
A = np.array([[1,2,3,4],[5,6,7,8]])
B = np.array([[1,2],[3,4],[5,6],[7,8]])
result = A.T @ B.T
      `);
      expect(jsResult2.shape).toEqual(pyResult2.shape);
      expect(arraysClose(jsResult2.toArray(), pyResult2.value)).toBe(true);
    });

    it('matches NumPy for sliced transposed matrices', () => {
      // Create a larger matrix and take a slice, then transpose
      const A = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const ASlice = A.slice('0:2', '1:4'); // 2x3 submatrix
      const B = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);

      // ASlice.T @ B: (3x2) @ (3x2) -> Error! Need B to be (2xN)
      // Let's do ASlice @ B: (2x3) @ (3x2) = 2x2
      const jsResult = ASlice.matmul(B);
      const pyResult = runNumPy(`
A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
ASlice = A[0:2, 1:4]
B = np.array([[1,2],[3,4],[5,6]])
result = ASlice @ B
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);

      // Now test with transpose
      const jsResult2 = ASlice.transpose().matmul(B.transpose());
      const pyResult2 = runNumPy(`
A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
ASlice = A[0:2, 1:4]
B = np.array([[1,2],[3,4],[5,6]])
result = ASlice.T @ B.T
      `);
      expect(jsResult2.shape).toEqual(pyResult2.shape);
      expect(arraysClose(jsResult2.toArray(), pyResult2.value)).toBe(true);
    });
  });
});
