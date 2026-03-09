/**
 * Python NumPy validation tests for WASM-accelerated linear algebra.
 *
 * WASM acceleration is now built into matmul directly — no separate
 * entrypoint needed. These tests verify that the unified matmul
 * (which tries WASM first, falls back to JS) matches NumPy.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, matmul, dot, reshape, linalg } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: WASM-accelerated Linear Algebra', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        'Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup: source ~/.zshrc && conda activate py313\n'
      );
    }
  });

  describe('matmul (WASM)', () => {
    it('matches NumPy for 2x2 @ 2x2', () => {
      const jsResult = matmul(
        array([
          [1, 2],
          [3, 4],
        ]),
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
      const jsResult = matmul(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
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
      const jsResult = matmul(
        array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]),
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

    it('matches NumPy for larger matrices (above WASM threshold)', () => {
      // 20x20 = 400 elements per input, well above 256 threshold
      const aData: number[][] = [];
      const bData: number[][] = [];
      const pyA: string[] = [];
      const pyB: string[] = [];
      for (let i = 0; i < 20; i++) {
        const aRow = Array.from({ length: 20 }, (_, j) => i * 20 + j + 1);
        const bRow = Array.from({ length: 20 }, (_, j) => (i + j) * 0.1);
        aData.push(aRow);
        bData.push(bRow);
        pyA.push(`[${aRow.join(',')}]`);
        pyB.push(`[${bRow.join(',')}]`);
      }
      const jsResult = matmul(array(aData), array(bData));
      const pyResult = runNumPy(`
a = np.array([${pyA.join(',')}])
b = np.array([${pyB.join(',')}])
result = a @ b
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-6)).toBe(true);
    });

    it('matches NumPy for 1D @ 2D', () => {
      const jsResult = matmul(
        array([1, 2, 3]),
        array([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) @ np.array([[1, 2], [3, 4], [5, 6]])
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D @ 1D', () => {
      const jsResult = matmul(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        array([1, 2, 3])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([1, 2, 3])
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for batched 3D matmul', () => {
      const a = reshape(array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
      const b = reshape(array([1, 0, 0, 1, 2, 1, 1, 2]), [2, 2, 2]);
      const jsResult = matmul(a, b);
      const pyResult = runNumPy(`
a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
b = np.array([[[1,0],[0,1]],[[2,1],[1,2]]])
result = a @ b
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('dot via matmul (WASM)', () => {
    it('2D dot delegates to matmul — matches NumPy', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = dot(a, b);
      const pyResult = runNumPy(`
result = np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('linalg.multi_dot (WASM)', () => {
    it('chain of matmuls matches NumPy', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const c = array([
        [9, 10],
        [11, 12],
      ]);
      const jsResult = linalg.multi_dot([a, b, c]);
      const pyResult = runNumPy(`
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.array([[9,10],[11,12]])
result = np.linalg.multi_dot([a, b, c])
      `);
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
