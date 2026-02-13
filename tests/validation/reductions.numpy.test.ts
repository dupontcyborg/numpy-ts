/**
 * Python NumPy validation tests for reduction operations with axis support
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Reductions with Axis Support', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('sum() with axis', () => {
    it('validates sum() without axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates sum(axis=0) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=-1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(-1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.sum(1, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=0) for 3D array', () => {
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
      const result = arr.sum(0);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=1) for 3D array', () => {
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
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates sum(axis=2) for 3D array', () => {
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
      const result = arr.sum(2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.sum(axis=2)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('mean() with axis', () => {
    it('validates mean() without axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates mean(axis=0) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=1) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.mean(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.mean(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates mean(axis=-1) for 3D array', () => {
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
      const result = arr.mean(-1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.mean(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('max() with axis', () => {
    it('validates max() without axis', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max();

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates max(axis=0) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(0);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=1) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(1);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=-1) for 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.max(-1);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = arr.max(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates max(axis=1) for 3D array', () => {
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
      const result = arr.max(1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.max(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('min() with axis', () => {
    it('validates min() without axis', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min();

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates min(axis=0) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(0);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=1) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(1);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=0, keepdims=True) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(0, true);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=-1) for 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [1, 9, 2],
      ]);
      const result = arr.min(-1);

      const npResult = runNumPy(`
arr = np.array([[5, 3, 8], [1, 9, 2]])
result = arr.min(axis=-1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates min(axis=2) for 3D array', () => {
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
      const result = arr.min(2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = arr.min(axis=2)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('complex scenarios', () => {
    it('validates large matrix reduction', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const result = arr.sum(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = arr.sum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates chained reductions', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = (arr.sum(0) as any).mean();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.sum(axis=0).mean()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates reduction preserving shape', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.max(1, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.max(axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect((result as any).ndim).toBe(2);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('prod() with axis', () => {
    it('validates prod() without axis', () => {
      const arr = array([
        [2, 3],
        [4, 5],
      ]);
      const result = arr.prod();

      const npResult = runNumPy(`
arr = np.array([[2, 3], [4, 5]])
result = arr.prod()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates prod(axis=0) for 2D array', () => {
      const arr = array([
        [2, 3],
        [4, 5],
      ]);
      const result = arr.prod(0);

      const npResult = runNumPy(`
arr = np.array([[2, 3], [4, 5]])
result = arr.prod(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates prod(axis=1, keepdims=True)', () => {
      const arr = array([
        [2, 3],
        [4, 5],
      ]);
      const result = arr.prod(1, true);

      const npResult = runNumPy(`
arr = np.array([[2, 3], [4, 5]])
result = arr.prod(axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('argmin() with axis', () => {
    it('validates argmin() without axis', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmin();

      const npResult = runNumPy(`
arr = np.array([[3, 1], [2, 4]])
result = arr.argmin()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates argmin(axis=0)', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmin(0);

      const npResult = runNumPy(`
arr = np.array([[3, 1], [2, 4]])
result = arr.argmin(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argmin(axis=1)', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmin(1);

      const npResult = runNumPy(`
arr = np.array([[3, 1], [2, 4]])
result = arr.argmin(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('argmax() with axis', () => {
    it('validates argmax() without axis', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmax();

      const npResult = runNumPy(`
arr = np.array([[3, 1], [2, 4]])
result = arr.argmax()
`);

      expect(result).toBe(npResult.value);
    });

    it('validates argmax(axis=0)', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmax(0);

      const npResult = runNumPy(`
arr = np.array([[3, 1], [2, 4]])
result = arr.argmax(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('var() with axis', () => {
    it('validates var() without axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var();

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.var()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates var(axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.var(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates var(axis=0, ddof=1)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var(0, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.var(axis=0, ddof=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates var(axis=0, keepdims=True)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var(0, 0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.var(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('std() with axis', () => {
    it('validates std() without axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.std();

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.std()
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates std(axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.std(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.std(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates std(axis=0, ddof=1)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.std(0, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.std(axis=0, ddof=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('all() with axis', () => {
    it('validates all() without axis', () => {
      const arr = array([
        [1, 1],
        [1, 0],
      ]);
      const result = arr.all();

      const npResult = runNumPy(`
arr = np.array([[1, 1], [1, 0]])
result = bool(arr.all())
`);

      expect(result).toBe(npResult.value);
    });

    it('validates all(axis=0)', () => {
      const arr = array([
        [1, 1],
        [1, 0],
      ]);
      const result = arr.all(0);

      const npResult = runNumPy(`
arr = np.array([[1, 1], [1, 0]])
result = arr.all(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect((result as any).toArray()).toEqual(npResult.value.map((v: number) => (v ? 1 : 0)));
    });

    it('validates all(axis=0, keepdims=True)', () => {
      const arr = array([
        [1, 1],
        [1, 0],
      ]);
      const result = arr.all(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 1], [1, 0]])
result = arr.all(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
    });
  });

  describe('any() with axis', () => {
    it('validates any() without axis', () => {
      const arr = array([
        [0, 0],
        [1, 0],
      ]);
      const result = arr.any();

      const npResult = runNumPy(`
arr = np.array([[0, 0], [1, 0]])
result = bool(arr.any())
`);

      expect(result).toBe(npResult.value);
    });

    it('validates any(axis=0)', () => {
      const arr = array([
        [0, 0],
        [1, 0],
      ]);
      const result = arr.any(0);

      const npResult = runNumPy(`
arr = np.array([[0, 0], [1, 0]])
result = arr.any(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect((result as any).toArray()).toEqual(npResult.value.map((v: number) => (v ? 1 : 0)));
    });

    it('validates any(axis=0, keepdims=True)', () => {
      const arr = array([
        [0, 0],
        [1, 0],
      ]);
      const result = arr.any(0, true);

      const npResult = runNumPy(`
arr = np.array([[0, 0], [1, 0]])
result = arr.any(axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
    });
  });

  describe('cumsum() with axis', () => {
    it('validates cumsum() without axis (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumsum();

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.cumsum()
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates cumsum(axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumsum(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.cumsum(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates cumsum(axis=1)', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.cumsum(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.cumsum(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('cumprod() with axis', () => {
    it('validates cumprod() without axis (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumprod();

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.cumprod()
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates cumprod(axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumprod(0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.cumprod(axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates cumprod(axis=1)', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.cumprod(1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.cumprod(axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('ptp() with axis', () => {
    it('validates ptp() without axis', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp();

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = np.ptp(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates ptp(axis=0)', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(0);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = np.ptp(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates ptp(axis=1)', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(1);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = np.ptp(arr, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates ptp(axis=0, keepdims=True)', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(0, true);

      const npResult = runNumPy(`
arr = np.array([[1, 5, 3], [9, 2, 6]])
result = np.ptp(arr, axis=0, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('median() with axis', () => {
    it('validates median() without axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.median();

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.median(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates median(axis=0)', () => {
      const arr = array([
        [1, 3, 5],
        [2, 4, 6],
      ]);
      const result = arr.median(0);

      const npResult = runNumPy(`
arr = np.array([[1, 3, 5], [2, 4, 6]])
result = np.median(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates median(axis=1)', () => {
      const arr = array([
        [1, 3, 2],
        [6, 4, 5],
      ]);
      const result = arr.median(1);

      const npResult = runNumPy(`
arr = np.array([[1, 3, 2], [6, 4, 5]])
result = np.median(arr, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates median(axis=1, keepdims=True)', () => {
      const arr = array([
        [1, 3, 2],
        [6, 4, 5],
      ]);
      const result = arr.median(1, true);

      const npResult = runNumPy(`
arr = np.array([[1, 3, 2], [6, 4, 5]])
result = np.median(arr, axis=1, keepdims=True)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('percentile() with axis', () => {
    it('validates percentile(50) without axis', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.percentile(50);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.percentile(arr, 50)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates percentile(25) without axis', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.percentile(25);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.percentile(arr, 25)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates percentile(50, axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = arr.percentile(50, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = np.percentile(arr, 50, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('quantile() with axis', () => {
    it('validates quantile(0.5) without axis', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.quantile(0.5);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.quantile(arr, 0.5)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates quantile(0.25) without axis', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.quantile(0.25);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.quantile(arr, 0.25)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates quantile(0.5, axis=1)', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.quantile(0.5, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.quantile(arr, 0.5, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('average() with weights', () => {
    it('validates average() without weights', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.average();

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.average(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates average() with weights', () => {
      const arr = array([1, 2, 3]);
      const weights = array([3, 2, 1]);
      const result = arr.average(weights);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
weights = np.array([3, 2, 1])
result = np.average(arr, weights=weights)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates average(axis=0)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.average(undefined, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.average(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates average(axis=1) with weights', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const weights = array([3, 1]);
      const result = arr.average(weights, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
weights = np.array([3, 1])
result = np.average(arr, axis=1, weights=weights)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nansum() with axis', () => {
    it('validates nansum() without axis', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nansum();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 3, np.nan, 5])
result = np.nansum(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nansum(axis=0)', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nansum(0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = np.nansum(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nansum(axis=1)', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nansum(1);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = np.nansum(arr, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanmean() with axis', () => {
    it('validates nanmean() without axis', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nanmean();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 3, np.nan, 5])
result = np.nanmean(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanmean(axis=0)', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nanmean(0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = np.nanmean(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanvar() and nanstd() with axis', () => {
    it('validates nanvar() without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      const result = arr.nanvar();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3])
result = np.nanvar(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanstd() without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      const result = arr.nanstd();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3])
result = np.nanstd(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanvar(axis=0)', () => {
      const arr = array([
        [1, NaN],
        [2, 4],
      ]);
      const result = arr.nanvar(0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan], [2, 4]])
result = np.nanvar(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanmin() and nanmax() with axis', () => {
    it('validates nanmin() without axis', () => {
      const arr = array([NaN, 3, NaN, 1, NaN, 5]);
      const result = arr.nanmin();

      const npResult = runNumPy(`
arr = np.array([np.nan, 3, np.nan, 1, np.nan, 5])
result = np.nanmin(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nanmax() without axis', () => {
      const arr = array([NaN, 3, NaN, 9, NaN, 5]);
      const result = arr.nanmax();

      const npResult = runNumPy(`
arr = np.array([np.nan, 3, np.nan, 9, np.nan, 5])
result = np.nanmax(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nanmin(axis=0)', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 2],
      ]);
      const result = arr.nanmin(0);

      const npResult = runNumPy(`
arr = np.array([[np.nan, 5, 3], [9, np.nan, 2]])
result = np.nanmin(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nanmax(axis=0)', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 8],
      ]);
      const result = arr.nanmax(0);

      const npResult = runNumPy(`
arr = np.array([[np.nan, 5, 3], [9, np.nan, 8]])
result = np.nanmax(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanargmin() and nanargmax() with axis', () => {
    it('validates nanargmin() without axis', () => {
      const arr = array([NaN, 3, NaN, 1, NaN, 5]);
      const result = arr.nanargmin();

      const npResult = runNumPy(`
arr = np.array([np.nan, 3, np.nan, 1, np.nan, 5])
result = np.nanargmin(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nanargmax() without axis', () => {
      const arr = array([NaN, 3, NaN, 9, NaN, 5]);
      const result = arr.nanargmax();

      const npResult = runNumPy(`
arr = np.array([np.nan, 3, np.nan, 9, np.nan, 5])
result = np.nanargmax(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nanargmin(axis=0)', () => {
      const arr = array([
        [NaN, 5],
        [2, NaN],
      ]);
      const result = arr.nanargmin(0);

      const npResult = runNumPy(`
arr = np.array([[np.nan, 5], [2, np.nan]])
result = np.nanargmin(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nancumsum() and nancumprod() with axis', () => {
    it('validates nancumsum() without axis', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nancumsum();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 3, np.nan, 5])
result = np.nancumsum(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nancumprod() without axis', () => {
      const arr = array([2, NaN, 3, NaN, 4]);
      const result = arr.nancumprod();

      const npResult = runNumPy(`
arr = np.array([2, np.nan, 3, np.nan, 4])
result = np.nancumprod(arr)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nancumsum(axis=0)', () => {
      const arr = array([
        [1, NaN],
        [NaN, 4],
      ]);
      const result = arr.nancumsum(0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan], [np.nan, 4]])
result = np.nancumsum(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanmedian() with axis', () => {
    it('validates nanmedian() without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      const result = arr.nanmedian();

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3])
result = np.nanmedian(arr)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanmedian(axis=0)', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanmedian(0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = np.nanmedian(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nanmedian(axis=1)', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nanmedian(1);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = np.nanmedian(arr, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanprod() with axis', () => {
    it('validates nanprod() without axis', () => {
      const arr = array([2, NaN, 3, NaN, 4]);
      const result = arr.nanprod();

      const npResult = runNumPy(`
arr = np.array([2, np.nan, 3, np.nan, 4])
result = np.nanprod(arr)
`);

      expect(result).toBe(npResult.value);
    });

    it('validates nanprod(axis=0)', () => {
      const arr = array([
        [2, NaN],
        [3, 4],
      ]);
      const result = arr.nanprod(0);

      const npResult = runNumPy(`
arr = np.array([[2, np.nan], [3, 4]])
result = np.nanprod(arr, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanquantile() with axis', () => {
    it('validates nanquantile(0.5) without axis', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nanquantile(0.5);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 3, np.nan, 5])
result = np.nanquantile(arr, 0.5)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanquantile(0.25) without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3, NaN, 4]);
      const result = arr.nanquantile(0.25);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3, np.nan, 4])
result = np.nanquantile(arr, 0.25)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanquantile(0.75) without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3, NaN, 4]);
      const result = arr.nanquantile(0.75);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3, np.nan, 4])
result = np.nanquantile(arr, 0.75)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanquantile(0.5, axis=0)', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanquantile(0.5, 0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = np.nanquantile(arr, 0.5, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nanquantile(0.5, axis=1)', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nanquantile(0.5, 1);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = np.nanquantile(arr, 0.5, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('nanpercentile() with axis', () => {
    it('validates nanpercentile(50) without axis', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nanpercentile(50);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 3, np.nan, 5])
result = np.nanpercentile(arr, 50)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanpercentile(25) without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3, NaN, 4]);
      const result = arr.nanpercentile(25);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3, np.nan, 4])
result = np.nanpercentile(arr, 25)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanpercentile(75) without axis', () => {
      const arr = array([1, NaN, 2, NaN, 3, NaN, 4]);
      const result = arr.nanpercentile(75);

      const npResult = runNumPy(`
arr = np.array([1, np.nan, 2, np.nan, 3, np.nan, 4])
result = np.nanpercentile(arr, 75)
`);

      expect(Math.abs((result as number) - npResult.value)).toBeLessThan(1e-10);
    });

    it('validates nanpercentile(50, axis=0)', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanpercentile(50, 0);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = np.nanpercentile(arr, 50, axis=0)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates nanpercentile(50, axis=1)', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nanpercentile(50, 1);

      const npResult = runNumPy(`
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = np.nanpercentile(arr, 50, axis=1)
`);

      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});
