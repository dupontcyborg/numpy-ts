/**
 * NumPy validation tests for reduction kwargs: where=, initial=, dtype=.
 * Covers sum, prod, max, min, all, any.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import { any, arange, array, max, min, prod, sum } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: reduction kwargs', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('where=', () => {
    it('sum with boolean mask', () => {
      const a = arange(10);
      const mask = a.greater(3);
      const result = sum(a, undefined, undefined, { where: mask }) as number;
      const np = runNumPy(`
a = np.arange(10)
result = np.sum(a, where=a > 3)
`);
      expect(result).toBeCloseTo(np.value, 10);
    });

    it('sum along axis with where=', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const mask = a.greater(3);
      const result = sum(a, 1, undefined, { where: mask }) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.sum(a, axis=1, where=a > 3)
`);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('prod with where=', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = prod(a, undefined, undefined, {
        where: [true, false, true, false, true],
      }) as number;
      const np = runNumPy(`
result = np.prod(np.array([1, 2, 3, 4, 5]), where=[True, False, True, False, True])
`);
      expect(result).toBeCloseTo(np.value, 10);
    });

    it('max with where=', () => {
      const a = array([1, 100, 3, 50, 5]);
      const result = max(a, undefined, undefined, {
        where: [true, false, true, false, true],
      }) as number;
      const np = runNumPy(`
result = np.max(np.array([1, 100, 3, 50, 5]),
                where=np.array([True, False, True, False, True]),
                initial=-1000000)
`);
      expect(result).toBe(np.value);
    });

    it('any with where=', () => {
      const a = array([false, true, false, true]);
      const result = any(a, undefined, undefined, { where: [true, false, true, false] }) as boolean;
      const np = runNumPy(`
result = np.any(np.array([False, True, False, True]), where=[True, False, True, False])
`);
      expect(result).toBe(Boolean(np.value));
    });
  });

  describe('initial=', () => {
    it('sum with initial', () => {
      const a = arange(5);
      const result = sum(a, undefined, undefined, { initial: 100 }) as number;
      const np = runNumPy(`
result = np.sum(np.arange(5), initial=100)
`);
      expect(result).toBeCloseTo(np.value, 10);
    });

    it('prod with initial', () => {
      const a = array([1, 2, 3]);
      const result = prod(a, undefined, undefined, { initial: 10 }) as number;
      const np = runNumPy(`
result = np.prod(np.array([1, 2, 3]), initial=10)
`);
      expect(result).toBeCloseTo(np.value, 10);
    });

    it('max with initial pulling result up', () => {
      const a = array([1, 2, 3]);
      const result = max(a, undefined, undefined, { initial: 100 }) as number;
      const np = runNumPy(`
result = np.max(np.array([1, 2, 3]), initial=100)
`);
      expect(result).toBe(np.value);
    });

    it('min with initial pulling result down', () => {
      const a = array([10, 20, 30]);
      const result = min(a, undefined, undefined, { initial: 5 }) as number;
      const np = runNumPy(`
result = np.min(np.array([10, 20, 30]), initial=5)
`);
      expect(result).toBe(np.value);
    });

    it('axis-reduce with initial broadcasts elementwise', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = sum(a, 0, undefined, { initial: 100 }) as NDArray;
      const np = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6]])
result = np.sum(a, axis=0, initial=100)
`);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });

  describe('combined where= + initial=', () => {
    it('all-False mask falls through to initial', () => {
      const a = array([1, 2, 3]);
      const result = sum(a, undefined, undefined, {
        where: [false, false, false],
        initial: 99,
      }) as number;
      const np = runNumPy(`
result = np.sum(np.array([1, 2, 3]),
                where=np.array([False, False, False]),
                initial=99)
`);
      expect(result).toBe(np.value);
    });
  });

  describe('dtype=', () => {
    it('sum with dtype=float64 on int input', () => {
      const a = arange(5);
      const result = sum(a, undefined, undefined, { dtype: 'float64' }) as number;
      const np = runNumPy(`
result = float(np.sum(np.arange(5), dtype=np.float64))
`);
      expect(result).toBeCloseTo(np.value, 10);
    });
  });
});
