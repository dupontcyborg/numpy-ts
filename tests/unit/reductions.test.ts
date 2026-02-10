/**
 * Unit tests for reduction operations with axis support
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src';

describe('Reductions with Axis Support', () => {
  describe('sum()', () => {
    describe('without axis (all elements)', () => {
      it('sums all elements in 1D array', () => {
        const arr = array([1, 2, 3, 4, 5]);
        expect(arr.sum()).toBe(15);
      });

      it('sums all elements in 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.sum()).toBe(21);
      });

      it('sums all elements in 3D array', () => {
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
        expect(arr.sum()).toBe(36);
      });
    });

    describe('with axis=0', () => {
      it('sums along axis 0 for 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 7, 9]);
      });

      it('sums along axis 0 for 3D array', () => {
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
        const result = arr.sum(0) as any;
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [6, 8],
          [10, 12],
        ]);
      });
    });

    describe('with axis=1', () => {
      it('sums along axis 1 for 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([6, 15]);
      });

      it('sums along axis 1 for 3D array', () => {
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
        const result = arr.sum(1) as any;
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [4, 6],
          [12, 14],
        ]);
      });
    });

    describe('with negative axis', () => {
      it('handles axis=-1 (last axis)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(-1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([6, 15]);
      });

      it('handles axis=-2 (second-to-last axis)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(-2) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 7, 9]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[5, 7, 9]]);
      });

      it('keeps dimensions for axis=1', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.sum(1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(result.toArray()).toEqual([[6], [15]]);
      });
    });

    describe('error cases', () => {
      it('throws on out of bounds axis', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(() => arr.sum(2)).toThrow(/out of bounds/);
      });

      it('throws on negative out of bounds axis', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(() => arr.sum(-3)).toThrow(/out of bounds/);
      });
    });
  });

  describe('mean()', () => {
    describe('without axis', () => {
      it('computes mean of all elements', () => {
        const arr = array([1, 2, 3, 4, 5]);
        expect(arr.mean()).toBe(3);
      });

      it('computes mean of 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.mean()).toBe(3.5);
      });
    });

    describe('with axis=0', () => {
      it('computes mean along axis 0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([2.5, 3.5, 4.5]);
      });
    });

    describe('with axis=1', () => {
      it('computes mean along axis 1', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([2, 5]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.mean(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[2.5, 3.5, 4.5]]);
      });
    });
  });

  describe('max()', () => {
    describe('without axis', () => {
      it('finds max of all elements', () => {
        const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
        expect(arr.max()).toBe(9);
      });

      it('finds max of 2D array', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        expect(arr.max()).toBe(9);
      });
    });

    describe('with axis=0', () => {
      it('finds max along axis 0', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([9, 5, 6]);
      });
    });

    describe('with axis=1', () => {
      it('finds max along axis 1', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([5, 9]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [1, 5, 3],
          [9, 2, 6],
        ]);
        const result = arr.max(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[9, 5, 6]]);
      });
    });
  });

  describe('min()', () => {
    describe('without axis', () => {
      it('finds min of all elements', () => {
        const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
        expect(arr.min()).toBe(1);
      });

      it('finds min of 2D array', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        expect(arr.min()).toBe(1);
      });
    });

    describe('with axis=0', () => {
      it('finds min along axis 0', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(0) as any;
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([1, 3, 2]);
      });
    });

    describe('with axis=1', () => {
      it('finds min along axis 1', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(1) as any;
        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([3, 1]);
      });
    });

    describe('with keepdims=true', () => {
      it('keeps dimensions for axis=0', () => {
        const arr = array([
          [5, 3, 8],
          [1, 9, 2],
        ]);
        const result = arr.min(0, true) as any;
        expect(result.shape).toEqual([1, 3]);
        expect(result.toArray()).toEqual([[1, 3, 2]]);
      });
    });
  });

  describe('complex scenarios', () => {
    it('handles large matrices', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);

      const sumAxis0 = arr.sum(0) as any;
      expect(sumAxis0.toArray()).toEqual([15, 18, 21, 24]);

      const sumAxis1 = arr.sum(1) as any;
      expect(sumAxis1.toArray()).toEqual([10, 26, 42]);
    });

    it('handles 3D arrays with different axes', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // shape: (2, 2, 2)

      const sumAxis0 = arr.sum(0) as any;
      expect(sumAxis0.shape).toEqual([2, 2]);

      const sumAxis1 = arr.sum(1) as any;
      expect(sumAxis1.shape).toEqual([2, 2]);

      const sumAxis2 = arr.sum(2) as any;
      expect(sumAxis2.shape).toEqual([2, 2]);
    });

    it('combines reductions', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      // Sum along axis 0, then mean of result
      const sumAxis0 = arr.sum(0) as any;
      const meanOfSum = sumAxis0.mean();
      expect(meanOfSum).toBe(7); // [5, 7, 9] -> mean = 7
    });
  });

  describe('prod()', () => {
    it('computes product of all elements', () => {
      const arr = array([2, 3, 4]);
      expect(arr.prod()).toBe(24);
    });

    it('computes product along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.prod(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([3, 8]);
    });

    it('computes product along axis=1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.prod(1) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([2, 12]);
    });

    it('computes product with keepdims=true', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.prod(1, true) as any;
      expect(result.shape).toEqual([2, 1]);
      expect(result.toArray()).toEqual([[2], [12]]);
    });
  });

  describe('argmin()', () => {
    it('finds index of minimum value', () => {
      const arr = array([3, 1, 4, 1, 5]);
      expect(arr.argmin()).toBe(1);
    });

    it('finds indices of minimum along axis=0', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmin(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 0]);
    });

    it('finds indices of minimum along axis=1', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmin(1) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 0]);
    });
  });

  describe('argmax()', () => {
    it('finds index of maximum value', () => {
      const arr = array([3, 1, 4, 1, 5]);
      expect(arr.argmax()).toBe(4);
    });

    it('finds indices of maximum along axis=0', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmax(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([0, 1]);
    });

    it('finds indices of maximum along axis=1', () => {
      const arr = array([
        [3, 1],
        [2, 4],
      ]);
      const result = arr.argmax(1) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([0, 1]);
    });
  });

  describe('var()', () => {
    it('computes variance of all elements', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.var();
      expect(result).toBeCloseTo(2.0, 5);
    });

    it('computes variance with ddof=1', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.var(undefined, 1);
      expect(result).toBeCloseTo(2.5, 5);
    });

    it('computes variance along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()[0]).toBeCloseTo(1.0, 5);
      expect(result.toArray()[1]).toBeCloseTo(1.0, 5);
    });

    it('computes variance with keepdims=true', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.var(0, 0, true) as any;
      expect(result.shape).toEqual([1, 2]);
    });
  });

  describe('std()', () => {
    it('computes std of all elements', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.std();
      expect(result).toBeCloseTo(Math.sqrt(2.0), 5);
    });

    it('computes std with ddof=1', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.std(undefined, 1);
      expect(result).toBeCloseTo(Math.sqrt(2.5), 5);
    });

    it('computes std along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.std(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()[0]).toBeCloseTo(1.0, 5);
      expect(result.toArray()[1]).toBeCloseTo(1.0, 5);
    });
  });

  describe('all()', () => {
    it('returns true when all elements are truthy', () => {
      const arr = array([1, 2, 3]);
      expect(arr.all()).toBe(true);
    });

    it('returns false when any element is falsy', () => {
      const arr = array([1, 0, 3]);
      expect(arr.all()).toBe(false);
    });

    it('tests all along axis=0', () => {
      const arr = array([
        [1, 0],
        [1, 1],
      ]);
      const result = arr.all(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 0]);
    });

    it('tests all with keepdims=true', () => {
      const arr = array([
        [1, 0],
        [1, 1],
      ]);
      const result = arr.all(0, true) as any;
      expect(result.shape).toEqual([1, 2]);
    });
  });

  describe('any()', () => {
    it('returns false when all elements are falsy', () => {
      const arr = array([0, 0, 0]);
      expect(arr.any()).toBe(false);
    });

    it('returns true when any element is truthy', () => {
      const arr = array([0, 1, 0]);
      expect(arr.any()).toBe(true);
    });

    it('tests any along axis=0', () => {
      const arr = array([
        [0, 0],
        [1, 0],
      ]);
      const result = arr.any(0) as any;
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 0]);
    });

    it('tests any with keepdims=true', () => {
      const arr = array([
        [0, 0],
        [1, 0],
      ]);
      const result = arr.any(0, true) as any;
      expect(result.shape).toEqual([1, 2]);
    });
  });
});
