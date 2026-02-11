/**
 * Extended unit tests for reduction operations
 * Tests edge cases, BigInt dtypes, and special scenarios
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  sum,
  mean,
  prod,
  std,
  cumsum,
  cumprod,
  nansum,
  nanprod,
  nanmean,
  ptp,
  median,
  average,
  diff,
  nanvar,
  nanstd,
  nanmin,
  nanmax,
  nanargmin,
  nanargmax,
  nancumsum,
  nancumprod,
  nanmedian,
  nanquantile,
  nanpercentile,
  percentile,
  quantile,
} from '../../src';

describe('Extended reduction tests', () => {
  describe('sum() with BigInt dtypes', () => {
    it('sums int64 arrays', () => {
      const arr = array([1, 2, 3, 4, 5], 'int64');
      expect(arr.sum()).toBe(15);
    });

    it('sums uint64 arrays', () => {
      const arr = array([10, 20, 30], 'uint64');
      expect(arr.sum()).toBe(60);
    });

    it('sums int64 array along axis', () => {
      const arr = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'int64'
      );
      const result = arr.sum(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(5));
      expect((result as any).data[1]).toBe(BigInt(7));
      expect((result as any).data[2]).toBe(BigInt(9));
    });

    it('sums uint64 array along axis', () => {
      const arr = array(
        [
          [1, 2],
          [3, 4],
        ],
        'uint64'
      );
      const result = arr.sum(1);
      expect((result as any).dtype).toBe('uint64');
      expect((result as any).data[0]).toBe(BigInt(3));
      expect((result as any).data[1]).toBe(BigInt(7));
    });

    it('sums int64 with keepdims', () => {
      const arr = array([[1, 2, 3]], 'int64');
      const result = arr.sum(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(6));
    });
  });

  describe('mean() with different dtypes', () => {
    it('computes mean of int32 array', () => {
      const arr = array([1, 2, 3, 4, 5], 'int32');
      expect(arr.mean()).toBe(3);
    });

    it('computes mean of int64 array', () => {
      const arr = array([2, 4, 6, 8], 'int64');
      expect(arr.mean()).toBe(5);
    });

    it('computes mean of uint64 array', () => {
      const arr = array([10, 20, 30], 'uint64');
      expect(arr.mean()).toBe(20);
    });

    it('computes mean along axis for int64', () => {
      const arr = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int64'
      );
      const result = arr.mean(0);
      expect((result as any).shape).toEqual([2]);
      // Mean promotes to float64 for accurate computation
      expect((result as any).dtype).toBe('float64');
      expect((result as any).toArray()).toEqual([2, 3]);
    });

    it('computes mean with keepdims for int64', () => {
      const arr = array([[2, 4, 6]], 'int64');
      const result = arr.mean(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      // Mean returns float64
      expect((result as any).dtype).toBe('float64');
      expect((result as any).toArray()).toEqual([[4]]);
    });
  });

  describe('max() with different dtypes', () => {
    it('finds max of int32 array', () => {
      const arr = array([1, 5, 3, 9, 2], 'int32');
      expect(arr.max()).toBe(9);
    });

    it('finds max of int64 array', () => {
      const arr = array([10, 50, 30, 90, 20], 'int64');
      expect(arr.max()).toBe(90);
    });

    it('finds max of uint64 array', () => {
      const arr = array([100, 500, 300], 'uint64');
      expect(arr.max()).toBe(500);
    });

    it('finds max along axis for int64', () => {
      const arr = array(
        [
          [1, 5],
          [3, 2],
        ],
        'int64'
      );
      const result = arr.max(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(3));
      expect((result as any).data[1]).toBe(BigInt(5));
    });

    it('finds max with keepdims for uint64', () => {
      const arr = array([[5, 10, 3]], 'uint64');
      const result = arr.max(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(10));
    });

    it('finds max along different axes', () => {
      const arr = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'int32'
      );
      const result0 = arr.max(0);
      expect((result0 as any).toArray()).toEqual([4, 5, 6]);

      const result1 = arr.max(1);
      expect((result1 as any).toArray()).toEqual([3, 6]);
    });
  });

  describe('min() with different dtypes', () => {
    it('finds min of int32 array', () => {
      const arr = array([5, 1, 9, 2, 7], 'int32');
      expect(arr.min()).toBe(1);
    });

    it('finds min of int64 array', () => {
      const arr = array([50, 10, 90, 20, 70], 'int64');
      expect(arr.min()).toBe(10);
    });

    it('finds min of uint64 array', () => {
      const arr = array([500, 100, 300], 'uint64');
      expect(arr.min()).toBe(100);
    });

    it('finds min along axis for int64', () => {
      const arr = array(
        [
          [5, 1],
          [2, 3],
        ],
        'int64'
      );
      const result = arr.min(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(2));
      expect((result as any).data[1]).toBe(BigInt(1));
    });

    it('finds min with keepdims for uint64', () => {
      const arr = array([[10, 5, 15]], 'uint64');
      const result = arr.min(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(5));
    });

    it('finds min along different axes', () => {
      const arr = array(
        [
          [6, 5, 4],
          [3, 2, 1],
        ],
        'int32'
      );
      const result0 = arr.min(0);
      expect((result0 as any).toArray()).toEqual([3, 2, 1]);

      const result1 = arr.min(1);
      expect((result1 as any).toArray()).toEqual([4, 1]);
    });
  });

  describe('prod() with different dtypes', () => {
    it('computes product of int32 array', () => {
      const arr = array([2, 3, 4], 'int32');
      expect(arr.prod()).toBe(24);
    });

    it('computes product of int64 array', () => {
      const arr = array([2, 3, 4], 'int64');
      expect(arr.prod()).toBe(24);
    });

    it('computes product of uint64 array', () => {
      const arr = array([5, 6], 'uint64');
      expect(arr.prod()).toBe(30);
    });

    it('computes product along axis for int64', () => {
      const arr = array(
        [
          [2, 3],
          [4, 5],
        ],
        'int64'
      );
      const result = arr.prod(0);
      expect((result as any).dtype).toBe('int64');
      expect((result as any).data[0]).toBe(BigInt(8));
      expect((result as any).data[1]).toBe(BigInt(15));
    });

    it('computes product with keepdims for uint64', () => {
      const arr = array([[2, 3, 4]], 'uint64');
      const result = arr.prod(1, true);
      expect((result as any).shape).toEqual([1, 1]);
      expect((result as any).data[0]).toBe(BigInt(24));
    });
  });

  describe('Edge cases', () => {
    it('handles empty arrays for sum', () => {
      const arr = zeros([0], 'float64');
      expect(arr.sum()).toBe(0);
    });

    it('handles empty arrays for mean', () => {
      const arr = zeros([0], 'float64');
      // Mean of empty array is NaN
      expect(arr.mean()).toBeNaN();
    });

    it('handles single element arrays', () => {
      const arr = array([42]);
      expect(arr.sum()).toBe(42);
      expect(arr.mean()).toBe(42);
      expect(arr.max()).toBe(42);
      expect(arr.min()).toBe(42);
      expect(arr.prod()).toBe(42);
    });

    it('handles 1-d reductions', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.sum(0);
      // For 1-d arrays, sum along axis 0 returns scalar
      expect(result).toBe(15);
    });

    it('handles negative values in int64', () => {
      const arr = array([-5, -3, -1], 'int64');
      expect(arr.sum()).toBe(-9);
      expect(arr.max()).toBe(-1);
      expect(arr.min()).toBe(-5);
    });

    it('handles mixed positive and negative in int32', () => {
      const arr = array([-5, 10, -3, 8], 'int32');
      expect(arr.sum()).toBe(10);
      expect(arr.mean()).toBe(2.5);
      expect(arr.max()).toBe(10);
      expect(arr.min()).toBe(-5);
    });

    it('sums bool arrays', () => {
      const arr = array([1, 0, 1, 1, 0], 'bool');
      expect(arr.sum()).toBe(3);
    });

    it('computes mean of bool arrays', () => {
      const arr = array([1, 0, 1, 1, 0], 'bool');
      expect(arr.mean()).toBe(0.6);
    });
  });

  describe('3D arrays with different axes', () => {
    it('reduces 3D int64 array along axis 0', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('reduces 3D int64 array along axis 1', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(1);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('reduces 3D int64 array along axis 2', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        'int64'
      );
      const result = arr.sum(2);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).dtype).toBe('int64');
    });

    it('uses negative axis in 3D arrays', () => {
      const arr = array(
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        'int32'
      );
      const result = arr.sum(-1);
      expect((result as any).shape).toEqual([1, 2]);
    });
  });

  describe('std() and var() operations', () => {
    it('computes std of float64 array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.std();
      expect(result).toBeCloseTo(1.4142, 4);
    });

    it('computes var of float64 array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.var();
      expect(result).toBeCloseTo(2.0, 1);
    });

    it('computes std along axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.std(0);
      expect((result as any).shape).toEqual([3]);
    });

    it('computes var with keepdims', () => {
      const arr = array([[1, 2, 3]]);
      const result = arr.var(1, 0, true) as any;
      expect(result.shape).toEqual([1, 1]);
      // Variance of [1, 2, 3] is approximately 0.667
      const variance = result.toArray()[0][0];
      expect(variance).toBeCloseTo(0.6667, 3);
    });

    it('computes std of int32 array', () => {
      const arr = array([2, 4, 6, 8], 'int32');
      const result = arr.std();
      expect(result).toBeGreaterThan(0);
    });
  });

  describe('Error handling', () => {
    it('throws error for invalid axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => arr.sum(5)).toThrow(/out of bounds/);
      expect(() => arr.sum(-5)).toThrow(/out of bounds/);
    });

    it('throws error for invalid axis in 3D', () => {
      const arr = array([[[1, 2]]]);
      expect(() => arr.sum(3)).toThrow(/out of bounds/);
      expect(() => arr.sum(-4)).toThrow(/out of bounds/);
    });
  });

  describe('cumsum()', () => {
    it('computes cumulative sum of all elements (flattened)', () => {
      const arr = array([1, 2, 3, 4]);
      const result = arr.cumsum();
      expect((result as any).toArray()).toEqual([1, 3, 6, 10]);
    });

    it('computes cumulative sum of 2D array (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumsum();
      expect((result as any).toArray()).toEqual([1, 3, 6, 10]);
    });

    it('computes cumulative sum along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumsum(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).toArray()).toEqual([
        [1, 2],
        [4, 6],
      ]);
    });

    it('computes cumulative sum along axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.cumsum(1);
      expect((result as any).shape).toEqual([2, 3]);
      expect((result as any).toArray()).toEqual([
        [1, 3, 6],
        [4, 9, 15],
      ]);
    });

    it('computes cumulative sum along axis=-1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumsum(-1);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).toArray()).toEqual([
        [1, 3],
        [3, 7],
      ]);
    });
  });

  describe('cumprod()', () => {
    it('computes cumulative product of all elements (flattened)', () => {
      const arr = array([1, 2, 3, 4]);
      const result = arr.cumprod();
      expect((result as any).toArray()).toEqual([1, 2, 6, 24]);
    });

    it('computes cumulative product of 2D array (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumprod();
      expect((result as any).toArray()).toEqual([1, 2, 6, 24]);
    });

    it('computes cumulative product along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.cumprod(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).toArray()).toEqual([
        [1, 2],
        [3, 8],
      ]);
    });

    it('computes cumulative product along axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.cumprod(1);
      expect((result as any).shape).toEqual([2, 3]);
      expect((result as any).toArray()).toEqual([
        [1, 2, 6],
        [4, 20, 120],
      ]);
    });
  });

  describe('ptp()', () => {
    it('computes peak-to-peak (range) of all elements', () => {
      const arr = array([1, 5, 3, 9, 2]);
      expect(arr.ptp()).toBe(8);
    });

    it('computes ptp of 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      expect(arr.ptp()).toBe(8);
    });

    it('computes ptp along axis=0', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([8, 3, 3]);
    });

    it('computes ptp along axis=1', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([4, 7]);
    });

    it('computes ptp with keepdims=true', () => {
      const arr = array([
        [1, 5, 3],
        [9, 2, 6],
      ]);
      const result = arr.ptp(0, true);
      expect((result as any).shape).toEqual([1, 3]);
      expect((result as any).toArray()).toEqual([[8, 3, 3]]);
    });
  });

  describe('median()', () => {
    it('computes median of odd-length array', () => {
      const arr = array([1, 3, 2, 5, 4]);
      expect(arr.median()).toBe(3);
    });

    it('computes median of even-length array', () => {
      const arr = array([1, 2, 3, 4]);
      expect(arr.median()).toBe(2.5);
    });

    it('computes median of 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.median()).toBe(3.5);
    });

    it('computes median along axis=0', () => {
      const arr = array([
        [1, 3, 5],
        [2, 4, 6],
      ]);
      const result = arr.median(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([1.5, 3.5, 5.5]);
    });

    it('computes median along axis=1', () => {
      const arr = array([
        [1, 3, 2],
        [6, 4, 5],
      ]);
      const result = arr.median(1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5]);
    });

    it('computes median with keepdims=true', () => {
      const arr = array([
        [1, 3, 2],
        [6, 4, 5],
      ]);
      const result = arr.median(1, true);
      expect((result as any).shape).toEqual([2, 1]);
      expect((result as any).toArray()).toEqual([[2], [5]]);
    });
  });

  describe('percentile()', () => {
    it('computes 50th percentile (median)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.percentile(50)).toBe(3);
    });

    it('computes 0th percentile (min)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.percentile(0)).toBe(1);
    });

    it('computes 100th percentile (max)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.percentile(100)).toBe(5);
    });

    it('computes 25th percentile', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.percentile(25);
      expect(result).toBeCloseTo(2, 5);
    });

    it('computes percentile along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = arr.percentile(50, 0);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([3, 4]);
    });

    it('computes percentile with keepdims=true', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.percentile(50, 0, true);
      expect((result as any).shape).toEqual([1, 2]);
    });
  });

  describe('quantile()', () => {
    it('computes 0.5 quantile (median)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.quantile(0.5)).toBe(3);
    });

    it('computes 0.0 quantile (min)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.quantile(0)).toBe(1);
    });

    it('computes 1.0 quantile (max)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.quantile(1)).toBe(5);
    });

    it('computes 0.25 quantile', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.quantile(0.25);
      expect(result).toBeCloseTo(2, 5);
    });

    it('computes quantile along axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.quantile(0.5, 1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5]);
    });
  });

  describe('average()', () => {
    it('computes unweighted average', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.average()).toBe(3);
    });

    it('computes weighted average', () => {
      const arr = array([1, 2, 3]);
      const weights = array([3, 2, 1]);
      // (1*3 + 2*2 + 3*1) / (3+2+1) = 10/6 = 1.666...
      expect(arr.average(weights)).toBeCloseTo(10 / 6, 5);
    });

    it('computes average along axis=0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.average(undefined, 0);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 3]);
    });

    it('computes average along axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.average(undefined, 1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5]);
    });

    it('computes weighted average along axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const weights = array([3, 1]);
      const result = arr.average(weights, 1);
      expect((result as any).shape).toEqual([2]);
      // Row 0: (1*3 + 2*1) / 4 = 5/4 = 1.25
      // Row 1: (3*3 + 4*1) / 4 = 13/4 = 3.25
      expect((result as any).toArray()[0]).toBeCloseTo(1.25, 5);
      expect((result as any).toArray()[1]).toBeCloseTo(3.25, 5);
    });
  });

  describe('nansum()', () => {
    it('sums array ignoring NaN values', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      expect(arr.nansum()).toBe(9);
    });

    it('sums all non-NaN elements in 2D array', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      expect(arr.nansum()).toBe(13);
    });

    it('sums along axis=0 ignoring NaN', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nansum(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([5, 5, 3]);
    });

    it('sums along axis=1 ignoring NaN', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nansum(1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([4, 11]);
    });

    it('handles array with all NaN returning 0', () => {
      const arr = array([NaN, NaN, NaN]);
      expect(arr.nansum()).toBe(0);
    });
  });

  describe('nanprod()', () => {
    it('computes product ignoring NaN values', () => {
      const arr = array([2, NaN, 3, NaN, 4]);
      expect(arr.nanprod()).toBe(24);
    });

    it('handles 2D array with NaN', () => {
      const arr = array([
        [2, NaN],
        [3, 4],
      ]);
      expect(arr.nanprod()).toBe(24);
    });

    it('computes product along axis ignoring NaN', () => {
      const arr = array([
        [2, NaN],
        [3, 4],
      ]);
      const result = arr.nanprod(0);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([6, 4]);
    });
  });

  describe('nanmean()', () => {
    it('computes mean ignoring NaN values', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      expect(arr.nanmean()).toBe(3);
    });

    it('handles 2D array with NaN', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      // (1 + 3 + 4 + 5) / 4 = 13/4 = 3.25
      expect(arr.nanmean()).toBe(3.25);
    });

    it('computes mean along axis ignoring NaN', () => {
      const arr = array([
        [1, NaN, 3],
        [4, 5, NaN],
      ]);
      const result = arr.nanmean(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([2.5, 5, 3]);
    });
  });

  describe('nanvar()', () => {
    it('computes variance ignoring NaN values', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      // Mean of [1, 2, 3] = 2, variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
      expect(arr.nanvar()).toBeCloseTo(2 / 3, 5);
    });

    it('computes variance along axis ignoring NaN', () => {
      const arr = array([
        [1, NaN],
        [2, 4],
      ]);
      const result = arr.nanvar(0);
      expect((result as any).shape).toEqual([2]);
      // Column 0: mean=1.5, var=((1-1.5)^2 + (2-1.5)^2)/2 = 0.25
      expect((result as any).toArray()[0]).toBeCloseTo(0.25, 5);
      // Column 1: only 4, var=0
      expect((result as any).toArray()[1]).toBe(0);
    });
  });

  describe('nanstd()', () => {
    it('computes std ignoring NaN values', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      // variance = 2/3, std = sqrt(2/3)
      expect(arr.nanstd()).toBeCloseTo(Math.sqrt(2 / 3), 5);
    });

    it('computes std along axis ignoring NaN', () => {
      const arr = array([
        [1, NaN],
        [3, 4],
      ]);
      const result = arr.nanstd(0);
      expect((result as any).shape).toEqual([2]);
      // Column 0: mean=2, var=1, std=1
      expect((result as any).toArray()[0]).toBeCloseTo(1, 5);
      // Column 1: only 4, std=0
      expect((result as any).toArray()[1]).toBe(0);
    });
  });

  describe('nanmin()', () => {
    it('finds minimum ignoring NaN values', () => {
      const arr = array([NaN, 3, NaN, 1, NaN, 5]);
      expect(arr.nanmin()).toBe(1);
    });

    it('handles 2D array with NaN', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 2],
      ]);
      expect(arr.nanmin()).toBe(2);
    });

    it('finds min along axis ignoring NaN', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 2],
      ]);
      const result = arr.nanmin(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([9, 5, 2]);
    });
  });

  describe('nanmax()', () => {
    it('finds maximum ignoring NaN values', () => {
      const arr = array([NaN, 3, NaN, 9, NaN, 5]);
      expect(arr.nanmax()).toBe(9);
    });

    it('handles 2D array with NaN', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 2],
      ]);
      expect(arr.nanmax()).toBe(9);
    });

    it('finds max along axis ignoring NaN', () => {
      const arr = array([
        [NaN, 5, 3],
        [9, NaN, 8],
      ]);
      const result = arr.nanmax(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([9, 5, 8]);
    });
  });

  describe('nanargmin()', () => {
    it('finds index of minimum ignoring NaN values', () => {
      const arr = array([NaN, 3, NaN, 1, NaN, 5]);
      expect(arr.nanargmin()).toBe(3);
    });

    it('finds indices of min along axis ignoring NaN', () => {
      const arr = array([
        [NaN, 5],
        [2, NaN],
      ]);
      const result = arr.nanargmin(0);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([1, 0]);
    });
  });

  describe('nanargmax()', () => {
    it('finds index of maximum ignoring NaN values', () => {
      const arr = array([NaN, 3, NaN, 9, NaN, 5]);
      expect(arr.nanargmax()).toBe(3);
    });

    it('finds indices of max along axis ignoring NaN', () => {
      const arr = array([
        [NaN, 5],
        [2, NaN],
      ]);
      const result = arr.nanargmax(0);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([1, 0]);
    });
  });

  describe('nancumsum()', () => {
    it('computes cumulative sum treating NaN as 0', () => {
      const arr = array([1, NaN, 3, NaN, 5]);
      const result = arr.nancumsum();
      expect((result as any).toArray()).toEqual([1, 1, 4, 4, 9]);
    });

    it('computes nancumsum along axis', () => {
      const arr = array([
        [1, NaN],
        [NaN, 4],
      ]);
      const result = arr.nancumsum(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).toArray()).toEqual([
        [1, 0],
        [1, 4],
      ]);
    });
  });

  describe('nancumprod()', () => {
    it('computes cumulative product treating NaN as 1', () => {
      const arr = array([2, NaN, 3, NaN, 4]);
      const result = arr.nancumprod();
      expect((result as any).toArray()).toEqual([2, 2, 6, 6, 24]);
    });

    it('computes nancumprod along axis', () => {
      const arr = array([
        [2, NaN],
        [NaN, 3],
      ]);
      const result = arr.nancumprod(0);
      expect((result as any).shape).toEqual([2, 2]);
      expect((result as any).toArray()).toEqual([
        [2, 1],
        [2, 3],
      ]);
    });
  });

  describe('nanmedian()', () => {
    it('computes median ignoring NaN values', () => {
      const arr = array([1, NaN, 2, NaN, 3]);
      expect(arr.nanmedian()).toBe(2);
    });

    it('computes nanmedian of even-length non-NaN elements', () => {
      const arr = array([1, NaN, 2, NaN, 3, 4]);
      expect(arr.nanmedian()).toBe(2.5);
    });

    it('computes nanmedian along axis', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanmedian(0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([1, 5, 4.5]);
    });

    it('returns NaN when all values are NaN', () => {
      const arr = array([NaN, NaN, NaN]);
      expect(arr.nanmedian()).toBeNaN();
    });

    it('computes nanmedian with negative axis', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanmedian(-1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5.5]);
    });

    it('returns NaN for slices with all NaN along axis', () => {
      const arr = array([
        [NaN, NaN],
        [1, 2],
      ]);
      const result = arr.nanmedian(1);
      expect((result as any).shape).toEqual([2]);
      const values = (result as any).toArray();
      expect(values[0]).toBeNaN();
      expect(values[1]).toBe(1.5);
    });

    it('computes nanmedian with keepdims=true', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanmedian(0, true);
      expect((result as any).shape).toEqual([1, 3]);
    });
  });

  describe('nanquantile()', () => {
    it('computes 0.5 quantile ignoring NaN values', () => {
      const arr = array([1, NaN, 2, NaN, 3, 4, 5]);
      expect(arr.nanquantile(0.5)).toBe(3);
    });

    it('computes 0.0 quantile (min) ignoring NaN', () => {
      const arr = array([NaN, 1, 2, 3, NaN]);
      expect(arr.nanquantile(0)).toBe(1);
    });

    it('computes 1.0 quantile (max) ignoring NaN', () => {
      const arr = array([NaN, 1, 2, 3, NaN]);
      expect(arr.nanquantile(1)).toBe(3);
    });

    it('computes 0.25 quantile with interpolation', () => {
      const arr = array([1, NaN, 2, 3, 4, NaN]);
      const result = arr.nanquantile(0.25);
      expect(result).toBeCloseTo(1.75, 5);
    });

    it('returns NaN when all values are NaN', () => {
      const arr = array([NaN, NaN, NaN]);
      expect(arr.nanquantile(0.5)).toBeNaN();
    });

    it('throws for quantile < 0', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.nanquantile(-0.1)).toThrow('Quantile must be between 0 and 1');
    });

    it('throws for quantile > 1', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.nanquantile(1.1)).toThrow('Quantile must be between 0 and 1');
    });

    it('computes nanquantile along axis=0', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
        [7, 8, NaN],
      ]);
      const result = arr.nanquantile(0.5, 0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([4, 6.5, 4.5]);
    });

    it('computes nanquantile along axis=1', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanquantile(0.5, 1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5.5]);
    });

    it('computes nanquantile with keepdims=true', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanquantile(0.5, 0, true);
      expect((result as any).shape).toEqual([1, 3]);
    });

    it('computes nanquantile with negative axis', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanquantile(0.5, -1);
      expect((result as any).shape).toEqual([2]);
      expect((result as any).toArray()).toEqual([2, 5.5]);
    });

    it('throws for out of bounds axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => arr.nanquantile(0.5, 5)).toThrow('axis 5 is out of bounds');
    });

    it('returns NaN for slices with all NaN values along axis', () => {
      const arr = array([
        [NaN, NaN],
        [1, 2],
      ]);
      const result = arr.nanquantile(0.5, 1);
      expect((result as any).shape).toEqual([2]);
      const values = (result as any).toArray();
      expect(values[0]).toBeNaN();
      expect(values[1]).toBe(1.5);
    });
  });

  describe('nanpercentile()', () => {
    it('computes 50th percentile ignoring NaN values', () => {
      const arr = array([1, NaN, 2, NaN, 3, 4, 5]);
      expect(arr.nanpercentile(50)).toBe(3);
    });

    it('computes 0th percentile (min) ignoring NaN', () => {
      const arr = array([NaN, 1, 2, 3, NaN]);
      expect(arr.nanpercentile(0)).toBe(1);
    });

    it('computes 100th percentile (max) ignoring NaN', () => {
      const arr = array([NaN, 1, 2, 3, NaN]);
      expect(arr.nanpercentile(100)).toBe(3);
    });

    it('computes 25th percentile with interpolation', () => {
      const arr = array([1, NaN, 2, 3, 4, NaN]);
      const result = arr.nanpercentile(25);
      expect(result).toBeCloseTo(1.75, 5);
    });

    it('computes nanpercentile along axis=0', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
        [7, 8, NaN],
      ]);
      const result = arr.nanpercentile(50, 0);
      expect((result as any).shape).toEqual([3]);
      expect((result as any).toArray()).toEqual([4, 6.5, 4.5]);
    });

    it('computes nanpercentile with keepdims=true', () => {
      const arr = array([
        [1, NaN, 3],
        [NaN, 5, 6],
      ]);
      const result = arr.nanpercentile(50, 0, true);
      expect((result as any).shape).toEqual([1, 3]);
    });
  });

  // ========================================
  // Standalone function: BigInt operations
  // ========================================
  describe('BigInt operations (standalone functions)', () => {
    it('sum on int64', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const r = sum(a);
      expect(r).toBeDefined();
    });

    it('mean on int64 (promotes to float)', () => {
      const a = array([2n, 4n, 6n], 'int64');
      const r = mean(a);
      expect(r).toBeDefined();
    });

    it('cumsum on int64', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const r = cumsum(a);
      expect(r.size).toBe(3);
    });

    it('diff on int64', () => {
      const a = array([1n, 3n, 6n, 10n], 'int64');
      const r = diff(a);
      expect(r.size).toBe(3);
    });
  });

  // ========================================
  // Standalone function: Reduction keepdims & axis branches
  // ========================================
  describe('Reduction keepdims & axis branches (standalone functions)', () => {
    it('sum with keepdims', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = sum(a, 1, true);
      expect(r.shape).toEqual([2, 1]);
    });

    it('mean with keepdims', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = mean(a, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });

    it('prod with axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = prod(a, 1);
      expect(r.toArray()).toEqual([2, 12]);
    });

    it('std with axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = std(a, 0);
      expect(r.size).toBe(2);
    });

    it('ptp with axis', () => {
      const a = array([
        [1, 5],
        [2, 8],
      ]);
      const r = ptp(a, 1);
      expect(r.toArray()).toEqual([4, 6]);
    });

    it('median with keepdims', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = median(a, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });

    it('cumsum with axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = cumsum(a, 0);
      expect(r.toArray()).toEqual([
        [1, 2],
        [4, 6],
      ]);
    });

    it('cumprod with axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const r = cumprod(a, 0);
      expect(r.toArray()).toEqual([
        [1, 2],
        [3, 8],
      ]);
    });

    it('nansum with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      const r = nansum(a, 1);
      expect(r.toArray()).toEqual([1, 7]);
    });

    it('nanmean with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      const r = nanmean(a, 1);
      expect(r.size).toBe(2);
    });

    it('nanprod with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      const r = nanprod(a, 1);
      expect(r.size).toBe(2);
    });

    it('average with weights', () => {
      const a = array([1, 2, 3, 4]);
      const w = array([4, 3, 2, 1]);
      const r = average(a, undefined, w);
      expect(r).toBeDefined();
    });
  });

  // ========================================
  // Standalone function: Reduction branches (dtype & axis variations)
  // ========================================
  describe('Reduction branches (standalone functions)', () => {
    it('sum bool array', () => {
      const a = array([1, 0, 1, 1], 'bool');
      expect(sum(a)).toBeDefined();
    });

    it('cumsum int32', () => {
      const a = array([1, 2, 3], 'int32');
      expect(cumsum(a).size).toBe(3);
    });

    it('cumprod float32', () => {
      const a = array([1, 2, 3], 'float32');
      expect(cumprod(a).size).toBe(3);
    });

    it('ptp 2D axis=0', () => {
      const a = array([
        [1, 5],
        [2, 8],
      ]);
      expect(ptp(a, 0).size).toBe(2);
    });

    it('median float32', () => {
      const a = array([3, 1, 2], 'float32');
      expect(median(a)).toBeDefined();
    });

    it('percentile with axis', () => {
      const a = array([
        [10, 20],
        [30, 40],
      ]);
      expect(percentile(a, 50, 0).size).toBe(2);
    });

    it('quantile with axis', () => {
      const a = array([
        [10, 20],
        [30, 40],
      ]);
      expect(quantile(a, 0.5, 0).size).toBe(2);
    });

    it('average with axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(average(a, 1).size).toBe(2);
    });

    it('nanvar with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      expect(nanvar(a, 1).size).toBe(2);
    });

    it('nanstd with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      expect(nanstd(a, 1).size).toBe(2);
    });

    it('nanmin/nanmax axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      expect(nanmin(a, 1).size).toBe(2);
      expect(nanmax(a, 1).size).toBe(2);
    });

    it('nanargmin/nanargmax', () => {
      const a = array([NaN, 2, 1, 3]);
      expect(nanargmin(a)).toBeDefined();
      expect(nanargmax(a)).toBeDefined();
    });

    it('nancumsum/nancumprod', () => {
      const a = array([1, NaN, 3]);
      expect(nancumsum(a).size).toBe(3);
      expect(nancumprod(a).size).toBe(3);
    });

    it('nanmedian with axis', () => {
      const a = array([
        [1, NaN],
        [3, 4],
      ]);
      expect(nanmedian(a, 1).size).toBe(2);
    });

    it('nanquantile', () => {
      const a = array([1, NaN, 3, 4]);
      expect(nanquantile(a, 0.5)).toBeDefined();
    });

    it('nanpercentile', () => {
      const a = array([1, NaN, 3, 4]);
      expect(nanpercentile(a, 50)).toBeDefined();
    });
  });
});
