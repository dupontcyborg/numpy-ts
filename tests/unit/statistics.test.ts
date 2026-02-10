import { describe, it, expect } from 'vitest';
import {
  array,
  bincount,
  digitize,
  histogram,
  histogram2d,
  histogramdd,
  correlate,
  convolve,
  cov,
  corrcoef,
  histogram_bin_edges,
  trapezoid,
} from '../../src';

describe('Statistics Operations', () => {
  describe('bincount', () => {
    it('counts occurrences of non-negative integers', () => {
      const arr = array([0, 1, 1, 3, 2, 1, 7]);
      const result = bincount(arr);
      // 0: 1, 1: 3, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1
      expect(result.toArray()).toEqual([1, 3, 1, 1, 0, 0, 0, 1]);
    });

    it('respects minlength parameter', () => {
      const arr = array([0, 1, 1]);
      const result = bincount(arr, undefined, 5);
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([1, 2, 0, 0, 0]);
    });

    it('supports weights', () => {
      const arr = array([0, 1, 1, 2]);
      const weights = array([0.5, 1.0, 0.5, 2.0]);
      const result = bincount(arr, weights);
      // 0: 0.5, 1: 1.5 (1.0 + 0.5), 2: 2.0
      expect(result.toArray()).toEqual([0.5, 1.5, 2.0]);
    });

    it('throws for negative integers', () => {
      const arr = array([-1, 0, 1]);
      expect(() => bincount(arr)).toThrow('non-negative integers');
    });

    it('handles empty array', () => {
      const arr = array([], 'int32');
      const result = bincount(arr, undefined, 3);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([0, 0, 0]);
    });
  });

  describe('digitize', () => {
    it('bins values into increasing bins', () => {
      const x = array([0.2, 6.4, 3.0, 1.6]);
      const bins = array([0.0, 1.0, 2.5, 4.0, 10.0]);
      const result = digitize(x, bins);
      // 0.2 goes to bin 1, 6.4 goes to bin 4, 3.0 goes to bin 3, 1.6 goes to bin 2
      expect(result.toArray()).toEqual([1, 4, 3, 2]);
    });

    it('handles right=true', () => {
      const x = array([0.0, 1.0, 2.5]);
      const bins = array([0.0, 1.0, 2.5, 4.0]);
      const resultLeft = digitize(x, bins, false);
      const resultRight = digitize(x, bins, true);
      // With right=false, value equals bin edge -> goes to next bin
      // With right=true, value equals bin edge -> stays in current bin
      expect(resultLeft.toArray()).toEqual([1, 2, 3]);
      expect(resultRight.toArray()).toEqual([0, 1, 2]);
    });

    it('handles decreasing bins', () => {
      const x = array([0.5, 1.5, 2.5]);
      const bins = array([4.0, 2.0, 1.0, 0.0]);
      const result = digitize(x, bins);
      expect(result.toArray()).toEqual([3, 2, 1]);
    });

    it('handles values outside bin range', () => {
      const x = array([-1, 5, 2]);
      const bins = array([0, 1, 2, 3]);
      const result = digitize(x, bins);
      expect(result.toArray()).toEqual([0, 4, 3]);
    });
  });

  describe('histogram', () => {
    it('computes histogram with default bins', () => {
      const arr = array([1, 2, 1, 3, 4, 2, 2, 3, 3, 3]);
      const [hist, binEdges] = histogram(arr, 4);
      expect(hist.shape).toEqual([4]);
      expect(binEdges.shape).toEqual([5]);
      // Check that histogram sums to total count
      const histArr = hist.toArray() as number[];
      expect(histArr.reduce((a, b) => a + b, 0)).toBe(10);
    });

    it('computes histogram with specific range', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const [hist, binEdges] = histogram(arr, 5, [0, 5]);
      expect(hist.shape).toEqual([5]);
      const edges = binEdges.toArray() as number[];
      expect(edges[0]).toBe(0);
      expect(edges[5]).toBe(5);
    });

    it('computes histogram with custom bin edges', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const bins = array([0, 2, 4, 6]);
      const [hist, binEdges] = histogram(arr, bins);
      expect(hist.toArray()).toEqual([1, 2, 2]);
      expect(binEdges.toArray()).toEqual([0, 2, 4, 6]);
    });

    it('computes density histogram', () => {
      const arr = array([1, 2, 1, 2, 3, 4]);
      const [hist] = histogram(arr, 4, undefined, true);
      // Density should integrate to 1
      const histArr = hist.toArray() as number[];
      const binWidth = 0.75; // (4-1)/4
      const integral = histArr.reduce((a, b) => a + b * binWidth, 0);
      expect(Math.abs(integral - 1)).toBeLessThan(1e-10);
    });

    it('supports weights', () => {
      const arr = array([1, 2, 3]);
      const weights = array([2, 1, 3]);
      const [hist] = histogram(arr, 3, [1, 4], false, weights);
      expect(hist.toArray()).toEqual([2, 1, 3]);
    });
  });

  describe('histogram2d', () => {
    it('computes 2D histogram', () => {
      const x = array([0, 0, 1, 1]);
      const y = array([0, 1, 0, 1]);
      const [hist, xEdges, yEdges] = histogram2d(x, y, 2);
      expect(hist.shape).toEqual([2, 2]);
      expect(xEdges.shape).toEqual([3]);
      expect(yEdges.shape).toEqual([3]);
      // Each quadrant should have 1 point
      expect(hist.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    it('handles different bin counts per axis', () => {
      const x = array([0, 0.5, 1, 1.5, 2]);
      const y = array([0, 1, 0, 1, 0]);
      const [hist] = histogram2d(x, y, [2, 2]);
      expect(hist.shape).toEqual([2, 2]);
    });

    it('throws for mismatched x and y lengths', () => {
      const x = array([1, 2, 3]);
      const y = array([1, 2]);
      expect(() => histogram2d(x, y)).toThrow('same length');
    });
  });

  describe('histogramdd', () => {
    it('computes 1D histogram from 1D array', () => {
      const sample = array([1, 2, 3, 4, 5]);
      const [hist, edges] = histogramdd(sample, 5);
      expect(hist.shape).toEqual([5]);
      expect(edges.length).toBe(1);
      expect(edges[0]!.shape).toEqual([6]);
    });

    it('computes 2D histogram from (N, 2) array', () => {
      const sample = array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);
      const [hist, edges] = histogramdd(sample, 2);
      expect(hist.shape).toEqual([2, 2]);
      expect(edges.length).toBe(2);
    });

    it('respects per-dimension bin counts', () => {
      const sample = array([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);
      const [hist] = histogramdd(sample, [3, 2]);
      expect(hist.shape).toEqual([3, 2]);
    });
  });

  describe('correlate', () => {
    it('computes cross-correlation in full mode', () => {
      const a = array([1, 2, 3]);
      const v = array([0, 1, 0.5]);
      const result = correlate(a, v, 'full');
      expect(result.shape).toEqual([5]);
      // Full correlation length = len(a) + len(v) - 1 = 5
    });

    it('computes cross-correlation in same mode', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = array([1, 2, 3]);
      const result = correlate(a, v, 'same');
      expect(result.shape).toEqual([5]); // Same as input a
    });

    it('computes cross-correlation in valid mode', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = array([1, 2, 3]);
      const result = correlate(a, v, 'valid');
      expect(result.shape).toEqual([3]); // max - min + 1 = 5 - 3 + 1 = 3
    });

    it('auto-correlation gives maximum at center', () => {
      const a = array([1, 2, 3, 2, 1]);
      const result = correlate(a, a, 'full');
      const resultArr = result.toArray() as number[];
      const maxIdx = resultArr.indexOf(Math.max(...resultArr));
      expect(maxIdx).toBe(4); // Center of full correlation
    });
  });

  describe('convolve', () => {
    it('computes convolution in full mode', () => {
      const a = array([1, 2, 3]);
      const v = array([0, 1, 0.5]);
      const result = convolve(a, v, 'full');
      expect(result.shape).toEqual([5]);
      // Convolution: [1*0, 1*1+2*0, 1*0.5+2*1+3*0, 2*0.5+3*1, 3*0.5]
      //            = [0, 1, 2.5, 4, 1.5]
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[2]! - 2.5)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[3]! - 4)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[4]! - 1.5)).toBeLessThan(1e-10);
    });

    it('computes convolution in same mode', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = array([1, 2, 3]);
      const result = convolve(a, v, 'same');
      expect(result.shape).toEqual([5]); // Same as input a
    });

    it('computes convolution in valid mode', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = array([1, 2, 3]);
      const result = convolve(a, v, 'valid');
      expect(result.shape).toEqual([3]);
    });

    it('convolution with [1] is identity', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = array([1]);
      const result = convolve(a, v, 'same');
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('cov', () => {
    it('computes variance for 1D array (returns 0-d scalar like NumPy)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = cov(arr);
      expect(result.shape).toEqual([]);
      // Variance with ddof=1: sum((x - mean)^2) / (n-1)
      // mean = 3, sum of squares = (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 10
      // variance = 10 / 4 = 2.5
      expect(Math.abs((result.get([]) as number) - 2.5)).toBeLessThan(1e-10);
    });

    it('computes covariance matrix for 2D array (rowvar=true)', () => {
      const arr = array([
        [0, 2],
        [1, 1],
        [2, 0],
      ]);
      // 3 variables (rows), 2 observations (columns)
      const result = cov(arr);
      expect(result.shape).toEqual([3, 3]);
      // Check that diagonal contains variances
    });

    it('computes covariance with rowvar=false', () => {
      const arr = array([
        [0, 1, 2],
        [2, 1, 0],
      ]);
      // 2 observations (rows), 3 variables (columns)
      const result = cov(arr, undefined, false);
      expect(result.shape).toEqual([3, 3]);
    });

    it('computes covariance of two 1D arrays', () => {
      const x = array([1, 2, 3, 4, 5]);
      const y = array([5, 4, 3, 2, 1]);
      const result = cov(x, y);
      expect(result.shape).toEqual([2, 2]);
      // x and y are negatively correlated
      const covArr = result.toArray() as number[][];
      expect(covArr[0]![1]!).toBeLessThan(0);
      expect(covArr[1]![0]!).toBeLessThan(0);
    });

    it('respects bias parameter', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const resultUnbiased = cov(arr, undefined, true, false);
      const resultBiased = cov(arr, undefined, true, true);
      // Biased estimate divides by n, unbiased by n-1
      // So biased result should be smaller (multiplied by (n-1)/n = 4/5)
      const unbiased = resultUnbiased.get([]) as number;
      const biased = resultBiased.get([]) as number;
      expect(Math.abs(biased / unbiased - 0.8)).toBeLessThan(1e-10);
    });

    it('respects ddof parameter', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result0 = cov(arr, undefined, true, false, 0); // ddof=0 (biased)
      const result1 = cov(arr, undefined, true, false, 1); // ddof=1 (unbiased)
      const result2 = cov(arr, undefined, true, false, 2); // ddof=2
      const v0 = result0.get([]) as number;
      const v1 = result1.get([]) as number;
      const v2 = result2.get([]) as number;
      expect(v0).toBeLessThan(v1);
      expect(v1).toBeLessThan(v2);
    });
  });

  describe('corrcoef', () => {
    it('computes correlation for 1D array (returns 0-d scalar of 1 like NumPy)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = corrcoef(arr);
      expect(result.shape).toEqual([]);
      expect(result.get([])).toBe(1);
    });

    it('computes correlation coefficient matrix', () => {
      const arr = array([
        [1, 2, 3],
        [3, 2, 1],
      ]);
      const result = corrcoef(arr);
      expect(result.shape).toEqual([2, 2]);
      const corrArr = result.toArray() as number[][];
      // Diagonal should be 1
      expect(Math.abs(corrArr[0]![0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(corrArr[1]![1]! - 1)).toBeLessThan(1e-10);
      // Off-diagonal should be -1 (perfect negative correlation)
      expect(Math.abs(corrArr[0]![1]! + 1)).toBeLessThan(1e-10);
      expect(Math.abs(corrArr[1]![0]! + 1)).toBeLessThan(1e-10);
    });

    it('computes correlation of two 1D arrays', () => {
      const x = array([1, 2, 3, 4, 5]);
      const y = array([2, 4, 6, 8, 10]);
      const result = corrcoef(x, y);
      expect(result.shape).toEqual([2, 2]);
      const corrArr = result.toArray() as number[][];
      // x and y are perfectly correlated
      expect(Math.abs(corrArr[0]![1]! - 1)).toBeLessThan(1e-10);
    });

    it('returns values in [-1, 1] range', () => {
      const arr = array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 3, 2, 4],
      ]);
      const result = corrcoef(arr);
      const corrArr = result.toArray() as number[][];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(corrArr[i]![j]!).toBeGreaterThanOrEqual(-1);
          expect(corrArr[i]![j]!).toBeLessThanOrEqual(1);
        }
      }
    });
  });

  describe('histogram_bin_edges', () => {
    it('computes bin edges with default bins', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const edges = histogram_bin_edges(arr, 5);
      expect(edges.shape).toEqual([6]);
      const edgesArr = edges.toArray() as number[];
      expect(edgesArr[0]).toBe(1);
      expect(edgesArr[5]).toBe(5);
    });

    it('computes bin edges with specified range', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const edges = histogram_bin_edges(arr, 4, [0, 8]);
      expect(edges.shape).toEqual([5]);
      const edgesArr = edges.toArray() as number[];
      expect(edgesArr[0]).toBe(0);
      expect(edgesArr[4]).toBe(8);
    });

    it('uses auto bin selection', () => {
      const arr = array([1, 2, 2, 3, 3, 3, 4, 5]);
      const edges = histogram_bin_edges(arr, 'auto');
      expect(edges.size).toBeGreaterThan(1);
    });

    it('uses sturges bin selection', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const edges = histogram_bin_edges(arr, 'sturges');
      // Sturges: ceil(log2(n) + 1) = ceil(log2(10) + 1) = 5
      expect(edges.shape).toEqual([6]); // 5 bins = 6 edges
    });

    it('uses sqrt bin selection', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      const edges = histogram_bin_edges(arr, 'sqrt');
      // sqrt(16) = 4 bins
      expect(edges.shape).toEqual([5]); // 4 bins = 5 edges
    });

    it('handles single value', () => {
      const arr = array([5, 5, 5, 5]);
      const edges = histogram_bin_edges(arr, 3);
      expect(edges.shape).toEqual([4]);
      // When all values are same, range is expanded to [val - 0.5, val + 0.5]
    });

    it('handles empty array with at least 1 bin', () => {
      const arr = array([], 'float64');
      const edges = histogram_bin_edges(arr, 5);
      expect(edges.size).toBeGreaterThanOrEqual(2);
    });
  });

  describe('trapezoid', () => {
    it('integrates constant function', () => {
      // Integral of y=2 from x=0 to x=4 is 8
      const y = array([2, 2, 2, 2, 2]);
      const result = trapezoid(y);
      expect(result).toBe(8); // 4 intervals of width 1, height 2
    });

    it('integrates linear function', () => {
      // Integral of y=x from x=0 to x=4 is 8
      const y = array([0, 1, 2, 3, 4]);
      const result = trapezoid(y);
      expect(result).toBe(8);
    });

    it('integrates with custom dx', () => {
      // Integral of y=1 from x=0 to x=2 with dx=0.5
      const y = array([1, 1, 1, 1, 1]);
      const result = trapezoid(y, undefined, 0.5);
      expect(result).toBe(2); // 4 intervals of width 0.5, height 1
    });

    it('integrates with custom x values', () => {
      // Integral of y=x from x=0 to x=2 (non-uniform spacing)
      const y = array([0, 1, 4]);
      const x = array([0, 1, 2]);
      const result = trapezoid(y, x);
      // Area = 0.5*(0+1)*1 + 0.5*(1+4)*1 = 0.5 + 2.5 = 3
      expect(result).toBe(3);
    });

    it('integrates along axis for 2D array', () => {
      // Each row is [1, 2, 3, 4], integral should be 7.5 for each row
      const y = array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
      ]);
      const result = trapezoid(y, undefined, 1, 1);
      expect((result as { toArray: () => number[] }).toArray()).toEqual([7.5, 15]);
    });

    it('integrates along axis 0 for 2D array', () => {
      const y = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = trapezoid(y, undefined, 1, 0);
      // For each column: integrate over 2 points
      // Column 0: 0.5*(1+4)*1 = 2.5
      // Column 1: 0.5*(2+5)*1 = 3.5
      // Column 2: 0.5*(3+6)*1 = 4.5
      expect((result as { toArray: () => number[] }).toArray()).toEqual([2.5, 3.5, 4.5]);
    });

    it('throws for array with less than 2 points', () => {
      const y = array([1]);
      expect(() => trapezoid(y)).toThrow();
    });

    it('handles negative axis', () => {
      const y = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result1 = trapezoid(y, undefined, 1, -1);
      const result2 = trapezoid(y, undefined, 1, 1);
      expect((result1 as { toArray: () => number[] }).toArray()).toEqual(
        (result2 as { toArray: () => number[] }).toArray()
      );
    });
  });
});
