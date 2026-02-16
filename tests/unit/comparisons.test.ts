/**
 * Unit tests for comparison operations
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  array_equal,
  array_equiv,
  greater,
  greater_equal,
  less,
  less_equal,
  equal,
  not_equal,
  isclose,
  allclose,
  logical_and,
  logical_or,
  logical_not,
  isfinite,
  isinf,
  isnan,
  Complex,
} from '../../src';

describe('Comparison Operations', () => {
  describe('greater()', () => {
    describe('scalar comparisons', () => {
      it('compares 1D array with scalar', () => {
        const arr = array([1, 2, 3, 4, 5]);
        const result = arr.greater(3);
        expect(result.dtype).toBe('bool');
        expect(result.shape).toEqual([5]);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1]);
      });

      it('compares 2D array with scalar', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.greater(3);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1, 1]);
      });
    });

    describe('array comparisons', () => {
      it('compares two 1D arrays', () => {
        const a = array([1, 2, 3]);
        const b = array([2, 2, 2]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3]);
        expect(Array.from(result.data)).toEqual([0, 0, 1]);
      });

      it('compares two 2D arrays', () => {
        const a = array([
          [1, 5],
          [3, 2],
        ]);
        const b = array([
          [2, 4],
          [3, 1],
        ]);
        const result = a.greater(b);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([0, 1, 0, 1]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts (3, 4) > (4,)', () => {
        const a = array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ]);
        const b = array([2, 5, 8, 11]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3, 4]);
        // Row 0: [1>2, 2>5, 3>8, 4>11] = [0,0,0,0]
        // Row 1: [5>2, 6>5, 7>8, 8>11] = [1,1,0,0]
        // Row 2: [9>2, 10>5, 11>8, 12>11] = [1,1,1,1]
        expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]);
      });

      it('broadcasts (3, 1) > (1, 4)', () => {
        const a = array([[1], [2], [3]]);
        const b = array([[1, 2, 3, 4]]);
        const result = a.greater(b);
        expect(result.shape).toEqual([3, 4]);
      });
    });
  });

  describe('greater_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.greater_equal(3);
      expect(Array.from(result.data)).toEqual([0, 0, 1, 1, 1]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.greater_equal(b);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });
  });

  describe('less()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less(3);
      expect(Array.from(result.data)).toEqual([1, 1, 0, 0, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.less(b);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });

    it('broadcasts arrays', () => {
      const a = array([[1], [2], [3]]);
      const b = array([2]);
      const result = a.less(b);
      expect(result.shape).toEqual([3, 1]);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });
  });

  describe('less_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.less_equal(3);
      expect(Array.from(result.data)).toEqual([1, 1, 1, 0, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([2, 2, 2]);
      const result = a.less_equal(b);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });
  });

  describe('equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.equal(2);
      expect(Array.from(result.data)).toEqual([0, 1, 0, 1, 0]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });

    it('compares 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 0],
        [3, 0],
      ]);
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0]);
    });
  });

  describe('not_equal()', () => {
    it('compares with scalar', () => {
      const arr = array([1, 2, 3, 2, 1]);
      const result = arr.not_equal(2);
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0, 1]);
    });

    it('compares two arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      const result = a.not_equal(b);
      expect(Array.from(result.data)).toEqual([0, 0, 1]);
    });
  });

  describe('result properties', () => {
    it('returns bool dtype for all comparisons', () => {
      const arr = array([1, 2, 3]);
      expect(arr.greater(2).dtype).toBe('bool');
      expect(arr.less(2).dtype).toBe('bool');
      expect(arr.equal(2).dtype).toBe('bool');
      expect(arr.not_equal(2).dtype).toBe('bool');
      expect(arr.greater_equal(2).dtype).toBe('bool');
      expect(arr.less_equal(2).dtype).toBe('bool');
    });

    it('preserves shape in comparisons', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.greater(3);
      expect(result.shape).toEqual(arr.shape);
    });
  });

  describe('edge cases', () => {
    it('handles all false result', () => {
      const arr = array([1, 2, 3]);
      const result = arr.greater(10);
      expect(Array.from(result.data)).toEqual([0, 0, 0]);
    });

    it('handles all true result', () => {
      const arr = array([5, 6, 7]);
      const result = arr.greater(0);
      expect(Array.from(result.data)).toEqual([1, 1, 1]);
    });

    it('compares negative numbers', () => {
      const arr = array([-3, -1, 0, 1, 3]);
      const result = arr.greater(0);
      expect(Array.from(result.data)).toEqual([0, 0, 0, 1, 1]);
    });

    it('handles floating point comparisons', () => {
      const arr = array([1.1, 2.2, 3.3]);
      const result = arr.greater(2.0);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });
  });

  describe('broadcasting errors', () => {
    it('throws on incompatible shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2, 3]);
      expect(() => a.greater(b)).toThrow(/broadcast/);
    });
  });

  describe('chained comparisons', () => {
    it('can chain comparison results', () => {
      const arr = array([1, 2, 3, 4, 5]);
      // Find elements between 2 and 4 (exclusive)
      const gt2 = arr.greater(2);
      const lt4 = arr.less(4);
      // Manually AND the results
      const both = array(Array.from(gt2.data).map((v, i) => (v && lt4.data[i] ? 1 : 0)));
      expect(Array.from(both.data)).toEqual([0, 0, 1, 0, 0]);
    });
  });

  describe('isclose()', () => {
    describe('scalar comparisons', () => {
      it('compares with exact match', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.0);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.data)).toEqual([0, 1, 0]);
      });

      it('compares with small difference within tolerance', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.00001, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([0, 1, 0]);
      });

      it('compares with difference outside tolerance', () => {
        const arr = array([1.0, 2.0, 3.0]);
        const result = arr.isclose(2.1, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([0, 0, 0]);
      });

      it('uses absolute tolerance correctly', () => {
        const arr = array([1e-9, 2e-9]);
        const result = arr.isclose(0, 1e-5, 1e-8);
        // 1e-9 and 2e-9 are both within atol=1e-8 of 0
        expect(Array.from(result.data)).toEqual([1, 1]);
      });

      it('uses relative tolerance correctly', () => {
        const arr = array([1.0, 100.0]);
        const result = arr.isclose(1.001, 0.01, 0);
        // 1.0 vs 1.001: diff=0.001, threshold=0.01*1.001=0.01001 → close
        // 100.0 vs 1.001: diff=98.999, threshold=0.01*1.001=0.01001 → not close
        expect(Array.from(result.data)).toEqual([1, 0]);
      });

      it('handles BigInt arrays with scalar', () => {
        const arr = array([1n, 2n, 100n], 'int64');
        const result = arr.isclose(2, 0.1, 0);
        // 1 vs 2: diff=1, threshold=0.1*2=0.2 → not close
        // 2 vs 2: diff=0, threshold=0.2 → close
        // 100 vs 2: diff=98, threshold=0.2 → not close
        expect(Array.from(result.data)).toEqual([0, 1, 0]);
      });
    });

    describe('array comparisons', () => {
      it('compares two 1D arrays', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.00001, 2.0, 2.99999]);
        const result = a.isclose(b, 1e-4, 1e-8);
        expect(Array.from(result.data)).toEqual([1, 1, 1]);
      });

      it('compares two 2D arrays', () => {
        const a = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        const b = array([
          [1.0, 2.1],
          [3.0, 4.0],
        ]);
        const result = a.isclose(b, 1e-5, 1e-8);
        expect(Array.from(result.data)).toEqual([1, 0, 1, 1]);
      });

      it('handles floating point precision issues', () => {
        const a = array([0.1 + 0.2]); // Typically 0.30000000000000004
        const b = array([0.3]);
        const result = a.isclose(b);
        expect(Array.from(result.data)).toEqual([1]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts (2, 3) with (3,)', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        const result = a.isclose(b);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([1, 1, 1, 0, 0, 0]);
      });

      it('broadcasts (3, 1) with (1, 3)', () => {
        const a = array([[1.0], [2.0], [3.0]]);
        const b = array([[1.0, 2.0, 3.0]]);
        const result = a.isclose(b);
        expect(result.shape).toEqual([3, 3]);
      });
    });
  });

  describe('allclose()', () => {
    describe('scalar comparisons', () => {
      it('returns true when all elements are close', () => {
        const arr = array([1.0, 1.00001, 0.99999]);
        expect(arr.allclose(1.0, 1e-4, 1e-8)).toBe(true);
      });

      it('returns false when any element is not close', () => {
        const arr = array([1.0, 1.1, 1.0]);
        expect(arr.allclose(1.0, 1e-5, 1e-8)).toBe(false);
      });

      it('returns true for exact match', () => {
        const arr = array([2.0, 2.0, 2.0]);
        expect(arr.allclose(2.0)).toBe(true);
      });
    });

    describe('array comparisons', () => {
      it('returns true when arrays are close', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.00001, 2.0, 2.99999]);
        expect(a.allclose(b, 1e-4, 1e-8)).toBe(true);
      });

      it('returns false when arrays differ', () => {
        const a = array([1.0, 2.0, 3.0]);
        const b = array([1.0, 2.1, 3.0]);
        expect(a.allclose(b, 1e-5, 1e-8)).toBe(false);
      });

      it('handles 2D arrays', () => {
        const a = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        const b = array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        expect(a.allclose(b)).toBe(true);
      });

      it('handles floating point precision issues', () => {
        const a = array([0.1 + 0.2]);
        const b = array([0.3]);
        expect(a.allclose(b)).toBe(true);
      });
    });

    describe('broadcasting', () => {
      it('works with broadcast-compatible shapes', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        expect(a.allclose(b)).toBe(true);
      });

      it('returns false when any broadcast element differs', () => {
        const a = array([
          [1.0, 2.0, 3.0],
          [1.0, 2.1, 3.0],
        ]);
        const b = array([1.0, 2.0, 3.0]);
        expect(a.allclose(b, 1e-5, 1e-8)).toBe(false);
      });
    });

    describe('edge cases', () => {
      it('handles empty arrays', () => {
        const a = array([]);
        const b = array([]);
        expect(a.allclose(b)).toBe(true);
      });

      it('handles single element', () => {
        const a = array([1.0]);
        const b = array([1.00001]);
        expect(a.allclose(b, 1e-4, 1e-8)).toBe(true);
      });
    });
  });

  describe('array_equal', () => {
    it('returns true for identical arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      expect(array_equal(a, b)).toBe(true);
    });

    it('returns false for different arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('requires same shapes (no broadcasting)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2]);
      expect(array_equal(a, b)).toBe(false);
    });

    it('works with 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(array_equal(a, b)).toBe(true);
    });
  });

  describe('array_equiv', () => {
    it('returns true for identical arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3]);
      expect(array_equiv(a, b)).toBe(true);
    });

    it('returns false for different arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 4]);
      expect(array_equiv(a, b)).toBe(false);
    });

    it('broadcasts arrays before comparing', () => {
      const a = array([
        [1, 2],
        [1, 2],
      ]);
      const b = array([1, 2]);
      expect(array_equiv(a, b)).toBe(true);
    });

    it('broadcasts with size-1 dimensions', () => {
      const a = array([[1], [2], [3]]);
      const b = array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
      ]);
      expect(array_equiv(a, b)).toBe(true);
    });

    it('broadcasts 2D with 1D', () => {
      const a = array([
        [5, 5, 5],
        [5, 5, 5],
      ]);
      const b = array([5, 5, 5]);
      expect(array_equiv(a, b)).toBe(true);
    });

    it('returns false when broadcast values differ', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2]);
      expect(array_equiv(a, b)).toBe(false);
    });

    it('returns false for incompatible broadcast shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3, 4]);
      expect(array_equiv(a, b)).toBe(false);
    });

    it('works with scalars (0D)', () => {
      const a = array([5, 5, 5]);
      const b = array(5);
      expect(array_equiv(a, b)).toBe(true);
    });

    it('broadcasts complex multi-dimensional arrays', () => {
      const a = array([[[1, 2]], [[3, 4]]]);
      const b = array([1, 2]);
      expect(array_equiv(a, b)).toBe(false); // Different values

      // c has shape (2, 1, 2), d has shape (2)
      // They broadcast to (2, 1, 2) and all values are 5, so they ARE equivalent
      const c = array([[[5, 5]], [[5, 5]]]);
      const d = array([5, 5]);
      expect(array_equiv(c, d)).toBe(true);
    });

    it('handles different dtypes correctly', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([1.0, 2.0, 3.0], 'float64');
      expect(array_equiv(a, b)).toBe(true);
    });

    it('returns true for same-shape arrays (no actual broadcasting)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(array_equiv(a, b)).toBe(true);
    });
  });

  describe('Standalone comparison functions (branch coverage)', () => {
    it('greater int32', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([3, 2, 1], 'int32');
      expect(greater(a, b).size).toBe(3);
    });

    it('less float32', () => {
      const a = array([1, 2, 3], 'float32');
      expect(less(a, 2).size).toBe(3);
    });

    it('equal int32', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([1, 0, 3], 'int32');
      expect(equal(a, b).size).toBe(3);
    });

    it('not_equal', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 0, 3]);
      expect(not_equal(a, b).size).toBe(3);
    });

    it('isclose', () => {
      const a = array([1.0, 2.0001]);
      const b = array([1.0, 2.0]);
      expect(isclose(a, b).size).toBe(2);
    });

    it('logical operations on bool', () => {
      const a = array([1, 0, 1], 'bool');
      const b = array([1, 1, 0], 'bool');
      expect(logical_and(a, b).size).toBe(3);
      expect(logical_or(a, b).size).toBe(3);
      expect(logical_not(a).size).toBe(3);
    });

    it('isfinite/isinf/isnan', () => {
      const a = array([1, Infinity, NaN, -Infinity]);
      expect(isfinite(a).size).toBe(4);
      expect(isinf(a).size).toBe(4);
      expect(isnan(a).size).toBe(4);
    });
  });

  describe('Complex number comparisons', () => {
    it('greater with complex arrays (lexicographic ordering)', () => {
      const a = array([new Complex(2, 3), new Complex(1, 5), new Complex(2, 1)]);
      const b = array([new Complex(2, 1), new Complex(1, 3), new Complex(2, 4)]);
      const result = a.greater(b);

      // (2, 3) > (2, 1): real equal, 3 > 1 => true
      // (1, 5) > (1, 3): real equal, 5 > 3 => true
      // (2, 1) > (2, 4): real equal, 1 > 4 => false
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(1);
      expect(result.get([2])).toBe(0);
    });

    it('greater_equal with complex arrays', () => {
      const a = array([new Complex(2, 3), new Complex(1, 5)]);
      const b = array([new Complex(2, 3), new Complex(1, 6)]);
      const result = a.greater_equal(b);

      // (2, 3) >= (2, 3): equal => true
      // (1, 5) >= (1, 6): real equal, 5 >= 6 => false
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(0);
    });

    it('less with complex arrays', () => {
      const a = array([new Complex(1, 2), new Complex(2, 1)]);
      const b = array([new Complex(2, 1), new Complex(1, 2)]);
      const result = a.less(b);

      // (1, 2) < (2, 1): 1 < 2 => true
      // (2, 1) < (1, 2): 2 < 1 => false
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(0);
    });

    it('less_equal with complex arrays', () => {
      const a = array([new Complex(1, 2), new Complex(1, 2)]);
      const b = array([new Complex(1, 3), new Complex(1, 2)]);
      const result = a.less_equal(b);

      // (1, 2) <= (1, 3): real equal, 2 <= 3 => true
      // (1, 2) <= (1, 2): equal => true
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(1);
    });

    it('equal with complex arrays', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(1, 2), new Complex(3, 5)]);
      const result = a.equal(b);

      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(0);
    });

    it('not_equal with complex arrays', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(1, 2), new Complex(3, 5)]);
      const result = a.not_equal(b);

      expect(result.get([0])).toBe(0);
      expect(result.get([1])).toBe(1);
    });

    it('compares complex with real (real values get 0 imaginary part)', () => {
      const a = array([new Complex(2, 1), new Complex(2, 0)]);
      const b = array([2, 2]);
      const result = a.greater(b);

      // (2, 1) > (2, 0): real equal, 1 > 0 => true
      // (2, 0) > (2, 0): equal => false
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(0);
    });

    it('equal with complex arrays', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(1, 2)]);
      const b = array([new Complex(1, 2), new Complex(3, 4), new Complex(0, 0)]);
      const result = a.equal(b);
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(1);
      expect(result.get([2])).toBe(0);
    });

    it('less_equal with complex numbers (lexicographic)', () => {
      const a = array([new Complex(1, 2), new Complex(2, 1)]);
      const b = array([new Complex(1, 3), new Complex(2, 1)]);
      const result = a.less_equal(b);
      // (1,2) <= (1,3): real equal, 2 <= 3 => true
      // (2,1) <= (2,1): equal => true
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(1);
    });
  });

  describe('Non-contiguous array comparisons', () => {
    it('not_equal with non-contiguous array and scalar', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const aT = a.T; // Non-contiguous
      const result = aT.not_equal(2);
      expect(result.shape).toEqual([2, 2]);
      expect(result.dtype).toBe('bool');
    });

    it('isclose with non-contiguous arrays', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = array([
        [1.0001, 2.0001],
        [3.0001, 4.0001],
      ]);
      const result = isclose(a.T, b.T, 1e-3, 1e-3);
      expect(result.shape).toEqual([2, 2]);
    });

    it('isclose with non-contiguous array and scalar', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const aT = a.T; // Non-contiguous
      const result = isclose(aT, 2.0, 1e-5, 1e-5);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('BigInt comparisons', () => {
    it('not_equal with bigint array and scalar', () => {
      const a = array([1n, 2n, 3n, 2n], 'int64');
      const result = not_equal(a, 2);
      expect(result.shape).toEqual([4]);
      expect(result.dtype).toBe('bool');
    });

    it('greater with uint64 arrays', () => {
      const a = array([10n, 20n, 30n], 'uint64');
      const b = array([15n, 20n, 25n], 'uint64');
      const result = a.greater(b);
      expect(Array.from(result.data)).toEqual([0, 0, 1]);
    });

    it('less with mixed bigint and regular', () => {
      const a = array([5n, 10n], 'int64');
      const result = a.less(8);
      expect(Array.from(result.data)).toEqual([1, 0]);
    });
  });

  describe('Mixed precision comparisons', () => {
    it('comparison with float32 arrays', () => {
      const a = array([1.0, 2.0, 3.0], 'float32');
      const b = array([1.5, 2.0, 2.5], 'float32');
      const result = a.less(b);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });

    it('isclose with mixed float32 and float64', () => {
      const a = array([1.0, 2.0], 'float32');
      const b = array([1.0001, 2.0001], 'float64');
      const result = isclose(a, b, 1e-3, 1e-3);
      expect(result.shape).toEqual([2]);
    });

    it('operations with uint8', () => {
      const a = array([0, 128, 255], 'uint8');
      const result = a.greater(100);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });

    it('operations with int16', () => {
      const a = array([-100, 0, 100], 'int16');
      const b = array([0, 0, 0], 'int16');
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });

    it('operations with uint32', () => {
      const a = array([1, 2, 3], 'uint32');
      const result = isclose(a, 2, 1, 0);
      expect(result.shape).toEqual([3]);
    });

    it('equal with uint16 arrays', () => {
      const a = array([1, 2, 3], 'uint16');
      const b = array([1, 0, 3], 'uint16');
      const result = a.equal(b);
      expect(Array.from(result.data)).toEqual([1, 0, 1]);
    });

    it('not_equal with int8 arrays', () => {
      const a = array([1, 2, 3], 'int8');
      const b = array([1, 0, 3], 'int8');
      const result = a.not_equal(b);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });

    it('greater with mixed int8 and int32', () => {
      const a = array([1, 2, 3], 'int8');
      const b = array([0, 2, 4], 'int32');
      const result = a.greater(b);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });

    it('less with uint8 and uint64', () => {
      const a = array([1, 2, 3], 'uint8');
      const b = array([2n, 2n, 2n], 'uint64');
      const result = a.less(b);
      expect(result.shape).toEqual([3]);
    });

    it('greater_equal with float32 scalar', () => {
      const a = array([1.0, 2.0, 3.0], 'float32');
      const result = a.greater_equal(2.0);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });

    it('less_equal with int32 scalar', () => {
      const a = array([1, 2, 3], 'int32');
      const result = a.less_equal(2);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });

    it('equal with bigint and bigint', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const b = array([1n, 0n, 3n], 'int64');
      const result = equal(a, b);
      expect(Array.from(result.data)).toEqual([1, 0, 1]);
    });

    it('not_equal with uint64 arrays', () => {
      const a = array([1n, 2n, 3n], 'uint64');
      const b = array([1n, 0n, 3n], 'uint64');
      const result = not_equal(a, b);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });

    it('greater with int64 scalar', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const result = greater(a, 2);
      expect(result.shape).toEqual([3]);
    });

    it('less with uint64 scalar', () => {
      const a = array([1n, 2n, 3n], 'uint64');
      const result = less(a, 2);
      expect(result.shape).toEqual([3]);
    });

    it('greater_equal with bigint arrays', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const b = array([2n, 2n, 2n], 'int64');
      const result = greater_equal(a, b);
      expect(Array.from(result.data)).toEqual([0, 1, 1]);
    });

    it('less_equal with bigint arrays', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const b = array([2n, 2n, 2n], 'int64');
      const result = less_equal(a, b);
      expect(Array.from(result.data)).toEqual([1, 1, 0]);
    });

    it('equal with uint32 and uint16', () => {
      const a = array([1, 2, 3], 'uint32');
      const b = array([1, 0, 3], 'uint16');
      const result = equal(a, b);
      expect(result.shape).toEqual([3]);
    });

    it('greater with int8 and int16', () => {
      const a = array([1, 2, 3], 'int8');
      const b = array([0, 2, 4], 'int16');
      const result = greater(a, b);
      expect(result.shape).toEqual([3]);
    });

    it('less with mixed unsigned types', () => {
      const a = array([1, 2, 3], 'uint8');
      const b = array([2, 2, 2], 'uint32');
      const result = less(a, b);
      expect(result.shape).toEqual([3]);
    });

    it('isclose with int32 arrays', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([1, 2, 4], 'int32');
      const result = isclose(a, b, 1e-5, 1);
      expect(result.shape).toEqual([3]);
    });

    it('allclose with uint8 arrays', () => {
      const a = array([1, 2, 3], 'uint8');
      const b = array([1, 2, 3], 'uint8');
      const result = allclose(a, b);
      expect(result).toBe(true);
    });

    it('allclose with int8 arrays (not close)', () => {
      const a = array([1, 2, 3], 'int8');
      const b = array([1, 2, 4], 'int8');
      const result = allclose(a, b, 1e-5, 0);
      expect(result).toBe(false);
    });

    it('isclose with uint64 arrays', () => {
      const a = array([1n, 2n, 3n], 'uint64');
      const b = array([1n, 2n, 4n], 'uint64');
      const result = isclose(a, b);
      expect(result.shape).toEqual([3]);
    });

    it('equal with scalar and int8 array', () => {
      const a = array([1, 2, 3], 'int8');
      const result = equal(a, 2);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });

    it('not_equal with scalar and uint32 array', () => {
      const a = array([1, 2, 3], 'uint32');
      const result = not_equal(a, 2);
      expect(Array.from(result.data)).toEqual([1, 0, 1]);
    });

    it('greater with scalar and int16 array', () => {
      const a = array([1, 2, 3], 'int16');
      const result = greater(a, 2);
      expect(Array.from(result.data)).toEqual([0, 0, 1]);
    });

    it('less with scalar and uint16 array', () => {
      const a = array([1, 2, 3], 'uint16');
      const result = less(a, 2);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });

    it('greater_equal with complex arrays (different real parts)', () => {
      const a = array([new Complex(2, 1), new Complex(1, 1)]);
      const b = array([new Complex(1, 2), new Complex(1, 0)]);
      const result = greater_equal(a, b);
      // When real parts differ, comparison is based on real part only
      expect(result.shape).toEqual([2]);
    });

    it('less_equal with complex arrays (different real parts)', () => {
      const a = array([new Complex(1, 2), new Complex(2, 1)]);
      const b = array([new Complex(2, 1), new Complex(1, 2)]);
      const result = less_equal(a, b);
      // When real parts differ, comparison is based on real part only
      expect(result.shape).toEqual([2]);
    });

    it('greater with non-contiguous array and scalar', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = greater(a.T, 2);
      expect(result.shape).toEqual([2, 2]);
    });

    it('greater with non-contiguous complex array and scalar', () => {
      const a = array([
        [new Complex(1, 1), new Complex(2, 2)],
        [new Complex(3, 3), new Complex(4, 4)],
      ]);
      const result = greater(a.T, 2);
      expect(result.shape).toEqual([2, 2]);
    });
  });
});
