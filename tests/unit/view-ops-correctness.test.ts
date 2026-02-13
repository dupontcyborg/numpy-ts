/**
 * Tests that operations produce correct results when given views/slices
 * (non-zero offset, non-contiguous strides).
 *
 * These tests catch bugs where code accesses `storage.data[i]`
 * instead of using `storage.iget(i)` or `data[offset + i]`.
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  add,
  abs,
  dot,
  argsort,
  sort,
  count_nonzero,
  cumsum,
  cumprod,
  variance,
} from '../../src';

describe('Operations on views/slices', () => {
  // ========================================
  // Arithmetic operations
  // ========================================
  describe('arithmetic with sliced arrays', () => {
    it('add: contiguous slice (offset > 0)', () => {
      const a = array([10, 20, 30, 40, 50]);
      const sliced = a.slice('2:5'); // [30, 40, 50]
      const result = sliced.add(1);
      expect(result.toArray()).toEqual([31, 41, 51]);
    });

    it('add: step-sliced (non-contiguous)', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      const sliced = a.slice('::2'); // [1, 3, 5]
      const result = sliced.add(10);
      expect(result.toArray()).toEqual([11, 13, 15]);
    });

    it('add two sliced arrays', () => {
      const a = array([10, 20, 30, 40]);
      const b = array([1, 2, 3, 4]);
      const sa = a.slice('1:3'); // [20, 30]
      const sb = b.slice('0:2'); // [1, 2]
      const result = sa.add(sb);
      expect(result.toArray()).toEqual([21, 32]);
    });

    it('subtract: sliced array', () => {
      const a = array([10, 20, 30, 40, 50]);
      const sliced = a.slice('1:4'); // [20, 30, 40]
      const result = sliced.subtract(5);
      expect(result.toArray()).toEqual([15, 25, 35]);
    });

    it('multiply: step-sliced array', () => {
      const a = array([1, 2, 3, 4, 5]);
      const sliced = a.slice('::2'); // [1, 3, 5]
      const result = sliced.multiply(2);
      expect(result.toArray()).toEqual([2, 6, 10]);
    });

    it('standalone add with sliced arrays', () => {
      const a = array([10, 20, 30, 40, 50]);
      const sa = a.slice('1:4'); // [20, 30, 40]
      const result = add(sa, 100);
      expect(result.toArray()).toEqual([120, 130, 140]);
    });
  });

  // ========================================
  // Reductions
  // ========================================
  describe('reductions on sliced arrays', () => {
    it('sum: contiguous slice', () => {
      const a = array([10, 20, 30, 40, 50]);
      const sliced = a.slice('1:4'); // [20, 30, 40]
      expect(sliced.sum()).toBe(90);
    });

    it('sum: step-sliced (non-contiguous)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const sliced = a.slice('::2'); // [1, 3, 5]
      expect(sliced.sum()).toBe(9);
    });

    it('mean: sliced', () => {
      const a = array([10, 20, 30, 40, 50]);
      const sliced = a.slice('1:4'); // [20, 30, 40]
      expect(sliced.mean()).toBe(30);
    });

    it('max/min: sliced', () => {
      const a = array([5, 1, 4, 2, 3]);
      const sliced = a.slice('1:4'); // [1, 4, 2]
      expect(sliced.max()).toBe(4);
      expect(sliced.min()).toBe(1);
    });
  });

  // ========================================
  // Math operations
  // ========================================
  describe('math ops on sliced arrays', () => {
    it('sqrt: sliced', () => {
      const a = array([1, 4, 9, 16, 25]);
      const sliced = a.slice('1:4'); // [4, 9, 16]
      const result = sliced.sqrt();
      expect(result.toArray()).toEqual([2, 3, 4]);
    });

    it('negative: sliced', () => {
      const a = array([1, -2, 3, -4, 5]);
      const sliced = a.slice('1:4'); // [-2, 3, -4]
      const result = sliced.negative();
      expect(result.toArray()).toEqual([2, -3, 4]);
    });

    it('abs: step-sliced (via standalone absolute)', () => {
      const a = array([-5, 2, -3, 4, -1]);
      const sliced = a.slice('::2'); // [-5, -3, -1]
      // Use standalone abs() which works with any NDArray/NDArrayCore
      const result = abs(sliced);
      expect(result.toArray()).toEqual([5, 3, 1]);
    });
  });

  // ========================================
  // 2D sliced operations
  // ========================================
  describe('2D sliced operations', () => {
    it('add scalar to 2D sub-slice', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const sliced = a.slice('1:3', '1:3'); // [[5, 6], [8, 9]]
      const result = sliced.add(10);
      expect(result.toArray()).toEqual([
        [15, 16],
        [18, 19],
      ]);
    });

    it('sum 2D sub-slice', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const sliced = a.slice('1:3', '1:3'); // [[5, 6], [8, 9]]
      expect(sliced.sum()).toBe(28);
    });
  });

  // ========================================
  // Axis-based reductions on views
  // ========================================
  describe('axis-based reductions on views', () => {
    it('sum along axis=0 on 2D step-sliced view', () => {
      // 3x4 array, take every-other row → 2x4 non-contiguous view
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const sliced = a.slice('::2'); // rows 0,2 → [[1,2,3,4],[9,10,11,12]]
      const result = sliced.sum(0); // [10, 12, 14, 16]
      expect(result.toArray()).toEqual([10, 12, 14, 16]);
    });

    it('sum along axis=1 on 2D offset slice', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const sliced = a.slice('1:'); // [[4,5,6],[7,8,9]]
      const result = sliced.sum(1); // [15, 24]
      expect(result.toArray()).toEqual([15, 24]);
    });

    it('prod along axis on non-contiguous view', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const sliced = a.slice('::2'); // [[1,2,3],[7,8,9]]
      const result = sliced.prod(0); // [7, 16, 27]
      expect(result.toArray()).toEqual([7, 16, 27]);
    });

    it('max/min along axis on non-contiguous view', () => {
      const a = array([
        [10, 20, 30],
        [5, 25, 15],
        [40, 10, 35],
      ]);
      const sliced = a.slice('::2'); // [[10,20,30],[40,10,35]]
      expect(sliced.max(0).toArray()).toEqual([40, 20, 35]);
      expect(sliced.min(0).toArray()).toEqual([10, 10, 30]);
    });

    it('argmin/argmax along axis on non-contiguous view', () => {
      const a = array([
        [10, 20, 30],
        [5, 25, 15],
        [40, 10, 35],
      ]);
      const sliced = a.slice('::2'); // [[10,20,30],[40,10,35]]
      expect(sliced.argmax(0).toArray()).toEqual([1, 0, 1]);
      expect(sliced.argmin(0).toArray()).toEqual([0, 1, 0]);
    });

    it('variance along axis on non-contiguous view', () => {
      const a = array([
        [2, 4],
        [6, 8],
        [10, 12],
      ]);
      const sliced = a.slice('::2'); // [[2,4],[10,12]]
      // mean along axis 0: [6, 8]
      // var along axis 0: [16, 16]
      const result = variance(sliced, 0);
      expect(result.toArray()).toEqual([16, 16]);
    });
  });

  // ========================================
  // all/any on views
  // ========================================
  describe('all/any on views', () => {
    it('all: flat non-contiguous', () => {
      const a = array([1, 0, 1, 0, 1]);
      const sliced = a.slice('::2'); // [1, 1, 1]
      expect(sliced.all()).toBe(true);
    });

    it('all: flat with zero in view', () => {
      const a = array([1, 1, 0, 1, 1]);
      const sliced = a.slice('1:'); // [1, 0, 1, 1]
      expect(sliced.all()).toBe(false);
    });

    it('any: flat non-contiguous', () => {
      const a = array([0, 1, 0, 1, 0]);
      const sliced = a.slice('::2'); // [0, 0, 0]
      expect(sliced.any()).toBe(false);
    });

    it('any: flat offset', () => {
      const a = array([0, 0, 0, 1]);
      const sliced = a.slice('1:'); // [0, 0, 1]
      expect(sliced.any()).toBe(true);
    });

    it('all along axis on non-contiguous 2D view', () => {
      const a = array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
      ]);
      const sliced = a.slice('::2'); // [[1,1,1],[1,1,1]]
      expect(sliced.all(0).toArray()).toEqual([1, 1, 1]);
    });
  });

  // ========================================
  // Cumulative ops on views
  // ========================================
  describe('cumsum/cumprod on views', () => {
    it('cumsum flat on non-contiguous view', () => {
      const a = array([1, 2, 3, 4, 5]);
      const sliced = a.slice('::2'); // [1, 3, 5]
      const result = cumsum(sliced);
      expect(result.toArray()).toEqual([1, 4, 9]);
    });

    it('cumprod flat on non-contiguous view', () => {
      const a = array([1, 2, 3, 4, 5]);
      const sliced = a.slice('::2'); // [1, 3, 5]
      const result = cumprod(sliced);
      expect(result.toArray()).toEqual([1, 3, 15]);
    });

    it('cumsum flat on offset view', () => {
      const a = array([10, 20, 30, 40]);
      const sliced = a.slice('1:'); // [20, 30, 40]
      const result = cumsum(sliced);
      expect(result.toArray()).toEqual([20, 50, 90]);
    });

    it('cumsum along axis on non-contiguous 2D view', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const sliced = a.slice('::2'); // [[1,2],[5,6]]
      const result = cumsum(sliced, 0);
      expect(result.toArray()).toEqual([
        [1, 2],
        [6, 8],
      ]);
    });
  });

  // ========================================
  // Sorting on views
  // ========================================
  describe('sorting on views', () => {
    it('sort on non-contiguous 1D view', () => {
      const a = array([50, 10, 40, 20, 30]);
      const sliced = a.slice('::2'); // [50, 40, 30]
      const result = sort(sliced);
      expect(result.toArray()).toEqual([30, 40, 50]);
    });

    it('argsort on non-contiguous 1D view', () => {
      const a = array([50, 10, 40, 20, 30]);
      const sliced = a.slice('::2'); // [50, 40, 30]
      const result = argsort(sliced);
      expect(result.toArray()).toEqual([2, 1, 0]);
    });

    it('argsort along axis on 2D offset view', () => {
      const a = array([
        [3, 1, 2],
        [6, 4, 5],
        [9, 7, 8],
      ]);
      const sliced = a.slice('1:'); // [[6,4,5],[9,7,8]]
      const result = argsort(sliced); // sort along last axis
      expect(result.toArray()).toEqual([
        [1, 2, 0],
        [1, 2, 0],
      ]);
    });

    it('count_nonzero flat on non-contiguous view', () => {
      const a = array([1, 0, 2, 0, 3]);
      const sliced = a.slice('::2'); // [1, 2, 3]
      expect(count_nonzero(sliced)).toBe(3);
    });

    it('count_nonzero flat on offset view with zeros', () => {
      const a = array([5, 0, 0, 3]);
      const sliced = a.slice('1:'); // [0, 0, 3]
      expect(count_nonzero(sliced)).toBe(1);
    });

    it('count_nonzero along axis on 2D view', () => {
      const a = array([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 4],
      ]);
      const sliced = a.slice('::2'); // [[1,0,2],[3,0,4]]
      const result = count_nonzero(sliced, 0);
      expect(result.toArray()).toEqual([2, 0, 2]);
    });
  });

  // ========================================
  // dot product on views
  // ========================================
  describe('dot on views', () => {
    it('1D dot on non-contiguous view', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([10, 20, 30, 40, 50]);
      const sa = a.slice('::2'); // [1, 3, 5]
      const sb = b.slice('::2'); // [10, 30, 50]
      // 1*10 + 3*30 + 5*50 = 10 + 90 + 250 = 350
      expect(dot(sa, sb)).toBe(350);
    });

    it('2D @ 1D dot on offset view', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const b = array([10, 20, 30, 40]);
      const sa = a.slice('1:'); // [[4,5,6],[7,8,9]]
      const sb = b.slice('1:'); // [20, 30, 40]
      // [4*20+5*30+6*40, 7*20+8*30+9*40] = [80+150+240, 140+240+360] = [470, 740]
      const result = dot(sa, sb);
      expect(result.toArray()).toEqual([470, 740]);
    });
  });

  // ========================================
  // Chained operations on views
  // ========================================
  describe('chained operations on views', () => {
    it('slice → add → sum', () => {
      const a = array([10, 20, 30, 40, 50]);
      const result = a.slice('1:4').add(1).sum();
      expect(result).toBe(93); // (20+1) + (30+1) + (40+1)
    });

    it('slice → multiply → toArray', () => {
      const a = array([2, 4, 6, 8, 10]);
      const result = a.slice('::2').multiply(3).toArray();
      expect(result).toEqual([6, 18, 30]); // [2,6,10] * 3
    });
  });
});
