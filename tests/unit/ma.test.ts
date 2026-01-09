/**
 * Unit tests for Masked Array module (np.ma)
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src/core/ndarray';
import { MaskedArray, ma, NDArray } from '../../src/index';

// Import nomask from the ma module (masked used for documentation only)
const { nomask } = ma;

describe('Masked Arrays (ma)', () => {
  // ==========================================================================
  // MaskedArray Class
  // ==========================================================================

  describe('MaskedArray Class', () => {
    describe('constructor', () => {
      it('creates MaskedArray from array without mask', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5]);
        expect(arr.data.tolist()).toEqual([1, 2, 3, 4, 5]);
        expect(arr.mask).toBe(false);
      });

      it('creates MaskedArray with boolean mask', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5], {
          mask: [false, true, false, true, false],
        });
        expect(arr.data.tolist()).toEqual([1, 2, 3, 4, 5]);
        expect(arr.mask).not.toBe(false);
        expect((arr.mask as any).tolist()).toEqual([0, 1, 0, 1, 0]);
      });

      it('creates MaskedArray from NDArray', () => {
        const ndarr = array([1, 2, 3]);
        const arr = new MaskedArray(ndarr);
        expect(arr.data.tolist()).toEqual([1, 2, 3]);
      });

      it('handles explicit mask with masked constant', () => {
        // Create a masked array where the mask is explicitly provided
        const arr = new MaskedArray([1, 0, 3], { mask: [false, true, false] });
        expect(arr.mask).not.toBe(false);
        // The masked constant can be used when filling
        expect(arr.filled(-999).tolist()).toEqual([1, -999, 3]);
      });

      it('sets fill_value correctly', () => {
        const arr = new MaskedArray([1, 2, 3], { fill_value: -999 });
        expect(arr.fill_value).toBe(-999);
      });
    });

    describe('properties', () => {
      it('returns correct shape', () => {
        const arr = new MaskedArray([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.shape).toEqual([2, 3]);
      });

      it('returns correct ndim', () => {
        const arr = new MaskedArray([
          [1, 2],
          [3, 4],
        ]);
        expect(arr.ndim).toBe(2);
      });

      it('returns correct size', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5]);
        expect(arr.size).toBe(5);
      });

      it('returns correct dtype', () => {
        const arr = new MaskedArray([1.5, 2.5, 3.5]);
        expect(arr.dtype).toBe('float64');
      });
    });

    describe('filled()', () => {
      it('returns data with masked values replaced', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5], {
          mask: [false, true, false, true, false],
          fill_value: 0,
        });
        const filled = arr.filled();
        expect(filled.tolist()).toEqual([1, 0, 3, 0, 5]);
      });

      it('uses custom fill value', () => {
        const arr = new MaskedArray([1, 2, 3], {
          mask: [false, true, false],
          fill_value: 0,
        });
        const filled = arr.filled(-999);
        expect(filled.tolist()).toEqual([1, -999, 3]);
      });
    });

    describe('compressed()', () => {
      it('returns non-masked values only', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5], {
          mask: [false, true, false, true, false],
        });
        const compressed = arr.compressed();
        expect(compressed.tolist()).toEqual([1, 3, 5]);
      });

      it('returns all values if no mask', () => {
        const arr = new MaskedArray([1, 2, 3]);
        const compressed = arr.compressed();
        expect(compressed.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('count()', () => {
      it('returns count of non-masked elements', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5], {
          mask: [false, true, false, true, false],
        });
        expect(arr.count()).toBe(3);
      });

      it('returns size if no mask', () => {
        const arr = new MaskedArray([1, 2, 3, 4, 5]);
        expect(arr.count()).toBe(5);
      });
    });
  });

  // ==========================================================================
  // Array Creation Functions
  // ==========================================================================

  describe('Array Creation', () => {
    describe('ma.array', () => {
      it('creates MaskedArray from array', () => {
        const arr = ma.array([1, 2, 3]);
        expect(arr).toBeInstanceOf(MaskedArray);
        expect(arr.data.tolist()).toEqual([1, 2, 3]);
      });

      it('creates MaskedArray with mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        expect(arr.mask).not.toBe(false);
      });
    });

    describe('ma.zeros', () => {
      it('creates zeros MaskedArray', () => {
        const arr = ma.zeros([3, 3]);
        expect(arr).toBeInstanceOf(MaskedArray);
        expect(arr.shape).toEqual([3, 3]);
        expect(arr.data.sum()).toBe(0);
      });
    });

    describe('ma.ones', () => {
      it('creates ones MaskedArray', () => {
        const arr = ma.ones([2, 3]);
        expect(arr).toBeInstanceOf(MaskedArray);
        expect(arr.data.sum()).toBe(6);
      });
    });

    describe('ma.empty', () => {
      it('creates empty MaskedArray', () => {
        const arr = ma.empty([2, 2]);
        expect(arr).toBeInstanceOf(MaskedArray);
        expect(arr.shape).toEqual([2, 2]);
      });
    });

    describe('ma.arange', () => {
      it('creates arange MaskedArray', () => {
        const arr = ma.arange(0, 5);
        expect(arr.data.tolist()).toEqual([0, 1, 2, 3, 4]);
      });
    });

    describe('ma.masked_all', () => {
      it('creates fully masked array', () => {
        const arr = ma.masked_all([3]);
        expect(arr.mask).not.toBe(false);
        expect(arr.count()).toBe(0);
      });
    });

    describe('ma.masked_all_like', () => {
      it('creates fully masked array like another', () => {
        const template = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const arr = ma.masked_all_like(template);
        expect(arr.shape).toEqual([2, 2]);
        expect(arr.count()).toBe(0);
      });
    });
  });

  // ==========================================================================
  // Masking Functions
  // ==========================================================================

  describe('Masking Functions', () => {
    describe('ma.masked_where', () => {
      it('masks where condition is true', () => {
        const x = array([1, 2, 3, 4, 5]);
        const arr = ma.masked_where(x.greater(3), x);
        expect(arr.compressed().tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.masked_equal', () => {
      it('masks where value equals', () => {
        const arr = ma.masked_equal([1, 2, -999, 4, -999], -999);
        expect(arr.compressed().tolist()).toEqual([1, 2, 4]);
      });
    });

    describe('ma.masked_not_equal', () => {
      it('masks where value not equals', () => {
        const arr = ma.masked_not_equal([1, 2, 3, 2, 1], 2);
        expect(arr.compressed().tolist()).toEqual([2, 2]);
      });
    });

    describe('ma.masked_less', () => {
      it('masks where less than value', () => {
        const arr = ma.masked_less([1, 2, 3, 4, 5], 3);
        expect(arr.compressed().tolist()).toEqual([3, 4, 5]);
      });
    });

    describe('ma.masked_greater', () => {
      it('masks where greater than value', () => {
        const arr = ma.masked_greater([1, 2, 3, 4, 5], 3);
        expect(arr.compressed().tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.masked_inside', () => {
      it('masks values inside range', () => {
        const arr = ma.masked_inside([1, 2, 3, 4, 5], 2, 4);
        expect(arr.compressed().tolist()).toEqual([1, 5]);
      });
    });

    describe('ma.masked_outside', () => {
      it('masks values outside range', () => {
        const arr = ma.masked_outside([1, 2, 3, 4, 5], 2, 4);
        expect(arr.compressed().tolist()).toEqual([2, 3, 4]);
      });
    });

    describe('ma.masked_invalid', () => {
      it('masks NaN and Inf', () => {
        const arr = ma.masked_invalid([1, NaN, 3, Infinity, 5]);
        expect(arr.count()).toBe(3);
      });
    });

    describe('ma.masked_values', () => {
      it('masks specific values with tolerance', () => {
        const arr = ma.masked_values([1.0, 1.0001, 2.0, 3.0], 1.0, 0.001);
        expect(arr.count()).toBe(2); // Only 2.0 and 3.0 unmasked
      });
    });
  });

  // ==========================================================================
  // Mask Utilities
  // ==========================================================================

  describe('Mask Utilities', () => {
    describe('ma.getmask', () => {
      it('returns mask from MaskedArray', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const mask = ma.getmask(arr);
        expect(mask).not.toBe(false);
      });

      it('returns nomask for unmasked array', () => {
        const arr = ma.array([1, 2, 3]);
        const mask = ma.getmask(arr);
        expect(mask).toBe(nomask);
      });
    });

    describe('ma.getmaskarray', () => {
      it('always returns array mask', () => {
        const arr = ma.array([1, 2, 3]);
        const mask = ma.getmaskarray(arr);
        expect(mask.shape).toEqual([3]);
      });
    });

    describe('ma.is_masked', () => {
      it('returns true if any masked', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        expect(ma.is_masked(arr)).toBe(true);
      });

      it('returns false if none masked', () => {
        const arr = ma.array([1, 2, 3]);
        expect(ma.is_masked(arr)).toBe(false);
      });
    });

    describe('ma.make_mask', () => {
      it('creates mask from array', () => {
        const mask = ma.make_mask([0, 1, 0, 1]);
        expect(mask.tolist()).toEqual([0, 1, 0, 1]);
      });
    });
  });

  // ==========================================================================
  // Arithmetic Operations
  // ==========================================================================

  describe('Arithmetic Operations', () => {
    describe('add', () => {
      it('adds with mask propagation', () => {
        const a = ma.array([1, 2, 3], { mask: [false, true, false] });
        const b = ma.array([10, 20, 30]);
        const result = a.add(b);
        expect(result.filled(0).tolist()).toEqual([11, 0, 33]);
      });
    });

    describe('subtract', () => {
      it('subtracts with mask propagation', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const result = a.subtract(5);
        expect(result.filled(0).tolist()).toEqual([5, 0, 25]);
      });
    });

    describe('multiply', () => {
      it('multiplies with mask propagation', () => {
        const a = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = a.multiply(2);
        expect(result.filled(0).tolist()).toEqual([2, 0, 6]);
      });
    });

    describe('divide', () => {
      it('divides with mask propagation', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const result = a.divide(10);
        expect(result.filled(0).tolist()).toEqual([1, 0, 3]);
      });
    });

    describe('power', () => {
      it('raises to power with mask propagation', () => {
        const a = ma.array([2, 3, 4], { mask: [false, true, false] });
        const result = a.power(2);
        expect(result.filled(0).tolist()).toEqual([4, 0, 16]);
      });
    });
  });

  // ==========================================================================
  // Reduction Operations
  // ==========================================================================

  describe('Reduction Operations', () => {
    describe('sum', () => {
      it('sums non-masked values', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, true, false] });
        expect(arr.sum()).toBe(9); // 1 + 3 + 5
      });
    });

    describe('mean', () => {
      it('computes mean of non-masked values', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, true, false] });
        expect(arr.mean()).toBe(3); // (1 + 3 + 5) / 3
      });
    });

    describe('min', () => {
      it('finds min of non-masked values', () => {
        const arr = ma.array([5, 1, 3, 2, 4], { mask: [false, true, false, false, false] });
        expect(arr.min()).toBe(2);
      });
    });

    describe('max', () => {
      it('finds max of non-masked values', () => {
        const arr = ma.array([1, 5, 3, 2, 4], { mask: [false, true, false, false, false] });
        expect(arr.max()).toBe(4);
      });
    });

    describe('std', () => {
      it('computes std of non-masked values', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, true, false] });
        const std = arr.std();
        expect(Math.abs(std - 1.6329931618554521)).toBeLessThan(0.0001);
      });
    });

    describe('var', () => {
      it('computes variance of non-masked values', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, true, false] });
        const variance = arr.var();
        expect(Math.abs(variance - 2.6666666666666665)).toBeLessThan(0.0001);
      });
    });

    describe('prod', () => {
      it('computes product of non-masked values', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, true, false] });
        expect(arr.prod()).toBe(15); // 1 * 3 * 5
      });
    });
  });

  // ==========================================================================
  // Trigonometric Functions
  // ==========================================================================

  describe('Trigonometric Functions', () => {
    describe('ma.sin', () => {
      it('computes sin with mask propagation', () => {
        const arr = ma.array([0, Math.PI / 2, Math.PI], { mask: [false, true, false] });
        const result = ma.sin(arr);
        expect(result.mask).not.toBe(false);
        const filled = result.filled(0);
        expect(Math.abs(filled.item(0))).toBeLessThan(0.0001);
      });
    });

    describe('ma.cos', () => {
      it('computes cos with mask propagation', () => {
        const arr = ma.array([0, Math.PI, 2 * Math.PI]);
        const result = ma.cos(arr);
        expect(Math.abs(result.data.item(0) - 1)).toBeLessThan(0.0001);
      });
    });

    describe('ma.tan', () => {
      it('computes tan', () => {
        const result = ma.tan(ma.array([0]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
      });
    });

    describe('ma.arcsin', () => {
      it('computes arcsin', () => {
        const result = ma.arcsin(ma.array([0, 1]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(1) - Math.PI / 2)).toBeLessThan(0.0001);
      });
    });

    describe('ma.arctan2', () => {
      it('computes arctan2', () => {
        const y = ma.array([1, 0]);
        const x = ma.array([0, 1]);
        const result = ma.arctan2(y, x);
        expect(Math.abs(result.data.item(0) - Math.PI / 2)).toBeLessThan(0.0001);
      });
    });
  });

  // ==========================================================================
  // Hyperbolic Functions
  // ==========================================================================

  describe('Hyperbolic Functions', () => {
    describe('ma.sinh', () => {
      it('computes sinh', () => {
        const result = ma.sinh(ma.array([0]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
      });
    });

    describe('ma.cosh', () => {
      it('computes cosh', () => {
        const result = ma.cosh(ma.array([0]));
        expect(Math.abs(result.data.item(0) - 1)).toBeLessThan(0.0001);
      });
    });

    describe('ma.tanh', () => {
      it('computes tanh', () => {
        const result = ma.tanh(ma.array([0]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
      });
    });
  });

  // ==========================================================================
  // Exponential and Logarithmic Functions
  // ==========================================================================

  describe('Exponential and Log Functions', () => {
    describe('ma.exp', () => {
      it('computes exp', () => {
        const result = ma.exp(ma.array([0, 1]));
        expect(Math.abs(result.data.item(0) - 1)).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(1) - Math.E)).toBeLessThan(0.0001);
      });
    });

    describe('ma.log', () => {
      it('computes log', () => {
        const result = ma.log(ma.array([1, Math.E]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(1) - 1)).toBeLessThan(0.0001);
      });
    });

    describe('ma.log10', () => {
      it('computes log10', () => {
        const result = ma.log10(ma.array([1, 10, 100]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(1) - 1)).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(2) - 2)).toBeLessThan(0.0001);
      });
    });

    describe('ma.log2', () => {
      it('computes log2', () => {
        const result = ma.log2(ma.array([1, 2, 4, 8]));
        expect(Math.abs(result.data.item(0))).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(1) - 1)).toBeLessThan(0.0001);
        expect(Math.abs(result.data.item(2) - 2)).toBeLessThan(0.0001);
      });
    });
  });

  // ==========================================================================
  // Rounding Functions
  // ==========================================================================

  describe('Rounding Functions', () => {
    describe('ma.around', () => {
      it('rounds values', () => {
        const result = ma.around(ma.array([1.4, 1.5, 2.5, 3.5]));
        expect(result.data.tolist()).toEqual([1, 2, 2, 4]);
      });
    });

    describe('ma.ceil', () => {
      it('computes ceiling', () => {
        const result = ma.ceil(ma.array([1.1, 2.5, 3.9]));
        expect(result.data.tolist()).toEqual([2, 3, 4]);
      });
    });

    describe('ma.floor', () => {
      it('computes floor', () => {
        const result = ma.floor(ma.array([1.9, 2.5, 3.1]));
        expect(result.data.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.trunc', () => {
      it('truncates values', () => {
        const result = ma.trunc(ma.array([1.9, -1.9, 2.5]));
        expect(result.data.tolist()).toEqual([1, -1, 2]);
      });
    });
  });

  // ==========================================================================
  // Logical Functions
  // ==========================================================================

  describe('Logical Functions', () => {
    describe('ma.logical_and', () => {
      it('computes logical AND', () => {
        const a = ma.array([true, true, false, false] as any);
        const b = ma.array([true, false, true, false] as any);
        const result = ma.logical_and(a, b);
        expect(result.data.tolist()).toEqual([1, 0, 0, 0]);
      });
    });

    describe('ma.logical_or', () => {
      it('computes logical OR', () => {
        const a = ma.array([true, true, false, false] as any);
        const b = ma.array([true, false, true, false] as any);
        const result = ma.logical_or(a, b);
        expect(result.data.tolist()).toEqual([1, 1, 1, 0]);
      });
    });

    describe('ma.logical_not', () => {
      it('computes logical NOT', () => {
        const a = ma.array([true, false] as any);
        const result = ma.logical_not(a);
        expect(result.data.tolist()).toEqual([0, 1]);
      });
    });
  });

  // ==========================================================================
  // Comparison Functions
  // ==========================================================================

  describe('Comparison Functions', () => {
    describe('ma.equal', () => {
      it('compares equal', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([1, 0, 3]);
        const result = ma.equal(a, b);
        expect(result.data.tolist()).toEqual([1, 0, 1]);
      });
    });

    describe('ma.not_equal', () => {
      it('compares not equal', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([1, 0, 3]);
        const result = ma.not_equal(a, b);
        expect(result.data.tolist()).toEqual([0, 1, 0]);
      });
    });

    describe('ma.greater', () => {
      it('compares greater', () => {
        const result = ma.greater(ma.array([1, 2, 3]), ma.array([0, 2, 4]));
        expect(result.data.tolist()).toEqual([1, 0, 0]);
      });
    });

    describe('ma.less', () => {
      it('compares less', () => {
        const result = ma.less(ma.array([1, 2, 3]), ma.array([0, 2, 4]));
        expect(result.data.tolist()).toEqual([0, 0, 1]);
      });
    });
  });

  // ==========================================================================
  // Shape Operations
  // ==========================================================================

  describe('Shape Operations', () => {
    describe('reshape', () => {
      it('reshapes with mask', () => {
        const arr = ma.array([1, 2, 3, 4, 5, 6], { mask: [false, true, false, true, false, true] });
        const reshaped = arr.reshape(2, 3);
        expect(reshaped.shape).toEqual([2, 3]);
        expect(reshaped.mask).not.toBe(false);
      });
    });

    describe('transpose', () => {
      it('transposes with mask', () => {
        const arr = ma.array(
          [
            [1, 2],
            [3, 4],
          ],
          {
            mask: [
              [false, true],
              [true, false],
            ],
          }
        );
        const transposed = arr.transpose();
        expect(transposed.shape).toEqual([2, 2]);
      });
    });

    describe('ravel', () => {
      it('flattens with mask', () => {
        const arr = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const raveled = arr.ravel();
        expect(raveled.shape).toEqual([4]);
      });
    });
  });

  // ==========================================================================
  // Concatenation Functions
  // ==========================================================================

  describe('Concatenation Functions', () => {
    describe('ma.concatenate', () => {
      it('concatenates masked arrays', () => {
        const a = ma.array([1, 2], { mask: [false, true] });
        const b = ma.array([3, 4]);
        const result = ma.concatenate([a, b], 0);
        expect(result.data.tolist()).toEqual([1, 2, 3, 4]);
      });
    });

    describe('ma.vstack', () => {
      it('stacks vertically', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.vstack([a, b]);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('ma.hstack', () => {
      it('stacks horizontally', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.hstack([a, b]);
        expect(result.shape).toEqual([6]);
      });
    });
  });

  // ==========================================================================
  // Linear Algebra Functions
  // ==========================================================================

  describe('Linear Algebra Functions', () => {
    describe('ma.dot', () => {
      it('computes dot product', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.dot(a, b);
        expect(result).toBe(32); // 1*4 + 2*5 + 3*6
      });
    });

    describe('ma.inner', () => {
      it('computes inner product', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.inner(a, b);
        expect(result).toBe(32);
      });
    });

    describe('ma.outer', () => {
      it('computes outer product', () => {
        const a = ma.array([1, 2]);
        const b = ma.array([3, 4]);
        const result = ma.outer(a, b);
        expect(result.shape).toEqual([2, 2]);
        expect(result.data.tolist()).toEqual([
          [3, 4],
          [6, 8],
        ]);
      });
    });
  });

  // ==========================================================================
  // Set Operations
  // ==========================================================================

  describe('Set Operations', () => {
    describe('ma.unique', () => {
      it('returns unique non-masked values with masked entry at end', () => {
        // NumPy behavior: returns unique unmasked values plus one masked entry at end
        const arr = ma.array([1, 2, 2, 3, 3, 3], {
          mask: [false, false, true, false, false, true],
        });
        const result = ma.unique(arr);
        // compressed() gives just the unique unmasked values
        expect(result.compressed().tolist()).toEqual([1, 2, 3]);
        // Should have one masked entry at the end
        expect(result.count()).toBe(3);
        expect(result.size).toBe(4); // 3 unique values + 1 masked
      });

      it('returns unique values for unmasked array', () => {
        const arr = ma.array([1, 2, 2, 3, 3, 3]);
        const result = ma.unique(arr);
        expect(result.data.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.intersect1d', () => {
      it('returns intersection', () => {
        const a = ma.array([1, 2, 3, 4]);
        const b = ma.array([3, 4, 5, 6]);
        const result = ma.intersect1d(a, b);
        expect(result.data.tolist()).toEqual([3, 4]);
      });
    });

    describe('ma.union1d', () => {
      it('returns union', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([3, 4, 5]);
        const result = ma.union1d(a, b);
        expect(result.data.tolist()).toEqual([1, 2, 3, 4, 5]);
      });
    });
  });

  // ==========================================================================
  // Additional Functions
  // ==========================================================================

  describe('Additional Functions', () => {
    describe('ma.clip', () => {
      it('clips values', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        const result = ma.clip(arr, 2, 4);
        expect(result.data.tolist()).toEqual([2, 2, 3, 4, 4]);
      });
    });

    describe('ma.maximum', () => {
      it('computes element-wise maximum', () => {
        const a = ma.array([1, 4, 3]);
        const b = ma.array([2, 2, 4]);
        const result = ma.maximum(a, b);
        expect(result.data.tolist()).toEqual([2, 4, 4]);
      });
    });

    describe('ma.minimum', () => {
      it('computes element-wise minimum', () => {
        const a = ma.array([1, 4, 3]);
        const b = ma.array([2, 2, 4]);
        const result = ma.minimum(a, b);
        expect(result.data.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.diff', () => {
      it('computes differences', () => {
        const arr = ma.array([1, 3, 6, 10]);
        const result = ma.diff(arr);
        expect(result.data.tolist()).toEqual([2, 3, 4]);
      });
    });

    describe('ma.diag', () => {
      it('extracts diagonal', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = ma.diag(arr);
        expect(result.data.tolist()).toEqual([1, 5, 9]);
      });
    });

    describe('ma.allclose', () => {
      it('checks if arrays are close', () => {
        const a = ma.array([1.0, 2.0, 3.0]);
        const b = ma.array([1.0001, 2.0001, 3.0001]);
        expect(ma.allclose(a, b, 0.001)).toBe(true);
      });
    });

    describe('ma.allequal', () => {
      it('checks if arrays are equal', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([1, 2, 3]);
        expect(ma.allequal(a, b)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Iterator Functions
  // ==========================================================================

  describe('Iterator Functions', () => {
    describe('ma.ndenumerate', () => {
      it('iterates with indices', () => {
        const arr = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const items = [...ma.ndenumerate(arr)];
        expect(items.length).toBe(4);
        expect(items[0]).toEqual([[0, 0], 1]);
        expect(items[3]).toEqual([[1, 1], 4]);
      });
    });
  });

  // ==========================================================================
  // Fill Value Functions
  // ==========================================================================

  describe('Fill Value Functions', () => {
    describe('ma.default_fill_value', () => {
      it('returns default fill for dtype', () => {
        expect(ma.default_fill_value('float64')).toBe(1e20);
        expect(ma.default_fill_value('int32')).toBe(999999);
        expect(ma.default_fill_value('bool')).toBe(true);
      });
    });

    describe('ma.maximum_fill_value', () => {
      it('returns max fill value for dtype', () => {
        const arr = ma.array([1, 2, 3]); // float64 by default
        const maxFill = ma.maximum_fill_value(arr);
        expect(maxFill).toBe(Infinity);
      });
    });

    describe('ma.minimum_fill_value', () => {
      it('returns min fill value for dtype', () => {
        const arr = ma.array([1, 2, 3]); // float64 by default
        const minFill = ma.minimum_fill_value(arr);
        expect(minFill).toBe(-Infinity);
      });
    });
  });

  // ==========================================================================
  // Priority 1: Core Operations - Arithmetic
  // ==========================================================================

  describe('Arithmetic Operations (Extended)', () => {
    describe('ma.absolute', () => {
      it('computes absolute value with mask', () => {
        const arr = ma.array([-1, -2, 3, -4], { mask: [false, true, false, false] });
        const result = ma.absolute(arr);
        expect(result.filled(-999).tolist()).toEqual([1, -999, 3, 4]);
      });

      it('computes absolute value without mask', () => {
        const arr = ma.array([-1, -2, -3]);
        const result = ma.absolute(arr);
        expect(result.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.mod', () => {
      it('computes modulo with mask', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const b = ma.array([3, 3, 3]);
        const result = ma.mod(a, b);
        expect(result.filled(-999).tolist()).toEqual([1, -999, 0]);
      });

      it('computes modulo without mask', () => {
        const a = ma.array([10, 20, 30]);
        const result = ma.mod(a, 7);
        expect(result.tolist()).toEqual([3, 6, 2]);
      });
    });

    describe('ma.floor_divide', () => {
      it('computes floor division with mask', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const result = ma.floor_divide(a, 3);
        expect(result.filled(-999).tolist()).toEqual([3, -999, 10]);
      });

      it('computes floor division without mask', () => {
        const a = ma.array([10, 20, 30]);
        const result = ma.floor_divide(a, 4);
        expect(result.tolist()).toEqual([2, 5, 7]);
      });
    });

    describe('ma.true_divide', () => {
      it('computes true division with mask', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const result = ma.true_divide(a, 4);
        expect(result.filled(-999).tolist()).toEqual([2.5, -999, 7.5]);
      });
    });

    describe('ma.remainder', () => {
      it('computes remainder with mask', () => {
        const a = ma.array([10, 20, 30], { mask: [false, true, false] });
        const result = ma.remainder(a, 7);
        expect(result.filled(-999).tolist()).toEqual([3, -999, 2]);
      });
    });

    describe('ma.fmod', () => {
      it('computes float modulo with mask', () => {
        const a = ma.array([10.5, 20.5, 30.5], { mask: [false, true, false] });
        const result = ma.fmod(a, 3);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.fabs', () => {
      it('computes float absolute value with mask', () => {
        const arr = ma.array([-1.5, -2.5, 3.5], { mask: [false, true, false] });
        const result = ma.fabs(arr);
        expect(result.filled(-999).tolist()).toEqual([1.5, -999, 3.5]);
      });
    });
  });

  // ==========================================================================
  // Priority 1: Core Operations - Reductions
  // ==========================================================================

  describe('Reduction Operations (Extended)', () => {
    describe('ma.all', () => {
      it('returns true if all non-masked are truthy', () => {
        const arr = ma.array([1, 2, 0, 4], { mask: [false, false, true, false] });
        expect(ma.all(arr)).toBe(true);
      });

      it('returns false if any non-masked is falsy', () => {
        const arr = ma.array([1, 0, 3, 4]);
        expect(ma.all(arr)).toBe(false);
      });
    });

    describe('ma.any', () => {
      it('returns true if any non-masked is truthy', () => {
        const arr = ma.array([0, 0, 1, 0], { mask: [false, false, false, false] });
        expect(ma.any(arr)).toBe(true);
      });

      it('returns false if all non-masked are falsy', () => {
        const arr = ma.array([0, 0, 1, 0], { mask: [false, false, true, false] });
        expect(ma.any(arr)).toBe(false);
      });
    });

    describe('ma.argmax', () => {
      it('returns index of max non-masked value', () => {
        const arr = ma.array([1, 5, 3, 4], { mask: [false, true, false, false] });
        expect(ma.argmax(arr)).toBe(3); // index of 4 (5 is masked)
      });

      it('returns index of max without mask', () => {
        const arr = ma.array([1, 5, 3, 4]);
        expect(ma.argmax(arr)).toBe(1);
      });
    });

    describe('ma.argmin', () => {
      it('returns index of min non-masked value', () => {
        const arr = ma.array([5, 1, 3, 4], { mask: [false, true, false, false] });
        expect(ma.argmin(arr)).toBe(2); // index of 3 (1 is masked)
      });

      it('returns index of min without mask', () => {
        const arr = ma.array([5, 1, 3, 4]);
        expect(ma.argmin(arr)).toBe(1);
      });
    });

    describe('ma.median', () => {
      it('computes median of non-masked values', () => {
        const arr = ma.array([1, 100, 2, 3, 4], { mask: [false, true, false, false, false] });
        const result = ma.median(arr);
        expect(result).toBe(2.5); // median of [1, 2, 3, 4]
      });

      it('computes median without mask', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        expect(ma.median(arr)).toBe(3);
      });
    });

    describe('ma.ptp', () => {
      it('computes peak-to-peak (max - min) with mask', () => {
        const arr = ma.array([1, 100, 5, 10], { mask: [false, true, false, false] });
        expect(ma.ptp(arr)).toBe(9); // 10 - 1
      });

      it('computes peak-to-peak without mask', () => {
        const arr = ma.array([1, 5, 10]);
        expect(ma.ptp(arr)).toBe(9);
      });
    });

    describe('ma.cumprod', () => {
      it('computes cumulative product with mask', () => {
        const arr = ma.array([1, 2, 3, 4], { mask: [false, true, false, false] });
        const result = ma.cumprod(arr);
        expect(result.compressed().tolist().length).toBeGreaterThan(0);
      });

      it('computes cumulative product without mask', () => {
        const arr = ma.array([1, 2, 3, 4]);
        const result = ma.cumprod(arr);
        expect(result.tolist()).toEqual([1, 2, 6, 24]);
      });
    });

    describe('ma.cumsum', () => {
      it('computes cumulative sum with mask', () => {
        const arr = ma.array([1, 2, 3, 4], { mask: [false, true, false, false] });
        const result = ma.cumsum(arr);
        expect(result.compressed().tolist().length).toBeGreaterThan(0);
      });

      it('computes cumulative sum without mask', () => {
        const arr = ma.array([1, 2, 3, 4]);
        const result = ma.cumsum(arr);
        expect(result.tolist()).toEqual([1, 3, 6, 10]);
      });
    });

    describe('ma.average', () => {
      it('computes weighted average with mask', () => {
        const arr = ma.array([1, 2, 3, 4], { mask: [false, true, false, false] });
        const result = ma.average(arr);
        expect(typeof result).toBe('number');
      });

      it('computes simple average without weights', () => {
        const arr = ma.array([1, 2, 3, 4]);
        const result = ma.average(arr);
        expect(result).toBe(2.5);
      });
    });

    describe('ma.anom', () => {
      it('computes anomalies (deviations from mean)', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        const result = ma.anom(arr);
        // mean is 3, so anomalies are [-2, -1, 0, 1, 2]
        expect(result.tolist()).toEqual([-2, -1, 0, 1, 2]);
      });

      it('computes anomalies with mask', () => {
        const arr = ma.array([1, 100, 3, 5], { mask: [false, true, false, false] });
        const result = ma.anom(arr);
        expect(result.compressed().tolist().length).toBe(3);
      });
    });

    describe('ma.count_masked', () => {
      it('counts masked elements', () => {
        const arr = ma.array([1, 2, 3, 4], { mask: [false, true, true, false] });
        expect(ma.count_masked(arr)).toBe(2);
      });

      it('returns 0 for unmasked array', () => {
        const arr = ma.array([1, 2, 3, 4]);
        expect(ma.count_masked(arr)).toBe(0);
      });
    });
  });

  // ==========================================================================
  // Priority 1: Statistics
  // ==========================================================================

  describe('Statistics (Extended)', () => {
    describe('ma.cov', () => {
      it('computes covariance matrix', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = ma.cov(arr);
        expect(result.shape.length).toBe(2);
      });
    });

    describe('ma.corrcoef', () => {
      it('computes correlation coefficients', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = ma.corrcoef(arr);
        expect(result.shape.length).toBe(2);
      });
    });

    describe('ma.var_', () => {
      it('computes variance (alias)', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, false, false, false] });
        const result = ma.var_(arr);
        expect(typeof result).toBe('number');
      });
    });
  });

  // ==========================================================================
  // Priority 2: Array Manipulation - Shape Operations
  // ==========================================================================

  describe('Shape Operations (Extended)', () => {
    describe('ma.squeeze', () => {
      it('removes single-dimensional axes', () => {
        const arr = ma.array([[[1, 2, 3]]]);
        const result = ma.squeeze(arr);
        expect(result.shape).toEqual([3]);
      });

      it('preserves mask through squeeze', () => {
        // Create 1D array, then reshape to 3D to have proper mask handling
        const arr = ma.array([1, 2, 3], { mask: [0, 1, 0] });
        const expanded = ma.expand_dims(ma.expand_dims(arr, 0), 0);
        const result = ma.squeeze(expanded);
        expect(result.compressed().tolist()).toEqual([1, 3]);
      });
    });

    describe('ma.expand_dims', () => {
      it('adds axis at position', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.expand_dims(arr, 0);
        expect(result.shape).toEqual([1, 3]);
      });

      it('preserves mask through expand_dims', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.expand_dims(arr, 0);
        expect(result.compressed().tolist()).toEqual([1, 3]);
      });
    });

    describe('ma.swapaxes', () => {
      it('swaps two axes', () => {
        const arr = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const result = ma.swapaxes(arr, 0, 1);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    describe('ma.atleast_1d', () => {
      it('ensures at least 1D', () => {
        const arr = ma.array([5]);
        const result = ma.atleast_1d(arr);
        expect(result.shape.length).toBeGreaterThanOrEqual(1);
      });
    });

    describe('ma.atleast_2d', () => {
      it('ensures at least 2D', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.atleast_2d(arr);
        expect(result.shape.length).toBeGreaterThanOrEqual(2);
      });
    });

    describe('ma.atleast_3d', () => {
      it('ensures at least 3D', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.atleast_3d(arr);
        expect(result.shape.length).toBeGreaterThanOrEqual(3);
      });
    });

    describe('ma.repeat', () => {
      it('repeats elements', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.repeat(arr, 2);
        expect(result.tolist()).toEqual([1, 1, 2, 2, 3, 3]);
      });

      it('repeats with mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.repeat(arr, 2);
        expect(result.size).toBe(6);
      });
    });

    describe('ma.resize', () => {
      it('resizes array to new shape', () => {
        const arr = ma.array([1, 2, 3, 4]);
        const result = ma.resize(arr, [2, 2]);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    describe('ma.take', () => {
      it('takes elements by indices', () => {
        const arr = ma.array([10, 20, 30, 40, 50]);
        const result = ma.take(arr, [0, 2, 4]);
        expect(result.tolist()).toEqual([10, 30, 50]);
      });

      it('takes with mask', () => {
        const arr = ma.array([10, 20, 30, 40, 50], { mask: [false, true, false, true, false] });
        const result = ma.take(arr, [0, 1, 2]);
        expect(result.filled(-1).tolist()).toEqual([10, -1, 30]);
      });
    });

    describe('ma.put', () => {
      it('puts values at indices', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        ma.put(arr, [0, 2], [10, 30]);
        expect(arr.tolist()).toEqual([10, 2, 30, 4, 5]);
      });
    });

    describe('ma.putmask', () => {
      it('puts values where mask is true', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        const mask = [true, false, true, false, true];
        ma.putmask(arr, mask, [0]);
        expect(arr.tolist()).toEqual([0, 2, 0, 4, 0]);
      });
    });
  });

  // ==========================================================================
  // Priority 2: Stacking/Splitting
  // ==========================================================================

  describe('Stacking/Splitting (Extended)', () => {
    describe('ma.column_stack', () => {
      it('stacks 1D arrays as columns', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.column_stack([a, b]);
        expect(result.shape).toEqual([3, 2]);
      });
    });

    describe('ma.row_stack', () => {
      it('stacks arrays as rows', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.row_stack([a, b]);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('ma.dstack', () => {
      it('stacks along third axis', () => {
        const a = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const b = ma.array([
          [5, 6],
          [7, 8],
        ]);
        const result = ma.dstack([a, b]);
        expect(result.shape).toEqual([2, 2, 2]);
      });
    });

    describe('ma.stack', () => {
      it('stacks along new axis', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([4, 5, 6]);
        const result = ma.stack([a, b], 0);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('ma.hsplit', () => {
      it('splits horizontally', () => {
        const arr = ma.array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]);
        const result = ma.hsplit(arr, 2);
        expect(result.length).toBe(2);
        expect(result[0].shape).toEqual([2, 2]);
      });
    });

    describe('ma.append', () => {
      it('appends values to array', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.append(arr, [4, 5]);
        expect(result.tolist()).toEqual([1, 2, 3, 4, 5]);
      });
    });
  });

  // ==========================================================================
  // Priority 2: Sorting
  // ==========================================================================

  describe('Sorting (Extended)', () => {
    describe('ma.sort', () => {
      it('sorts array with mask', () => {
        const arr = ma.array([3, 1, 4, 1, 5], { mask: [false, false, true, false, false] });
        const result = ma.sort(arr);
        expect(result.compressed().tolist().length).toBe(4);
      });

      it('sorts array without mask', () => {
        const arr = ma.array([3, 1, 4, 1, 5]);
        const result = ma.sort(arr);
        expect(result.tolist()).toEqual([1, 1, 3, 4, 5]);
      });
    });

    describe('ma.argsort', () => {
      it('returns indices that would sort with mask', () => {
        const arr = ma.array([3, 1, 4, 1, 5], { mask: [false, false, true, false, false] });
        const result = ma.argsort(arr);
        expect(result.size).toBe(5);
      });

      it('returns indices that would sort without mask', () => {
        const arr = ma.array([3, 1, 4]);
        const result = ma.argsort(arr);
        expect(result.tolist()).toEqual([1, 0, 2]);
      });
    });
  });

  // ==========================================================================
  // Priority 3: Masking Operations
  // ==========================================================================

  describe('Mask Manipulation (Extended)', () => {
    describe('ma.mask_or', () => {
      it('ORs two masks', () => {
        const m1 = [1, 0, 0, 1];
        const m2 = [0, 1, 0, 1];
        const result = ma.mask_or(m1, m2);
        expect((result as NDArray).tolist()).toEqual([1, 1, 0, 1]);
      });
    });

    describe('ma.harden_mask', () => {
      it('hardens the mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        ma.harden_mask(arr);
        expect(arr.hardmask).toBe(true);
      });
    });

    describe('ma.soften_mask', () => {
      it('softens the mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false], hard_mask: true });
        ma.soften_mask(arr);
        expect(arr.hardmask).toBe(false);
      });
    });

    describe('ma.set_fill_value', () => {
      it('sets fill value', () => {
        const arr = ma.array([1, 2, 3]);
        ma.set_fill_value(arr, -999);
        expect(arr.fill_value).toBe(-999);
      });
    });

    describe('ma.fix_invalid', () => {
      it('masks invalid values', () => {
        const arr = ma.array([1, NaN, 3, Infinity]);
        const result = ma.fix_invalid(arr);
        expect(result.count()).toBe(2);
      });
    });

    describe('ma.is_mask', () => {
      it('returns true for valid mask', () => {
        const mask = [true, false, true];
        expect(ma.is_mask(mask)).toBe(true);
      });
    });

    describe('ma.getdata', () => {
      it('gets data without mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const data = ma.getdata(arr);
        expect(data.tolist()).toEqual([1, 2, 3]);
      });
    });
  });

  // ==========================================================================
  // Priority 3: Mask Finding
  // ==========================================================================

  describe('Mask Finding (Extended)', () => {
    describe('ma.nonzero', () => {
      it('finds non-zero indices', () => {
        const arr = ma.array([0, 1, 0, 2, 0, 3]);
        const result = ma.nonzero(arr);
        expect(result.length).toBeGreaterThan(0);
      });
    });

    describe('ma.where', () => {
      it('finds where condition is true', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        const result = ma.where(arr.data.greater(2), arr, ma.array([0, 0, 0, 0, 0])) as MaskedArray;
        expect(result.tolist()).toEqual([0, 0, 3, 4, 5]);
      });
    });

    describe('ma.choose', () => {
      it('chooses from arrays by index', () => {
        const choices = [ma.array([1, 2, 3]), ma.array([10, 20, 30]), ma.array([100, 200, 300])];
        const indices = ma.array([0, 1, 2]);
        const result = ma.choose(indices, choices);
        expect(result.tolist()).toEqual([1, 20, 300]);
      });
    });

    describe('ma.compress', () => {
      it('compresses by condition', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        const condition = [true, false, true, false, true];
        const result = ma.compress(condition, arr);
        expect(result.tolist()).toEqual([1, 3, 5]);
      });
    });
  });

  // ==========================================================================
  // Priority 3: Contiguous Regions
  // ==========================================================================

  describe('Contiguous Regions', () => {
    describe('ma.clump_masked', () => {
      it('finds contiguous masked regions', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, true, false, true] });
        const result = ma.clump_masked(arr);
        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('ma.clump_unmasked', () => {
      it('finds contiguous unmasked regions', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, true, true, false, true] });
        const result = ma.clump_unmasked(arr);
        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('ma.notmasked_edges', () => {
      it('finds first and last unmasked indices', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [true, false, false, false, true] });
        const result = ma.notmasked_edges(arr);
        expect(result).not.toBeNull();
      });
    });

    describe('ma.notmasked_contiguous', () => {
      it('finds contiguous unmasked slices', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, false, true, false, false] });
        const result = ma.notmasked_contiguous(arr);
        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('ma.flatnotmasked_edges', () => {
      it('finds flat first and last unmasked', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [true, false, false, false, true] });
        const result = ma.flatnotmasked_edges(arr);
        expect(result).not.toBeNull();
      });
    });

    describe('ma.flatnotmasked_contiguous', () => {
      it('finds flat contiguous unmasked', () => {
        const arr = ma.array([1, 2, 3, 4, 5], { mask: [false, false, true, false, false] });
        const result = ma.flatnotmasked_contiguous(arr);
        expect(Array.isArray(result)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Priority 4: Mathematical Functions
  // ==========================================================================

  describe('Mathematical Functions (Extended)', () => {
    describe('ma.arccos', () => {
      it('computes inverse cosine with mask', () => {
        const arr = ma.array([0, 0.5, 1], { mask: [false, true, false] });
        const result = ma.arccos(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.arctan', () => {
      it('computes inverse tangent with mask', () => {
        const arr = ma.array([0, 1, -1], { mask: [false, true, false] });
        const result = ma.arctan(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.arcsinh', () => {
      it('computes inverse hyperbolic sine with mask', () => {
        const arr = ma.array([0, 1, 2], { mask: [false, true, false] });
        const result = ma.arcsinh(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.arccosh', () => {
      it('computes inverse hyperbolic cosine with mask', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.arccosh(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.arctanh', () => {
      it('computes inverse hyperbolic tangent with mask', () => {
        const arr = ma.array([0, 0.5, -0.5], { mask: [false, true, false] });
        const result = ma.arctanh(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.exp2', () => {
      it('computes 2^x with mask', () => {
        const arr = ma.array([0, 1, 2, 3], { mask: [false, true, false, false] });
        const result = ma.exp2(arr);
        expect(result.filled(-1).tolist()).toEqual([1, -1, 4, 8]);
      });
    });

    describe('ma.expm1', () => {
      it('computes exp(x) - 1 with mask', () => {
        const arr = ma.array([0, 1, 2], { mask: [false, true, false] });
        const result = ma.expm1(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.log1p', () => {
      it('computes log(1 + x) with mask', () => {
        const arr = ma.array([0, 1, 2], { mask: [false, true, false] });
        const result = ma.log1p(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.angle', () => {
      it('computes angle for complex numbers', () => {
        const arr = ma.array([1, -1, 1]);
        const result = ma.angle(arr);
        expect(result.size).toBe(3);
      });
    });

    describe('ma.conjugate', () => {
      it('computes complex conjugate', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.conjugate(arr);
        expect(result.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.hypot', () => {
      it('computes hypotenuse with mask', () => {
        const a = ma.array([3, 5, 8], { mask: [false, true, false] });
        const b = ma.array([4, 12, 15]);
        const result = ma.hypot(a, b);
        expect(result.filled(-1).tolist()).toEqual([5, -1, 17]);
      });
    });

    describe('ma.fix', () => {
      it('rounds toward zero', () => {
        const arr = ma.array([-2.5, -1.5, 1.5, 2.5], { mask: [false, true, false, false] });
        const result = ma.fix(arr);
        expect(result.filled(-999).tolist()).toEqual([-2, -999, 1, 2]);
      });
    });

    describe('ma.rint', () => {
      it('rounds to nearest integer', () => {
        const arr = ma.array([1.4, 1.5, 1.6], { mask: [false, true, false] });
        const result = ma.rint(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });
  });

  // ==========================================================================
  // Priority 5: Linear Algebra & Polynomials
  // ==========================================================================

  describe('Linear Algebra (Extended)', () => {
    describe('ma.diagonal', () => {
      it('gets diagonal with mask', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = ma.diagonal(arr);
        expect(result.tolist()).toEqual([1, 5, 9]);
      });
    });

    describe('ma.diagflat', () => {
      it('creates diagonal matrix', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.diagflat(arr);
        expect(result.shape).toEqual([3, 3]);
      });
    });

    describe('ma.trace', () => {
      it('computes trace with mask', () => {
        const arr = ma.array([
          [1, 2],
          [3, 4],
        ]);
        const result = ma.trace(arr);
        expect(result).toBe(5);
      });
    });

    describe('ma.vander', () => {
      it('creates Vandermonde matrix', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.vander(arr, 3);
        expect(result.shape).toEqual([3, 3]);
      });
    });

    describe('ma.polyfit', () => {
      it('fits polynomial', () => {
        const x = ma.array([0, 1, 2, 3, 4]);
        const y = ma.array([0, 1, 4, 9, 16]);
        const result = ma.polyfit(x, y, 2);
        expect(result.size).toBe(3);
      });
    });

    describe('ma.convolve', () => {
      it('convolves two arrays', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([0, 1, 0.5]);
        const result = ma.convolve(a, b);
        expect(result.size).toBeGreaterThan(0);
      });
    });

    describe('ma.correlate', () => {
      it('correlates two arrays', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([0, 1, 0.5]);
        const result = ma.correlate(a, b);
        expect(result.size).toBeGreaterThan(0);
      });
    });
  });

  // ==========================================================================
  // Priority 6: Set Operations
  // ==========================================================================

  describe('Set Operations (Extended)', () => {
    describe('ma.setdiff1d', () => {
      it('computes set difference', () => {
        const a = ma.array([1, 2, 3, 4, 5]);
        const b = ma.array([3, 4, 5, 6, 7]);
        const result = ma.setdiff1d(a, b);
        expect(result.tolist()).toEqual([1, 2]);
      });
    });

    describe('ma.setxor1d', () => {
      it('computes set XOR', () => {
        const a = ma.array([1, 2, 3]);
        const b = ma.array([2, 3, 4]);
        const result = ma.setxor1d(a, b);
        expect(result.tolist()).toEqual([1, 4]);
      });
    });

    describe('ma.in1d', () => {
      it('tests membership', () => {
        const a = ma.array([1, 2, 3, 4, 5]);
        const b = ma.array([2, 4]);
        const result = ma.in1d(a, b);
        expect(result.tolist()).toEqual([0, 1, 0, 1, 0]);
      });
    });

    describe('ma.isin', () => {
      it('tests membership (newer API)', () => {
        const a = ma.array([1, 2, 3, 4, 5]);
        const b = ma.array([2, 4]);
        const result = ma.isin(a, b);
        expect(result.tolist()).toEqual([0, 1, 0, 1, 0]);
      });
    });
  });

  // ==========================================================================
  // Priority 7: Bitwise Operations
  // ==========================================================================

  describe('Bitwise Operations (Extended)', () => {
    describe('ma.bitwise_xor', () => {
      it('computes bitwise XOR with mask', () => {
        const a = ma.array([5, 3, 7], { mask: [false, true, false], dtype: 'int32' });
        const b = ma.array([3, 5, 1], { dtype: 'int32' });
        const result = ma.bitwise_xor(a, b);
        expect(result.filled(-1).tolist()).toEqual([6, -1, 6]);
      });
    });

    describe('ma.left_shift', () => {
      it('computes left shift with mask', () => {
        const arr = ma.array([1, 2, 4], { mask: [false, true, false], dtype: 'int32' });
        const result = ma.left_shift(arr, 2);
        expect(result.filled(-1).tolist()).toEqual([4, -1, 16]);
      });
    });

    describe('ma.right_shift', () => {
      it('computes right shift with mask', () => {
        const arr = ma.array([4, 8, 16], { mask: [false, true, false], dtype: 'int32' });
        const result = ma.right_shift(arr, 2);
        expect(result.filled(-1).tolist()).toEqual([1, -1, 4]);
      });
    });

    describe('ma.invert', () => {
      it('computes bitwise NOT with mask', () => {
        const arr = ma.array([0, 1, 2], { mask: [false, true, false], dtype: 'int32' });
        const result = ma.invert(arr);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });
  });

  // ==========================================================================
  // Priority 8: Logical Operations
  // ==========================================================================

  describe('Logical Operations (Extended)', () => {
    describe('ma.logical_xor', () => {
      it('computes logical XOR with mask', () => {
        const a = ma.array([1, 0, 1, 0], { mask: [false, true, false, false] });
        const b = ma.array([1, 1, 0, 0]);
        const result = ma.logical_xor(a, b);
        expect(result.compressed().tolist().length).toBe(3);
      });
    });

    describe('ma.greater_equal', () => {
      it('compares greater or equal with mask', () => {
        const a = ma.array([1, 2, 3], { mask: [false, true, false] });
        const b = ma.array([2, 2, 2]);
        const result = ma.greater_equal(a, b);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });

    describe('ma.less_equal', () => {
      it('compares less or equal with mask', () => {
        const a = ma.array([1, 2, 3], { mask: [false, true, false] });
        const b = ma.array([2, 2, 2]);
        const result = ma.less_equal(a, b);
        expect(result.compressed().tolist().length).toBe(2);
      });
    });
  });

  // ==========================================================================
  // Priority 9: Array Creation & Conversion
  // ==========================================================================

  describe('Array Creation (Extended)', () => {
    describe('ma.zeros_like', () => {
      it('creates zeros like another array', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.zeros_like(arr);
        expect(result.shape).toEqual([3]);
        // Underlying data is zeros, mask is preserved
        expect(result.data.tolist()).toEqual([0, 0, 0]);
        expect(result.compressed().tolist()).toEqual([0, 0]);
      });
    });

    describe('ma.ones_like', () => {
      it('creates ones like another array', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.ones_like(arr);
        // Underlying data is ones, mask is preserved
        expect(result.data.tolist()).toEqual([1, 1, 1]);
        expect(result.compressed().tolist()).toEqual([1, 1]);
      });
    });

    describe('ma.empty_like', () => {
      it('creates empty like another array', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.empty_like(arr);
        expect(result.shape).toEqual([3]);
      });
    });

    describe('ma.copy', () => {
      it('creates a copy', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.copy(arr);
        // Underlying data is copied
        expect(result.data.tolist()).toEqual([1, 2, 3]);
        expect(result.compressed().tolist()).toEqual([1, 3]);
      });
    });

    describe('ma.asarray', () => {
      it('converts to MaskedArray', () => {
        const arr = [1, 2, 3];
        const result = ma.asarray(arr);
        expect(result instanceof MaskedArray).toBe(true);
      });
    });

    describe('ma.asanyarray', () => {
      it('converts preserving subclass', () => {
        const arr = ma.array([1, 2, 3]);
        const result = ma.asanyarray(arr);
        expect(result instanceof MaskedArray).toBe(true);
      });
    });

    describe('ma.identity', () => {
      it('creates identity matrix', () => {
        const result = ma.identity(3);
        expect(result.shape).toEqual([3, 3]);
      });
    });

    describe('ma.indices', () => {
      it('creates grid indices', () => {
        const result = ma.indices([2, 3]);
        expect(result.shape[0]).toBe(2);
      });
    });

    describe('ma.frombuffer', () => {
      it('creates from buffer', () => {
        const buffer = new Float64Array([1, 2, 3]);
        const result = ma.frombuffer(buffer);
        expect(result.tolist()).toEqual([1, 2, 3]);
      });
    });

    describe('ma.fromfunction', () => {
      it('creates from function', () => {
        const result = ma.fromfunction((i: number) => i * 2, [5]);
        expect(result.tolist()).toEqual([0, 2, 4, 6, 8]);
      });
    });

    describe('ma.masked_array', () => {
      it('is alias for array', () => {
        const result = ma.masked_array([1, 2, 3], { mask: [false, true, false] });
        expect(result instanceof MaskedArray).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Priority 10: Utility & Info
  // ==========================================================================

  describe('Utility Functions', () => {
    describe('ma.shape', () => {
      it('returns shape', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(ma.shape(arr)).toEqual([2, 3]);
      });
    });

    describe('ma.size', () => {
      it('returns size', () => {
        const arr = ma.array([1, 2, 3, 4, 5]);
        expect(ma.size(arr)).toBe(5);
      });
    });

    describe('ma.ndim', () => {
      it('returns ndim', () => {
        const arr = ma.array([
          [1, 2],
          [3, 4],
        ]);
        expect(ma.ndim(arr)).toBe(2);
      });
    });

    describe('ma.isarray', () => {
      it('checks if MaskedArray', () => {
        const arr = ma.array([1, 2, 3]);
        expect(ma.isarray(arr)).toBe(true);
      });

      it('returns false for non-MaskedArray', () => {
        expect(ma.isarray([1, 2, 3])).toBe(false);
      });
    });

    describe('ma.isMA', () => {
      it('checks if MaskedArray', () => {
        const arr = ma.array([1, 2, 3]);
        expect(ma.isMA(arr)).toBe(true);
      });
    });

    describe('ma.isMaskedArray', () => {
      it('checks if MaskedArray', () => {
        const arr = ma.array([1, 2, 3]);
        expect(ma.isMaskedArray(arr)).toBe(true);
      });
    });

    describe('ma.ids', () => {
      it('returns data and mask ids', () => {
        const arr = ma.array([1, 2, 3], { mask: [false, true, false] });
        const result = ma.ids(arr);
        // Returns a tuple [dataId, maskId]
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBe(2);
        expect(typeof result[0]).toBe('number');
        expect(typeof result[1]).toBe('number');
      });
    });

    describe('ma.ndenumerate', () => {
      it('enumerates with indices', () => {
        const arr = ma.array([1, 2, 3]);
        const items = [...ma.ndenumerate(arr)];
        expect(items.length).toBe(3);
        expect(items[0][0]).toEqual([0]);
        expect(items[0][1]).toBe(1);
      });
    });

    describe('ma.apply_along_axis', () => {
      it('applies function along axis', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = ma.apply_along_axis((x: NDArray) => x.sum() as number, 1, arr);
        expect(result.tolist()).toEqual([6, 15]);
      });
    });

    describe('ma.apply_over_axes', () => {
      it('applies function over axes', () => {
        const arr = ma.array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = ma.apply_over_axes(
          (a: NDArray, axis: number) => a.sum(axis) as NDArray,
          arr,
          [0]
        );
        expect(result.size).toBeGreaterThan(0);
      });
    });
  });

  // ==========================================================================
  // Priority 11: Miscellaneous
  // ==========================================================================

  describe('Miscellaneous', () => {
    describe('ma.ediff1d', () => {
      it('computes differences of flattened array', () => {
        const arr = ma.array([
          [1, 2],
          [3, 5],
        ]);
        const result = ma.ediff1d(arr);
        expect(result.tolist()).toEqual([1, 1, 2]);
      });
    });

    describe('ma.common_fill_value', () => {
      it('returns common fill value for compatible arrays', () => {
        const a = ma.array([1, 2, 3], { fill_value: -999 });
        const b = ma.array([4, 5, 6], { fill_value: -999 });
        expect(ma.common_fill_value(a, b)).toBe(-999);
      });
    });

    describe('ma.flatten_mask', () => {
      it('flattens a mask', () => {
        // Use array() to create NDArray mask, since flatten_mask expects MaskInput (boolean, boolean[], number[], or NDArray)
        const mask = array([
          [1, 0],
          [0, 1],
        ]);
        const result = ma.flatten_mask(mask);
        expect((result as NDArray).tolist()).toEqual([1, 0, 0, 1]);
      });
    });

    describe('ma.make_mask_none', () => {
      it('creates no-mask of given shape', () => {
        const result = ma.make_mask_none([3, 3]);
        expect(result.tolist()).toEqual([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ]);
      });
    });
  });
});
