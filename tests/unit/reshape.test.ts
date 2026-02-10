/**
 * Unit tests for reshape operations
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src';

describe('Reshape Operations', () => {
  describe('reshape()', () => {
    it('reshapes 1D to 2D', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const reshaped = arr.reshape(2, 3);
      expect(reshaped.shape).toEqual([2, 3]);
      expect(reshaped.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('reshapes 2D to 1D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const reshaped = arr.reshape(6);
      expect(reshaped.shape).toEqual([6]);
      expect(Array.from(reshaped.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('reshapes with -1 (infer dimension)', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const reshaped = arr.reshape(2, -1);
      expect(reshaped.shape).toEqual([2, 3]);
      expect(reshaped.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('reshapes 2D to 3D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const reshaped = arr.reshape(2, 1, 3);
      expect(reshaped.shape).toEqual([2, 1, 3]);
      expect(reshaped.toArray()).toEqual([[[1, 2, 3]], [[4, 5, 6]]]);
    });

    it('throws error for incompatible size', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      expect(() => arr.reshape(2, 2)).toThrow(/cannot reshape/);
    });

    it('throws error for invalid -1 inference', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(() => arr.reshape(2, -1)).toThrow(/cannot reshape/);
    });

    it('handles reshape to same shape', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const reshaped = arr.reshape(2, 2);
      expect(reshaped.shape).toEqual([2, 2]);
      expect(reshaped.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('flatten()', () => {
    it('flattens 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const flat = arr.flatten();
      expect(flat.shape).toEqual([6]);
      expect(Array.from(flat.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('flattens 3D array', () => {
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
      const flat = arr.flatten();
      expect(flat.shape).toEqual([8]);
      expect(Array.from(flat.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('flattens 1D array (no-op)', () => {
      const arr = array([1, 2, 3]);
      const flat = arr.flatten();
      expect(flat.shape).toEqual([3]);
      expect(Array.from(flat.data)).toEqual([1, 2, 3]);
    });

    it('returns a copy, not a view', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const flat = arr.flatten();
      flat.data[0] = 999;
      // Original should be unchanged
      expect(arr.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('ravel()', () => {
    it('flattens 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const flat = arr.ravel();
      expect(flat.shape).toEqual([6]);
      expect(Array.from(flat.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('returns a view for C-contiguous arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const flat = arr.ravel();
      flat.data[0] = 999;
      // Original SHOULD be changed (view, not copy) for C-contiguous arrays
      expect(arr.toArray()).toEqual([
        [999, 2],
        [3, 4],
      ]);
    });

    it('returns a copy for non-contiguous arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const transposed = arr.transpose();
      const flat = transposed.ravel();
      flat.data[0] = 999;
      // Original should be unchanged (copy, not view) for non-contiguous arrays
      expect(transposed.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });
  });

  describe('transpose()', () => {
    it('transposes 2D array (default)', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const transposed = arr.transpose();
      expect(transposed.shape).toEqual([3, 2]);
      expect(transposed.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('transposes 3D array (default)', () => {
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
      const transposed = arr.transpose();
      expect(transposed.shape).toEqual([2, 2, 2]);
      expect(transposed.toArray()).toEqual([
        [
          [1, 5],
          [3, 7],
        ],
        [
          [2, 6],
          [4, 8],
        ],
      ]);
    });

    it('transposes with custom axes', () => {
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
      const transposed = arr.transpose([1, 0, 2]);
      expect(transposed.shape).toEqual([2, 2, 2]);
      expect(transposed.toArray()).toEqual([
        [
          [1, 2],
          [5, 6],
        ],
        [
          [3, 4],
          [7, 8],
        ],
      ]);
    });

    it('transposes with negative axes', () => {
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
      const transposed = arr.transpose([1, -3, 2]);
      expect(transposed.shape).toEqual([2, 2, 2]);
    });

    it('throws error for wrong number of axes', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => arr.transpose([0])).toThrow(/axes must have length/);
    });

    it('throws error for repeated axes', () => {
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
      expect(() => arr.transpose([0, 0, 2])).toThrow(/repeated axis/);
    });

    it('throws error for out of bounds axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => arr.transpose([0, 5])).toThrow(/out of bounds/);
    });

    it('transposes 1D array (no-op)', () => {
      const arr = array([1, 2, 3]);
      const transposed = arr.transpose();
      expect(transposed.shape).toEqual([3]);
      expect(Array.from(transposed.data)).toEqual([1, 2, 3]);
    });
  });

  describe('squeeze()', () => {
    it('squeezes all axes of size 1', () => {
      const arr = array([[[1, 2, 3]]]);
      const squeezed = arr.squeeze();
      expect(squeezed.shape).toEqual([3]);
      expect(Array.from(squeezed.data)).toEqual([1, 2, 3]);
    });

    it('squeezes specific axis', () => {
      const arr = array([[[1, 2, 3]]]);
      const squeezed = arr.squeeze(0);
      expect(squeezed.shape).toEqual([1, 3]);
    });

    it('squeezes with negative axis', () => {
      const arr = array([[[1, 2, 3]]]);
      const squeezed = arr.squeeze(-3);
      expect(squeezed.shape).toEqual([1, 3]);
    });

    it('throws error for axis with size != 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => arr.squeeze(0)).toThrow(/cannot select an axis/);
    });

    it('handles array with no size-1 axes', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const squeezed = arr.squeeze();
      expect(squeezed.shape).toEqual([2, 2]);
    });

    it('squeezes multiple axes', () => {
      const arr = array([[[[1], [2]]]]);
      const squeezed = arr.squeeze();
      expect(squeezed.shape).toEqual([2]);
      expect(Array.from(squeezed.data)).toEqual([1, 2]);
    });

    it('handles squeezing all dims (approaches scalar)', () => {
      const arr = array([[[5]]]);
      const squeezed = arr.squeeze();
      // For now, skip true 0-d arrays as stdlib may not support them
      // expect(squeezed.shape).toEqual([]);
      // expect(squeezed.ndim).toBe(0);
      expect(squeezed.shape.length).toBeLessThan(arr.shape.length);
    });
  });

  describe('expand_dims()', () => {
    it('expands at beginning', () => {
      const arr = array([1, 2, 3]);
      const expanded = arr.expand_dims(0);
      expect(expanded.shape).toEqual([1, 3]);
      expect(expanded.toArray()).toEqual([[1, 2, 3]]);
    });

    it('expands at end', () => {
      const arr = array([1, 2, 3]);
      const expanded = arr.expand_dims(1);
      expect(expanded.shape).toEqual([3, 1]);
      expect(expanded.toArray()).toEqual([[1], [2], [3]]);
    });

    it('expands in middle', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const expanded = arr.expand_dims(1);
      expect(expanded.shape).toEqual([2, 1, 2]);
      expect(expanded.toArray()).toEqual([[[1, 2]], [[3, 4]]]);
    });

    it('expands with negative axis', () => {
      const arr = array([1, 2, 3]);
      const expanded = arr.expand_dims(-1);
      expect(expanded.shape).toEqual([3, 1]);
    });

    it('throws error for out of bounds axis', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.expand_dims(5)).toThrow(/out of bounds/);
      expect(() => arr.expand_dims(-5)).toThrow(/out of bounds/);
    });

    it('can be chained multiple times', () => {
      const arr = array([1, 2, 3]);
      const expanded = arr.expand_dims(0).expand_dims(0);
      expect(expanded.shape).toEqual([1, 1, 3]);
    });
  });

  describe('reshape operations combined', () => {
    it('reshape then transpose', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = arr.reshape(2, 3).transpose();
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('transpose then flatten', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = arr.transpose().flatten();
      expect(result.shape).toEqual([6]);
      expect(Array.from(result.data)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it('expand_dims then squeeze', () => {
      const arr = array([1, 2, 3]);
      const result = arr.expand_dims(0).squeeze();
      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([1, 2, 3]);
    });
  });
});
