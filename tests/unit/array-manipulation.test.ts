/**
 * Unit tests for array manipulation functions
 *
 * Tests: swapaxes, moveaxis, concatenate, stack, vstack, hstack, dstack,
 *        split, array_split, vsplit, hsplit, tile, repeat, and new functions
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  arange,
  swapaxes,
  moveaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
  ravel,
  reshape,
  squeeze,
  expand_dims,
  flip,
  fliplr,
  flipud,
  rot90,
  roll,
  rollaxis,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  dsplit,
  column_stack,
  row_stack,
  resize,
  append,
  insert,
  pad,
  flatten,
  broadcast_to,
} from '../../src';
// Note: delete is a reserved keyword, so we use delete_ from ndarray directly
import { delete_ } from '../../src';

describe('Array Manipulation', () => {
  // ========================================
  // swapaxes
  // ========================================
  describe('swapaxes', () => {
    it('swaps axes of 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, 0, 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('swaps axes of 3D array', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const result = swapaxes(arr, 0, 2);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
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

    it('handles negative axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, -2, -1);
      expect(result.shape).toEqual([3, 2]);
    });

    it('same axis returns view', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = swapaxes(arr, 0, 0);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('returns a view (shares data)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = swapaxes(arr, 0, 1);
      expect(result.base).toBe(arr);
    });
  });

  // ========================================
  // moveaxis
  // ========================================
  describe('moveaxis', () => {
    it('moves single axis', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, 0, -1);
      expect(result.shape).toEqual([4, 5, 3]);
    });

    it('moves multiple axes', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, [0, 1], [-1, -2]);
      expect(result.shape).toEqual([5, 4, 3]);
    });

    it('handles negative axis', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, -1, 0);
      expect(result.shape).toEqual([5, 3, 4]);
    });

    it('returns a view', () => {
      const arr = zeros([3, 4, 5]);
      const result = moveaxis(arr, 0, -1);
      expect(result.base).toBe(arr);
    });
  });

  // ========================================
  // concatenate
  // ========================================
  describe('concatenate', () => {
    it('concatenates 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concatenate([a, b]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('concatenates 2D arrays along axis 0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 0);
      expect(result.shape).toEqual([4, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
    });

    it('concatenates 2D arrays along axis 1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 1);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
      ]);
    });

    it('handles negative axis', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], -1);
      expect(result.shape).toEqual([2, 4]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([[5, 6, 7]]);
      expect(() => concatenate([a, b], 0)).toThrow();
    });
  });

  // ========================================
  // stack
  // ========================================
  describe('stack', () => {
    it('stacks 1D arrays along axis 0', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 0);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('stacks 1D arrays along axis 1', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('stacks 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 0);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
    });

    it('handles negative axis', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], -1);
      expect(result.shape).toEqual([3, 2]);
    });

    it('throws error for different shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      expect(() => stack([a, b])).toThrow();
    });
  });

  // ========================================
  // vstack
  // ========================================
  describe('vstack', () => {
    it('stacks 1D arrays vertically', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vstack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('stacks 2D arrays vertically', () => {
      const a = array([[1, 2, 3]]);
      const b = array([[4, 5, 6]]);
      const result = vstack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });
  });

  // ========================================
  // hstack
  // ========================================
  describe('hstack', () => {
    it('stacks 1D arrays horizontally', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = hstack([a, b]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('stacks 2D arrays horizontally', () => {
      const a = array([[1], [2]]);
      const b = array([[3], [4]]);
      const result = hstack([a, b]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });
  });

  // ========================================
  // dstack
  // ========================================
  describe('dstack', () => {
    it('stacks 1D arrays depth-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = dstack([a, b]);
      expect(result.shape).toEqual([1, 3, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 4],
          [2, 5],
          [3, 6],
        ],
      ]);
    });

    it('stacks 2D arrays depth-wise', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = dstack([a, b]);
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.toArray()).toEqual([
        [
          [1, 5],
          [2, 6],
        ],
        [
          [3, 7],
          [4, 8],
        ],
      ]);
    });
  });

  // ========================================
  // split
  // ========================================
  describe('split', () => {
    it('splits 1D array into equal parts', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits at specified indices', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, [2, 4]);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits 2D array along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = split(arr, 2, 0);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [5, 6],
        [7, 8],
      ]);
    });

    it('throws error for unequal division', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(() => split(arr, 2)).toThrow();
    });

    it('returns views (shares data)', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);
      expect(result[0]!.base).toBe(arr);
    });
  });

  // ========================================
  // array_split
  // ========================================
  describe('array_split', () => {
    it('splits with unequal parts', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = array_split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5]);
    });

    it('handles equal division', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = array_split(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });
  });

  // ========================================
  // vsplit
  // ========================================
  describe('vsplit', () => {
    it('splits 2D array vertically', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = vsplit(arr, 2);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [5, 6],
        [7, 8],
      ]);
    });

    it('throws error for 1D array', () => {
      const arr = array([1, 2, 3, 4]);
      expect(() => vsplit(arr, 2)).toThrow();
    });
  });

  // ========================================
  // hsplit
  // ========================================
  describe('hsplit', () => {
    it('splits 1D array horizontally', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = hsplit(arr, 3);
      expect(result.length).toBe(3);
      expect(result[0]!.toArray()).toEqual([1, 2]);
      expect(result[1]!.toArray()).toEqual([3, 4]);
      expect(result[2]!.toArray()).toEqual([5, 6]);
    });

    it('splits 2D array horizontally', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = hsplit(arr, 2);
      expect(result.length).toBe(2);
      expect(result[0]!.toArray()).toEqual([
        [1, 2],
        [5, 6],
      ]);
      expect(result[1]!.toArray()).toEqual([
        [3, 4],
        [7, 8],
      ]);
    });
  });

  // ========================================
  // tile
  // ========================================
  describe('tile', () => {
    it('tiles 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = tile(arr, 3);
      expect(result.shape).toEqual([9]);
      expect(result.toArray()).toEqual([1, 2, 3, 1, 2, 3, 1, 2, 3]);
    });

    it('tiles 1D array with 2D reps', () => {
      const arr = array([1, 2]);
      const result = tile(arr, [2, 3]);
      expect(result.shape).toEqual([2, 6]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
      ]);
    });

    it('tiles 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, [2, 2]);
      expect(result.shape).toEqual([4, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4],
      ]);
    });

    it('handles single rep', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, 2);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
      ]);
    });
  });

  // ========================================
  // repeat
  // ========================================
  describe('repeat', () => {
    it('repeats elements of 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, 2);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it('repeats elements of 2D array (flattens)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2);
      expect(result.shape).toEqual([8]);
      expect(result.toArray()).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });

    it('repeats along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 0);
      expect(result.shape).toEqual([4, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
      ]);
    });

    it('repeats along axis 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 1);
      expect(result.shape).toEqual([2, 4]);
      expect(result.toArray()).toEqual([
        [1, 1, 2, 2],
        [3, 3, 4, 4],
      ]);
    });

    it('repeats with different counts per element', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, [1, 2, 3]);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 2, 3, 3, 3]);
    });
  });

  // ========================================
  // New Manipulation Functions
  // ========================================

  describe('ravel', () => {
    it('flattens 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = ravel(arr);
      expect(result.shape).toEqual([6]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });
  });

  describe('reshape', () => {
    it('reshapes array to new shape', () => {
      const arr = arange(6);
      const result = reshape(arr, [2, 3]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [0, 1, 2],
        [3, 4, 5],
      ]);
    });
  });

  describe('squeeze', () => {
    it('removes size-1 dimensions', () => {
      const arr = zeros([1, 3, 1]);
      const result = squeeze(arr);
      expect(result.shape).toEqual([3]);
    });

    it('removes specific size-1 dimension', () => {
      const arr = zeros([1, 3, 1]);
      const result = squeeze(arr, 0);
      expect(result.shape).toEqual([3, 1]);
    });
  });

  describe('expand_dims', () => {
    it('adds new axis at position 0', () => {
      const arr = array([1, 2, 3]);
      const result = expand_dims(arr, 0);
      expect(result.shape).toEqual([1, 3]);
    });

    it('adds new axis at position 1', () => {
      const arr = array([1, 2, 3]);
      const result = expand_dims(arr, 1);
      expect(result.shape).toEqual([3, 1]);
    });
  });

  describe('flip', () => {
    it('flips 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = flip(arr);
      expect(result.toArray()).toEqual([3, 2, 1]);
    });

    it('flips 2D array along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = flip(arr, 0);
      expect(result.toArray()).toEqual([
        [3, 4],
        [1, 2],
      ]);
    });

    it('flips 2D array along axis 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = flip(arr, 1);
      expect(result.toArray()).toEqual([
        [2, 1],
        [4, 3],
      ]);
    });
  });

  describe('fliplr', () => {
    it('flips 2D array left-right', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = fliplr(arr);
      expect(result.toArray()).toEqual([
        [3, 2, 1],
        [6, 5, 4],
      ]);
    });
  });

  describe('flipud', () => {
    it('flips 2D array up-down', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = flipud(arr);
      expect(result.toArray()).toEqual([
        [5, 6],
        [3, 4],
        [1, 2],
      ]);
    });
  });

  describe('rot90', () => {
    it('rotates array 90 degrees', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(arr);
      expect(result.toArray()).toEqual([
        [2, 4],
        [1, 3],
      ]);
    });

    it('rotates array 180 degrees', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(arr, 2);
      expect(result.toArray()).toEqual([
        [4, 3],
        [2, 1],
      ]);
    });
  });

  describe('roll', () => {
    it('rolls 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = roll(arr, 2);
      expect(result.toArray()).toEqual([4, 5, 1, 2, 3]);
    });

    it('rolls negative shift', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = roll(arr, -2);
      expect(result.toArray()).toEqual([3, 4, 5, 1, 2]);
    });

    it('rolls along axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = roll(arr, 1, 1);
      expect(result.toArray()).toEqual([
        [3, 1, 2],
        [6, 4, 5],
      ]);
    });
  });

  describe('rollaxis', () => {
    it('rolls axis to new position', () => {
      const arr = zeros([3, 4, 5]);
      const result = rollaxis(arr, 2, 0);
      expect(result.shape).toEqual([5, 3, 4]);
    });
  });

  describe('atleast_1d', () => {
    it('ensures array has at least 1 dimension', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_1d(arr);
      expect((result as any).shape).toEqual([3]);
    });
  });

  describe('atleast_2d', () => {
    it('converts 1D to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_2d(arr);
      expect((result as any).shape).toEqual([1, 3]);
    });

    it('keeps 2D unchanged', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = atleast_2d(arr);
      expect((result as any).shape).toEqual([2, 2]);
    });
  });

  describe('atleast_3d', () => {
    it('converts 1D to 3D', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_3d(arr);
      expect((result as any).shape).toEqual([1, 3, 1]);
    });

    it('converts 2D to 3D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = atleast_3d(arr);
      expect((result as any).shape).toEqual([2, 2, 1]);
    });
  });

  describe('dsplit', () => {
    it('splits 3D array along depth', () => {
      const arr = zeros([2, 2, 4]);
      const result = dsplit(arr, 2);
      expect(result.length).toBe(2);
      expect(result[0]!.shape).toEqual([2, 2, 2]);
    });
  });

  describe('column_stack', () => {
    it('stacks 1D arrays as columns', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = column_stack([a, b]);
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });
  });

  describe('row_stack', () => {
    it('stacks arrays vertically (alias for vstack)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = row_stack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });
  });

  describe('resize', () => {
    it('resizes to larger shape', () => {
      const arr = array([1, 2, 3]);
      const result = resize(arr, [6]);
      expect(result.toArray()).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it('resizes to smaller shape', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = resize(arr, [3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('resizes to different dimensions', () => {
      const arr = array([1, 2, 3]);
      const result = resize(arr, [2, 3]);
      expect(result.shape).toEqual([2, 3]);
    });
  });

  describe('append', () => {
    it('appends values to 1D array', () => {
      const arr = array([1, 2, 3]);
      const values = array([4, 5]);
      const result = append(arr, values);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('appends along axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const values = array([[5, 6]]);
      const result = append(arr, values, 0);
      expect(result.shape).toEqual([3, 2]);
    });
  });

  describe('delete_', () => {
    it('deletes element from 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = delete_(arr, 2);
      expect(result.toArray()).toEqual([1, 2, 4, 5]);
    });

    it('deletes multiple elements', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = delete_(arr, [1, 3]);
      expect(result.toArray()).toEqual([1, 3, 5]);
    });

    it('deletes along axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = delete_(arr, 1, 0);
      expect(result.shape).toEqual([2, 3]);
    });
  });

  describe('insert', () => {
    it('inserts value into 1D array', () => {
      const arr = array([1, 2, 4, 5]);
      const values = array([3]);
      const result = insert(arr, 2, values);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('inserts along axis (throws - not fully implemented in standalone)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const values = array([[5, 6]]);
      // insert along axis is not fully implemented in the standalone module
      expect(() => insert(arr, 1, values, 0)).toThrow('not fully implemented');
    });
  });

  describe('pad', () => {
    it('pads array with constant values', () => {
      const arr = array([1, 2, 3]);
      const result = pad(arr, 2);
      expect(result.toArray()).toEqual([0, 0, 1, 2, 3, 0, 0]);
    });

    it('pads 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = pad(arr, 1);
      expect(result.shape).toEqual([4, 4]);
    });

    it('pads with custom constant value', () => {
      const arr = array([1, 2, 3]);
      const result = pad(arr, 1, 'constant', 9);
      expect(result.toArray()).toEqual([9, 1, 2, 3, 9]);
    });

    it('pads with edge mode (throws - not fully implemented in standalone)', () => {
      const arr = array([1, 2, 3]);
      // pad edge mode is not fully implemented in the standalone module
      expect(() => pad(arr, 2, 'edge')).toThrow('not fully implemented');
    });
  });

  // ========================================
  // flatten (standalone function)
  // ========================================
  describe('flatten', () => {
    it('flattens 3D array', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(flatten(a).size).toBe(8);
    });
  });

  // ========================================
  // broadcast_to (shape manipulation)
  // ========================================
  describe('broadcast_to (from coverage gaps)', () => {
    it('broadcasts 1D to 2D', () => {
      const a = array([1, 2, 3]);
      expect(broadcast_to(a, [2, 3]).shape).toEqual([2, 3]);
    });
  });

  // ========================================
  // roll with multiple axes/shifts
  // ========================================
  describe('roll (advanced)', () => {
    it('rolls 2D array with array of shifts and axes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(roll(a, [1, 1], [0, 1]).size).toBe(4);
    });
  });
});
