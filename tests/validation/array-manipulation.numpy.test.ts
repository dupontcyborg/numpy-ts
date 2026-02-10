/**
 * Python NumPy validation tests for array manipulation operations
 *
 * Tests: swapaxes, moveaxis, concatenate, stack, vstack, hstack, dstack,
 *        split, array_split, vsplit, hsplit, tile, repeat
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  arange,
  zeros,
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
  flip,
  fliplr,
  flipud,
  rot90,
  roll,
  rollaxis,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  expand_dims,
  squeeze,
  ravel,
  reshape,
  append,
  delete_,
  insert,
  resize,
  pad,
  column_stack,
  row_stack,
  dsplit,
  concat,
  unstack,
  block,
  flatten,
  fill,
  item,
  tolist,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Array Manipulation', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  // ========================================
  // swapaxes
  // ========================================
  describe('swapaxes()', () => {
    it('validates swapaxes 2D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, 0, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.swapaxes(arr, 0, 1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates swapaxes 3D', () => {
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
      const result = swapaxes(arr, 0, 2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.swapaxes(arr, 0, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates swapaxes with negative axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, -2, -1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.swapaxes(arr, -2, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // moveaxis
  // ========================================
  describe('moveaxis()', () => {
    it('validates moveaxis single axis', () => {
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
      const result = moveaxis(arr, 0, -1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.moveaxis(arr, 0, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates moveaxis multiple axes', () => {
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
      const result = moveaxis(arr, [0, 1], [-1, -2]);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.moveaxis(arr, [0, 1], [-1, -2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // concatenate
  // ========================================
  describe('concatenate()', () => {
    it('validates concatenate 1D', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concatenate([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.concatenate([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate 2D axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 0);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concatenate([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate 2D axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 1);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concatenate([a, b], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate multiple arrays', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const c = array([5, 6]);
      const result = concatenate([a, b, c]);

      const npResult = runNumPy(`
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
result = np.concatenate([a, b, c])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // stack
  // ========================================
  describe('stack()', () => {
    it('validates stack 1D arrays axis=0', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 0);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates stack 1D arrays axis=1', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 1);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack([a, b], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates stack 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 0);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.stack([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // vstack
  // ========================================
  describe('vstack()', () => {
    it('validates vstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.vstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates vstack 2D arrays', () => {
      const a = array([[1, 2, 3]]);
      const b = array([[4, 5, 6]]);
      const result = vstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])
result = np.vstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // hstack
  // ========================================
  describe('hstack()', () => {
    it('validates hstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = hstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.hstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates hstack 2D arrays', () => {
      const a = array([[1], [2]]);
      const b = array([[3], [4]]);
      const result = hstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1], [2]])
b = np.array([[3], [4]])
result = np.hstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // dstack
  // ========================================
  describe('dstack()', () => {
    it('validates dstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = dstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates dstack 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = dstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.dstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // split
  // ========================================
  describe('split()', () => {
    it('validates split into equal parts', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates split at indices', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, [2, 4]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, [2, 4])
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates split 2D along axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = split(arr, 2, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = np.split(arr, 2, axis=0)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // array_split
  // ========================================
  describe('array_split()', () => {
    it('validates array_split unequal parts', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = array_split(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.array_split(arr, 3)
result = [r.tolist() for r in result]
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });
  });

  // ========================================
  // vsplit
  // ========================================
  describe('vsplit()', () => {
    it('validates vsplit 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = vsplit(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = np.vsplit(arr, 2)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // hsplit
  // ========================================
  describe('hsplit()', () => {
    it('validates hsplit 1D array', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = hsplit(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.hsplit(arr, 3)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates hsplit 2D array', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = hsplit(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.hsplit(arr, 2)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // tile
  // ========================================
  describe('tile()', () => {
    it('validates tile 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = tile(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.tile(arr, 3)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates tile 1D with 2D reps', () => {
      const arr = array([1, 2]);
      const result = tile(arr, [2, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2])
result = np.tile(arr, (2, 3))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates tile 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, [2, 2]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.tile(arr, (2, 2))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // repeat
  // ========================================
  describe('repeat()', () => {
    it('validates repeat 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, 2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.repeat(arr, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat 2D array (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat along axis 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat with different counts', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, [1, 2, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.repeat(arr, [1, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // New manipulation functions
  // ========================================

  describe('flip()', () => {
    it('validates flip all axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = flip(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.flip(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates flip along axis 0', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = flip(arr, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.flip(arr, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates flip along axis 1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = flip(arr, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.flip(arr, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('fliplr()', () => {
    it('validates fliplr', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = fliplr(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.fliplr(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('flipud()', () => {
    it('validates flipud', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = flipud(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.flipud(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('rot90()', () => {
    it('validates rot90 k=1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.rot90(arr)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates rot90 k=2', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.rot90(arr, k=2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates rot90 k=-1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = rot90(arr, -1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.rot90(arr, k=-1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('roll()', () => {
    it('validates roll 1D', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = roll(arr, 2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.roll(arr, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates roll negative shift', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = roll(arr, -2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.roll(arr, -2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates roll 2D along axis', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = roll(arr, 1, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.roll(arr, 1, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('atleast_1d()', () => {
    it('validates scalar to 1D', () => {
      const result = atleast_1d(array(5));

      const npResult = runNumPy(`result = np.atleast_1d(5)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates 1D stays 1D', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_1d(arr);

      const npResult = runNumPy(`result = np.atleast_1d([1, 2, 3])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('atleast_2d()', () => {
    it('validates 1D to 2D', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_2d(arr);

      const npResult = runNumPy(`result = np.atleast_2d([1, 2, 3])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates 2D stays 2D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = atleast_2d(arr);

      const npResult = runNumPy(`result = np.atleast_2d([[1, 2], [3, 4]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('atleast_3d()', () => {
    it('validates 1D to 3D', () => {
      const arr = array([1, 2, 3]);
      const result = atleast_3d(arr);

      const npResult = runNumPy(`result = np.atleast_3d([1, 2, 3])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates 2D to 3D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = atleast_3d(arr);

      const npResult = runNumPy(`result = np.atleast_3d([[1, 2], [3, 4]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('expand_dims()', () => {
    it('validates expand_dims axis=0', () => {
      const arr = array([1, 2, 3]);
      const result = expand_dims(arr, 0);

      const npResult = runNumPy(`result = np.expand_dims([1, 2, 3], axis=0)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates expand_dims axis=1', () => {
      const arr = array([1, 2, 3]);
      const result = expand_dims(arr, 1);

      const npResult = runNumPy(`result = np.expand_dims([1, 2, 3], axis=1)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates expand_dims negative axis', () => {
      const arr = array([1, 2, 3]);
      const result = expand_dims(arr, -1);

      const npResult = runNumPy(`result = np.expand_dims([1, 2, 3], axis=-1)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('squeeze()', () => {
    it('validates squeeze all axes', () => {
      const arr = zeros([1, 3, 1, 4, 1]);
      const result = squeeze(arr);

      const npResult = runNumPy(`result = np.squeeze(np.zeros((1, 3, 1, 4, 1)))`);

      expect(result.shape).toEqual(npResult.shape);
    });

    it('validates squeeze specific axis', () => {
      const arr = zeros([1, 3, 1, 4]);
      const result = squeeze(arr, 0);

      const npResult = runNumPy(`result = np.squeeze(np.zeros((1, 3, 1, 4)), axis=0)`);

      expect(result.shape).toEqual(npResult.shape);
    });
  });

  describe('ravel()', () => {
    it('validates ravel 2D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = ravel(arr);

      const npResult = runNumPy(`result = np.ravel([[1, 2, 3], [4, 5, 6]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('reshape()', () => {
    it('validates reshape 1D to 2D', () => {
      const arr = arange(0, 6);
      const result = reshape(arr, [2, 3]);

      const npResult = runNumPy(`result = np.reshape(np.arange(6), (2, 3))`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates reshape with -1', () => {
      const arr = arange(0, 12);
      const result = reshape(arr, [3, -1]);

      const npResult = runNumPy(`result = np.reshape(np.arange(12), (3, -1))`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('append()', () => {
    it('validates append 1D', () => {
      const arr = array([1, 2, 3]);
      const result = append(arr, [4, 5, 6]);

      const npResult = runNumPy(`result = np.append([1, 2, 3], [4, 5, 6])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates append scalar', () => {
      const arr = array([1, 2, 3]);
      const result = append(arr, 4);

      const npResult = runNumPy(`result = np.append([1, 2, 3], 4)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('delete_()', () => {
    it('validates delete single index', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = delete_(arr, 2);

      const npResult = runNumPy(`result = np.delete([1, 2, 3, 4, 5], 2)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates delete multiple indices', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = delete_(arr, [0, 2, 4]);

      const npResult = runNumPy(`result = np.delete([1, 2, 3, 4, 5], [0, 2, 4])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('insert()', () => {
    it('validates insert single value', () => {
      const arr = array([1, 2, 3, 4]);
      const result = insert(arr, 2, 99);

      const npResult = runNumPy(`result = np.insert([1, 2, 3, 4], 2, 99)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates insert multiple values', () => {
      const arr = array([1, 2, 3, 4]);
      const result = insert(arr, 2, [97, 98, 99]);

      const npResult = runNumPy(`result = np.insert([1, 2, 3, 4], 2, [97, 98, 99])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('resize()', () => {
    it('validates resize to larger', () => {
      const arr = array([1, 2, 3]);
      const result = resize(arr, [5]);

      const npResult = runNumPy(`result = np.resize([1, 2, 3], 5)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates resize to smaller', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = resize(arr, [3]);

      const npResult = runNumPy(`result = np.resize([1, 2, 3, 4, 5], 3)`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates resize 2D', () => {
      const arr = array([1, 2, 3]);
      const result = resize(arr, [2, 3]);

      const npResult = runNumPy(`result = np.resize([1, 2, 3], (2, 3))`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('pad()', () => {
    it('validates pad constant 1D', () => {
      const arr = array([1, 2, 3]);
      const result = pad(arr, 2);

      const npResult = runNumPy(`result = np.pad([1, 2, 3], 2, mode='constant')`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates pad constant 2D', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = pad(arr, 1);

      const npResult = runNumPy(`result = np.pad([[1, 2], [3, 4]], 1, mode='constant')`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates pad with constant value', () => {
      const arr = array([1, 2, 3]);
      const result = pad(arr, 2, 'constant', 9);

      const npResult = runNumPy(
        `result = np.pad([1, 2, 3], 2, mode='constant', constant_values=9)`
      );

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('column_stack()', () => {
    it('validates column_stack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = column_stack([a, b]);

      const npResult = runNumPy(`result = np.column_stack([[1, 2, 3], [4, 5, 6]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates column_stack 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = column_stack([a, b]);

      const npResult = runNumPy(`result = np.column_stack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('row_stack()', () => {
    it('validates row_stack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = row_stack([a, b]);

      const npResult = runNumPy(`result = np.row_stack([[1, 2, 3], [4, 5, 6]])`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('dsplit()', () => {
    it('validates dsplit into equal parts', () => {
      const arr = zeros([2, 2, 4]);
      // Fill with sequential values
      for (let i = 0; i < 16; i++) {
        arr.set([Math.floor(i / 8), Math.floor((i % 8) / 4), i % 4], i);
      }
      const results = dsplit(arr, 2);

      const npResult = runNumPy(`
arr = np.arange(16).reshape(2, 2, 4)
parts = np.dsplit(arr, 2)
result = [p.tolist() for p in parts]
`);

      expect(results.length).toBe(npResult.value.length);
      for (let i = 0; i < results.length; i++) {
        expect(arraysClose(results[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });
  });

  describe('rollaxis()', () => {
    it('validates rollaxis', () => {
      const arr = zeros([3, 4, 5]);
      const result = rollaxis(arr, 2, 0);

      const npResult = runNumPy(`result = np.rollaxis(np.zeros((3, 4, 5)), 2, 0)`);

      expect(result.shape).toEqual(npResult.shape);
    });

    it('validates rollaxis default start', () => {
      const arr = zeros([3, 4, 5]);
      const result = rollaxis(arr, 1);

      const npResult = runNumPy(`result = np.rollaxis(np.zeros((3, 4, 5)), 1)`);

      expect(result.shape).toEqual(npResult.shape);
    });
  });

  // ========================================
  // concat (alias for concatenate)
  // ========================================
  describe('concat()', () => {
    it('validates concat 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concat([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.concat([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concat 2D arrays axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concat([a, b], 0);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concat([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concat 2D arrays axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concat([a, b], 1);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concat([a, b], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // unstack
  // ========================================
  describe('unstack()', () => {
    it('validates unstack 2D along axis=0', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const results = unstack(arr, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = [a.tolist() for a in np.unstack(arr, axis=0)]
`);

      expect(results.length).toBe(npResult.value.length);
      for (let i = 0; i < results.length; i++) {
        expect(arraysClose(results[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });

    it('validates unstack 2D along axis=1', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const results = unstack(arr, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = [a.tolist() for a in np.unstack(arr, axis=1)]
`);

      expect(results.length).toBe(npResult.value.length);
      for (let i = 0; i < results.length; i++) {
        expect(arraysClose(results[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });

    it('validates unstack 3D', () => {
      const arr = arange(0, 24).reshape([2, 3, 4]);
      const results = unstack(arr, 0);

      const npResult = runNumPy(`
arr = np.arange(24).reshape(2, 3, 4)
result = [a.tolist() for a in np.unstack(arr, axis=0)]
`);

      expect(results.length).toBe(npResult.value.length);
      for (let i = 0; i < results.length; i++) {
        expect(arraysClose(results[i]!.toArray(), npResult.value[i])).toBe(true);
      }
    });
  });

  // ========================================
  // block
  // ========================================
  describe('block()', () => {
    it('validates block with 1D arrays', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const result = block([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2])
b = np.array([3, 4])
result = np.block([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates block with 2D arrays horizontal', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = block([a, b]);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.block([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // flatten
  // ========================================
  describe('flatten()', () => {
    it('validates flatten 2D to 1D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = flatten(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr.flatten()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates flatten 3D to 1D', () => {
      const arr = arange(0, 24).reshape([2, 3, 4]);
      const result = flatten(arr);

      const npResult = runNumPy(`
arr = np.arange(24).reshape(2, 3, 4)
result = arr.flatten()
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // fill
  // ========================================
  describe('fill()', () => {
    it('validates fill with scalar', () => {
      const arr = zeros([3, 3]);
      fill(arr, 7);

      const npResult = runNumPy(`
arr = np.zeros((3, 3))
arr.fill(7)
result = arr
`);

      expect(arr.shape).toEqual(npResult.shape);
      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });

    it('validates fill with negative', () => {
      const arr = zeros([2, 4]);
      fill(arr, -5.5);

      const npResult = runNumPy(`
arr = np.zeros((2, 4))
arr.fill(-5.5)
result = arr
`);

      expect(arr.shape).toEqual(npResult.shape);
      expect(arraysClose(arr.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // item
  // ========================================
  describe('item()', () => {
    it('validates item from scalar-like', () => {
      const arr = array([42]);
      const result = item(arr);

      const npResult = runNumPy(`result = np.array([42]).item()`);

      expect(result).toBe(npResult.value);
    });

    it('validates item with index', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = item(arr, 2);

      const npResult = runNumPy(`result = np.array([1, 2, 3, 4, 5]).item(2)`);

      expect(result).toBe(npResult.value);
    });

    it('validates item from 2D with flat index', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = item(arr, 4);

      const npResult = runNumPy(`result = np.array([[1, 2, 3], [4, 5, 6]]).item(4)`);

      expect(result).toBe(npResult.value);
    });
  });

  // ========================================
  // tolist
  // ========================================
  describe('tolist()', () => {
    it('validates tolist 1D', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = tolist(arr);

      const npResult = runNumPy(`result = np.array([1, 2, 3, 4, 5]).tolist()`);

      expect(result).toEqual(npResult.value);
    });

    it('validates tolist 2D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = tolist(arr);

      const npResult = runNumPy(`result = np.array([[1, 2, 3], [4, 5, 6]]).tolist()`);

      expect(result).toEqual(npResult.value);
    });

    it('validates tolist 3D', () => {
      const arr = arange(0, 8).reshape([2, 2, 2]);
      const result = tolist(arr);

      const npResult = runNumPy(`result = np.arange(8).reshape(2, 2, 2).tolist()`);

      expect(result).toEqual(npResult.value);
    });
  });
});
