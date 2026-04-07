/**
 * DType Sweep: Shape manipulation functions.
 * Structural tests — validates shape/dtype preservation across ALL dtypes.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Shape manipulation', () => {
  for (const dtype of ALL_DTYPES) {
    const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : SMALL_DATA;
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1],
            [0, 1, 0],
          ]
        : SMALL_2D;

    it(`reshape ${dtype}`, () => {
      expect(np.reshape(array(data, dtype), [2, 3]).shape).toEqual([2, 3]);
    });

    it(`transpose ${dtype}`, () => {
      expect(np.transpose(array(data2d, dtype)).shape).toEqual([3, 2]);
    });

    it(`ravel ${dtype}`, () => {
      expect(np.ravel(array(data2d, dtype)).shape).toEqual([6]);
    });

    it(`concatenate ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      const b = array(dtype === 'bool' ? [0, 1] : [3, 4], dtype);
      const r = np.concatenate([a, b]);
      expect(r.shape).toEqual([4]);
      expect(r.dtype).toBe(dtype);
    });

    it(`stack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      const b = array(dtype === 'bool' ? [0, 1] : [3, 4], dtype);
      expect(np.stack([a, b]).shape).toEqual([2, 2]);
    });

    it(`tile ${dtype}`, () => {
      expect(np.tile(array(dtype === 'bool' ? [1, 0] : [1, 2], dtype), 3).shape).toEqual([6]);
    });

    it(`repeat ${dtype}`, () => {
      expect(np.repeat(array(dtype === 'bool' ? [1, 0] : [1, 2], dtype), 2).shape).toEqual([4]);
    });

    it(`roll ${dtype}`, () => {
      const d = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.roll(array(d, dtype), 1);
      const pyResult = runNumPy(
        `result = np.roll(np.array(${JSON.stringify(d)}, dtype=${npDtype(dtype)}), 1).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`flip ${dtype}`, () => {
      const d = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.flip(array(d, dtype));
      const pyResult = runNumPy(
        `result = np.flip(np.array(${JSON.stringify(d)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`squeeze ${dtype}`, () => {
      expect(np.squeeze(array(dtype === 'bool' ? [[1, 0]] : [[1, 2]], dtype)).shape).toEqual([2]);
    });

    it(`expand_dims ${dtype}`, () => {
      expect(np.expand_dims(array(dtype === 'bool' ? [1] : [1], dtype), 0).shape).toEqual([1, 1]);
    });

    it(`hstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.hstack([a, a]).shape).toEqual([4]);
    });

    it(`vstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.vstack([a, a]).shape).toEqual([2, 2]);
    });

    it(`dstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.dstack([a, a]).shape).toEqual([1, 2, 2]);
    });

    it(`column_stack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.column_stack([a, a]).shape).toEqual([2, 2]);
    });

    it(`append ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.append(a, a).shape).toEqual([4]);
    });

    it(`atleast_1d ${dtype}`, () => {
      expect(np.atleast_1d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(1);
    });

    it(`atleast_2d ${dtype}`, () => {
      expect(np.atleast_2d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(2);
    });

    it(`atleast_3d ${dtype}`, () => {
      expect(np.atleast_3d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(3);
    });

    it(`broadcast_to ${dtype}`, () => {
      expect(np.broadcast_to(array(dtype === 'bool' ? [1] : [1], dtype), [3]).shape).toEqual([3]);
    });

    it(`broadcast_arrays ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1] : [1], dtype);
      const b = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      const [ra, rb] = np.broadcast_arrays(a, b) as any[];
      expect(ra.shape).toEqual([3]);
      expect(rb.shape).toEqual([3]);
    });

    it(`split ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
      expect(np.split(a, 2).length).toBe(2);
    });

    it(`hsplit ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
      expect(np.hsplit(a, 2).length).toBe(2);
    });

    it(`vsplit ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]], dtype);
      expect(np.vsplit(a, 2).length).toBe(2);
    });

    it(`dsplit ${dtype}`, () => {
      const a = np.reshape(array(dtype === 'bool' ? [1, 0, 1, 0, 1, 0, 1, 0] : [1, 2, 3, 4, 5, 6, 7, 8], dtype), [2, 2, 2]);
      expect(np.dsplit(a, 2).length).toBe(2);
    });

    it(`array_split ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.array_split(a, 2).length).toBe(2);
    });

    it(`fliplr ${dtype}`, () => {
      expect(np.fliplr(array(data2d, dtype)).shape).toEqual([2, 3]);
    });

    it(`flipud ${dtype}`, () => {
      expect(np.flipud(array(data2d, dtype)).shape).toEqual([2, 3]);
    });

    it(`rot90 ${dtype}`, () => {
      const jsResult = np.rot90(array(data2d, dtype));
      const pyResult = runNumPy(
        `result = np.rot90(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`moveaxis ${dtype}`, () => {
      expect(np.moveaxis(array(data2d, dtype), 0, 1).shape).toEqual([3, 2]);
    });

    it(`swapaxes ${dtype}`, () => {
      expect(np.swapaxes(array(data2d, dtype), 0, 1).shape).toEqual([3, 2]);
    });

    it(`insert ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : isComplex(dtype) ? [1, 2, 3] : [1, 2, 3], dtype);
      const jsResult = np.insert(a, 1, dtype === 'bool' ? 0 : 99);
      expect(jsResult.shape).toEqual([4]);
    });

    it(`delete_ ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.delete_(a, 1).shape).toEqual([2]);
    });

    it(`resize ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.resize(a, [4]).shape).toEqual([4]);
    });

    it(`unstack ${dtype}`, () => {
      const a = array(data2d, dtype);
      const result = np.unstack(a);
      expect(result.length).toBe(2);
      expect(result[0]!.shape).toEqual([3]);
    });

    it(`diagflat ${dtype}`, () => {
      const jsResult = np.diagflat(array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype));
      expect(jsResult.shape).toEqual([2, 2]);
    });

    it(`flatten ${dtype}`, () => {
      expect(np.flatten(array(data2d, dtype)).shape).toEqual([6]);
    });

    it(`pad ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.pad(a, 2).shape).toEqual([7]);
    });

    it(`trim_zeros ${dtype}`, () => {
      const data0 = dtype === 'bool' ? [0, 1, 0] : [0, 1, 2, 0];
      const jsResult = np.trim_zeros(array(data0, dtype));
      const expected = dtype === 'bool' ? [1] : [1, 2];
      expect(jsResult.shape).toEqual([expected.length]);
    });

    it(`compress ${dtype}`, () => {
      const a = array(
        dtype === 'bool' ? [1, 0, 1] : isComplex(dtype) ? [1, 2, 3] : [1, 2, 3],
        dtype
      );
      const cond = array([1, 0, 1], 'bool');
      expect(np.compress(cond, a).shape).toEqual([2]);
    });

    it(`select ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      const cond = array([1, 0, 1], 'bool');
      const result = np.select([cond], [a]);
      expect(result.shape).toEqual([3]);
    });

    it(`diag ${dtype}`, () => {
      expect(np.diag(array(data2d, dtype)).shape).toEqual([2]);
    });

    it(`diagonal ${dtype}`, () => {
      expect(np.diagonal(array(data2d, dtype)).shape).toEqual([2]);
    });

    it(`fill_diagonal ${dtype}`, () => {
      const a = array(
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        dtype
      );
      np.fill_diagonal(a, dtype === 'bool' ? 1 : 9);
      expect(a.shape).toEqual([3, 3]);
    });
  }
});
